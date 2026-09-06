"""MQTT client for OwnTracks + ESP32 communication.

Subscribes to:
  - owntracks/shereef/#          — geofence enter/leave events
  - espresense/devices/+/smart_room — BLE presence + RSSI
  - espresense/rooms/smart_room/#   — node status + telemetry

Publishes to:
  - smart_room/state             — state snapshot (retained)
  - smart_room/command            — commands from Marvi (optional, bridge handles this)

The MQTT client is the primary communication channel between the runtime
and the outside world (ESP32, OwnTracks, Marvi).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import paho.mqtt.client as mqtt
    HAS_MQTT = True
except ImportError:
    HAS_MQTT = False
    mqtt = None  # type: ignore


class MQTTClient:
    """MQTT client for smart_room runtime."""

    def __init__(
        self,
        config: Dict[str, Any],
        on_presence: Callable[[bool, Optional[int], Optional[str]], None],
        on_geofence: Callable[[str, str], None],
        on_command: Callable[[Dict[str, Any]], None],
        on_node_status: Optional[Callable[[bool, Optional[str]], None]] = None,
        on_owntracks: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self._config = config
        self._on_presence = on_presence
        self._on_geofence = on_geofence
        self._on_command = on_command
        self._on_node_status = on_node_status or (lambda _online, _ip: None)
        self._on_owntracks = on_owntracks or (lambda _topic, _payload: None)
        self._client: Optional[Any] = None
        self._connected = False
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._active_identities: set[str] = set()
        self._entry_seen_at: Dict[str, float] = {}

        mqtt_cfg = config.get("mqtt", {})
        self._broker = mqtt_cfg.get("broker", "127.0.0.1")
        self._port = mqtt_cfg.get("port", 1883)

        # Topics
        ot_cfg = config.get("owntracks", {})
        self._owntracks_topic = ot_cfg.get("topic", "owntracks/shereef/#")
        #: The device half of the OwnTracks topic, for the `cmd` topic -- which
        #: is per device where the subscription is per user.
        self._owntracks_device = str(ot_cfg.get("device", "iphone")).strip() or "iphone"
        esp_cfg = config.get("esp32", {})
        self._room = str(esp_cfg.get("room_id", "smart_room")).strip().lower()
        self._owner_device_id = str(esp_cfg.get("owner_device_id", "")).strip().lower()
        self._espresense_topic = esp_cfg.get(
            "presence_topic", f"espresense/devices/+/{self._room}"
        )
        self._espresense_status_topic = esp_cfg.get(
            "status_topic", f"espresense/rooms/{self._room}/#"
        )
        self._state_topic = "smart_room/state"
        self._command_topic = "smart_room/command"
        self._commands_enabled = bool(mqtt_cfg.get("commands_enabled", False))

    def start(self) -> None:
        if not HAS_MQTT:
            logger.warning("paho-mqtt not installed — MQTT disabled")
            return

        if hasattr(mqtt, "CallbackAPIVersion"):
            self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id="smart_room_runtime")
        else:
            self._client = mqtt.Client(client_id="smart_room_runtime")
        username = os.getenv("SMART_ROOM_MQTT_USERNAME", "")
        password = os.getenv("SMART_ROOM_MQTT_PASSWORD", "")
        if username:
            self._client.username_pw_set(username, password)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect
        if hasattr(self._client, "reconnect_delay_set"):
            self._client.reconnect_delay_set(min_delay=1, max_delay=30)

        # Start in background thread
        self._stop.clear()
        self._thread = threading.Thread(target=self._connect_loop, daemon=True, name="smart_room_mqtt")
        self._thread.start()
        logger.info("MQTT client starting (broker=%s:%s)", self._broker, self._port)

    def stop(self) -> None:
        self._stop.set()
        if self._client:
            try:
                self._client.disconnect()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=3)

    def _connect_loop(self) -> None:
        """Connect and reconnect loop."""
        delay = 1
        while not self._stop.is_set():
            try:
                self._client.connect(self._broker, self._port, 60)
                self._client.loop_start()
                self._stop.wait()
                return
            except Exception as e:
                logger.warning("MQTT connect failed: %s — retrying in %ss", e, delay)
                self._stop.wait(delay)
                delay = min(30, delay * 2)

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            self._connected = True
            logger.info("MQTT connected")
            topics = [
                (self._owntracks_topic, 0),
                (self._espresense_topic, 0),
                (self._espresense_status_topic, 0),
            ]
            if self._commands_enabled:
                topics.append((self._command_topic, 0))
            client.subscribe(topics)
        else:
            logger.error("MQTT connect failed (rc=%s)", rc)

    def _on_disconnect(self, client, userdata, rc) -> None:
        self._connected = False
        if rc != 0:
            logger.warning("MQTT disconnected unexpectedly (rc=%s) — will reconnect", rc)

    def _on_message(self, client, userdata, msg) -> None:
        """Route incoming MQTT messages."""
        topic = msg.topic
        raw = msg.payload.decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {"value": raw}

        # One bad message must not take the client down with it.
        #
        # paho runs these callbacks on its own network thread, and an exception
        # that escapes here kills that thread -- which unsubscribes everything
        # at once. That is not hypothetical: a `PermissionError` from an
        # ordinary Windows file lock, raised inside the ESPresense handler,
        # stopped OwnTracks reports arriving for a fortnight. Nothing in the
        # room reported a fault, because from the room's point of view nothing
        # had failed; the socket was simply gone.
        try:
            # OwnTracks geofence events
            if "owntracks" in topic:
                self._handle_owntracks(payload, topic)
            # ESPresense BLE presence
            elif "espresense" in topic:
                self._handle_espresense(topic, payload)
            # Commands from Marvi
            elif self._commands_enabled and topic == self._command_topic:
                self._on_command(payload)
        except Exception:
            logger.exception("Dropped an MQTT message on %s; the client stays up", topic)

    def _handle_owntracks(self, payload: Dict[str, Any], topic: str = "") -> None:
        """Process OwnTracks geofence enter/leave events."""
        self._on_owntracks(topic, payload)
        event_type = payload.get("_type", "")
        if event_type == "transition":
            event = payload.get("event", "")
            zone = str(payload.get("desc", "unknown")).strip().lower()
            if event == "enter":
                self._on_geofence("enter", zone)
            elif event == "leave":
                self._on_geofence("leave", zone)
        elif event_type == "location":
            # `inregions` is level-triggered and self-healing, which is the
            # whole reason to read it: a `transition` fires once when you cross
            # a boundary, so a single missed one leaves the last edge standing
            # forever. A location report says where you are *now*.
            #
            # An empty list is the case that was being thrown away, and it is
            # not an absence of information -- it is the statement "in no
            # region at all". Discarding it meant leaving home without a
            # transition left the state reading `home` until something else
            # corrected it, and nothing else does.
            regions = payload.get("inregions")
            if isinstance(regions, list):
                self._on_geofence("sync", str(regions[0]).strip().lower() if regions else "")

    def _handle_espresense(self, topic: str, payload: Dict[str, Any]) -> None:
        """Process ESPresense BLE presence data."""
        parts = topic.split("/")
        if len(parts) == 4 and parts[:2] == ["espresense", "rooms"]:
            room, message_type = parts[2].lower(), parts[3].lower()
            if room != self._room:
                return
            if message_type == "status":
                self._on_node_status(
                    str(payload.get("value", "")).strip().lower() == "online", None
                )
            elif message_type == "telemetry":
                self._on_node_status(True, payload.get("ip"))
            return

        # ESPresense v4 publishes devices as espresense/devices/<id>/<room>.
        if len(parts) != 4 or parts[:2] != ["espresense", "devices"]:
            return
        identity = parts[2].strip().lower()
        if parts[3].strip().lower() != self._room:
            return
        # Generic Apple fingerprints are not unique. Until secure enrollment
        # supplies the owner's IRK-backed ID, they must never drive identity.
        if not self._owner_device_id or identity != self._owner_device_id:
            return

        rssi = payload.get("rssi")
        if not isinstance(rssi, (int, float)):
            return
        rssi = int(round(rssi))

        # If rssi is strong enough, presence is detected
        threshold = self._config.get("esp32", {}).get("rssi_enter_threshold", -70)
        exit_threshold = self._config.get("esp32", {}).get("rssi_exit_threshold", -85)
        was_active = identity in self._active_identities
        strong = rssi > threshold or (was_active and rssi > exit_threshold)
        if strong and not was_active:
            first = self._entry_seen_at.setdefault(identity, time.monotonic())
            debounce = max(0, float(self._config.get("esp32", {}).get("enter_debounce_seconds", 3)))
            if time.monotonic() - first < debounce:
                return
        if strong:
            self._active_identities.add(identity)
            self._entry_seen_at.pop(identity, None)
            self._on_presence(True, rssi, identity)
        else:
            self._entry_seen_at.pop(identity, None)
            self._active_identities.discard(identity)
            self._on_presence(False, rssi, identity)

    def ask_phone_to_report(self) -> bool:
        """Ask OwnTracks where the phone is, now. True when the ask went out.

        The one thing missing from a system built entirely on the phone
        volunteering: OwnTracks publishes when it feels like it -- on a
        boundary crossing, on a timer, when the OS wakes it -- so "when did we
        last hear" and "where is he" were the same question, and the answer to
        both was whatever arrived last, however old.

        `reportLocation` is a documented command on the device's own `cmd`
        topic. The phone answers with a normal `location` payload, which comes
        back through the usual path with `inregions` and the battery on it, so
        nothing downstream needs to know this was asked for rather than
        offered.
        """
        if not (self._client and self._connected):
            logger.info("Cannot ask the phone to report: MQTT is not connected")
            return False
        # `owntracks/smart_room/#` -> `owntracks/smart_room/iphone/cmd`. The
        # wildcard is stripped and the device appended, because the command
        # topic is per device and the subscription is per user.
        base = self._owntracks_topic.rstrip("/#").rstrip("/")
        topic = f"{base}/{self._owntracks_device}/cmd"
        self._client.publish(topic, json.dumps({"_type": "cmd", "action": "reportLocation"}))
        logger.info("Asked the phone to report its location on %s", topic)
        return True

    def publish_state(self, state_dict: Dict[str, Any]) -> None:
        """Publish state snapshot to MQTT (retained)."""
        if self._client and self._connected:
            self._client.publish(self._state_topic, json.dumps(state_dict), retain=True)

    @property
    def connected(self) -> bool:
        return self._connected
