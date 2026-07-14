"""MQTT client for OwnTracks + ESP32 communication.

Subscribes to:
  - owntracks/shereef/#          — geofence enter/leave events
  - espresense/rooms/smart_room/# — BLE presence + RSSI

Publishes to:
  - smart_room/state             — state snapshot (retained)
  - smart_room/command            — commands from Marvi (optional, bridge handles this)

The MQTT client is the primary communication channel between the runtime
and the outside world (ESP32, OwnTracks, Marvi).
"""

from __future__ import annotations

import json
import logging
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
        on_presence: Callable[[bool, Optional[int]], None],
        on_geofence: Callable[[str, str], None],
        on_command: Callable[[Dict[str, Any]], None],
    ):
        self._config = config
        self._on_presence = on_presence
        self._on_geofence = on_geofence
        self._on_command = on_command
        self._client: Optional[Any] = None
        self._connected = False
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        mqtt_cfg = config.get("mqtt", {})
        self._broker = mqtt_cfg.get("broker", "127.0.0.1")
        self._port = mqtt_cfg.get("port", 1883)

        # Topics
        ot_cfg = config.get("owntracks", {})
        self._owntracks_topic = ot_cfg.get("topic", "owntracks/shereef/#")
        esp_cfg = config.get("esp32", {})
        self._espresense_topic = esp_cfg.get("presence_topic", "espresense/rooms/smart_room/#")
        self._state_topic = "smart_room/state"
        self._command_topic = "smart_room/command"

    def start(self) -> None:
        if not HAS_MQTT:
            logger.warning("paho-mqtt not installed — MQTT disabled")
            return

        self._client = mqtt.Client(client_id="smart_room_runtime")
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect

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
        while not self._stop.is_set():
            try:
                self._client.connect(self._broker, self._port, 60)
                self._client.loop_start()
                # Wait for stop or disconnect
                while not self._stop.is_set():
                    time.sleep(1)
            except Exception as e:
                logger.warning("MQTT connect failed: %s — retrying in 5s", e)
                self._stop.wait(5)

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            self._connected = True
            logger.info("MQTT connected")
            client.subscribe([
                (self._owntracks_topic, 0),
                (self._espresense_topic, 0),
                (self._command_topic, 0),
            ])
        else:
            logger.error("MQTT connect failed (rc=%s)", rc)

    def _on_disconnect(self, client, userdata, rc) -> None:
        self._connected = False
        if rc != 0:
            logger.warning("MQTT disconnected unexpectedly (rc=%s) — will reconnect", rc)

    def _on_message(self, client, userdata, msg) -> None:
        """Route incoming MQTT messages."""
        topic = msg.topic
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {}

        # OwnTracks geofence events
        if "owntracks" in topic:
            self._handle_owntracks(payload)
        # ESPresense BLE presence
        elif "espresense" in topic:
            self._handle_espresense(topic, payload)
        # Commands from Marvi
        elif topic == self._command_topic:
            self._on_command(payload)

    def _handle_owntracks(self, payload: Dict[str, Any]) -> None:
        """Process OwnTracks geofence enter/leave events."""
        event_type = payload.get("_type", "")
        if event_type == "transition":
            event = payload.get("event", "")
            zone = payload.get("desc", "unknown")
            if event == "enter":
                self._on_geofence("enter", zone)
            elif event == "leave":
                self._on_geofence("leave", zone)

    def _handle_espresense(self, topic: str, payload: Dict[str, Any]) -> None:
        """Process ESPresense BLE presence data."""
        # ESPresense publishes to: espresense/rooms/<room>/<deviceId>
        # Payload includes: {"rssi": -65, "loc": "smart_room", "confidence": 0.85}
        rssi = payload.get("rssi")
        room = payload.get("loc", "smart_room")
        confidence = payload.get("confidence", 0)

        # If rssi is strong enough, presence is detected
        threshold = self._config.get("esp32", {}).get("rssi_enter_threshold", -70)
        if rssi is not None and rssi > threshold:
            self._on_presence(True, rssi)
        else:
            self._on_presence(False, rssi)

    def publish_state(self, state_dict: Dict[str, Any]) -> None:
        """Publish state snapshot to MQTT (retained)."""
        if self._client and self._connected:
            self._client.publish(self._state_topic, json.dumps(state_dict), retain=True)

    @property
    def connected(self) -> bool:
        return self._connected
