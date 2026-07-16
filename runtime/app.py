"""Smart room runtime — main daemon process.

Runs as a child process spawned by the Marvi gateway (via process_manager).
Provides a JSON-RPC server on localhost for tool handlers to call.

Architecture:
  - JSON-RPC server (localhost TCP) — receives commands from bridge.py
  - MQTT client — subscribes to OwnTracks + ESPresense, publishes state
  - Tuya controller — direct LAN control of bulb + HE20 (fallback path)
  - Presence fusion — BLE + mmWave + geofence → presence state
  - Automation engine — evaluates configured rules on state changes
  - Scheduler — time-based triggers (alarm, evening sleep, daily reset)
  - State store — atomic JSON persistence

The runtime NEVER writes memory/Honcho. It supplies raw transitions;
the subconscious proposes durable patterns.
"""

from __future__ import annotations

import json
import hmac
import logging
from logging.handlers import RotatingFileHandler
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Dict, Optional
from pathlib import Path

# Add parent paths when run as __main__
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plugins.smart_room.runtime.models import ActiveAlarm, Alarm, RoomState, now_iso, DeviceHealth, VALID_MODES
from plugins.smart_room.runtime.state_store import (
    append_transition,
    load_config,
    load_state,
    publish_alarm,
    publish_welcome,
    save_state,
)
from plugins.smart_room.runtime.event_bus import EventBus
from plugins.smart_room.runtime.presence_fusion import fuse
from plugins.smart_room.runtime.automation_engine import evaluate_automations, Action
from plugins.smart_room.runtime.scheduler import Scheduler
from plugins.smart_room.runtime.command_router import CommandRouter
from plugins.smart_room.runtime.health import check_device_health

logger = logging.getLogger(__name__)

# JSON-RPC server config
_DEFAULT_PORT = 17842
_STATE_POLL_INTERVAL = 10  # seconds between Tuya device polls


def _state_locked(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._state_lock:
            return method(self, *args, **kwargs)

    return wrapped


class Runtime:
    """Main smart room runtime."""

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._state = load_state()
        self._bus = EventBus()
        self._running = False
        self._exit_event = threading.Event()
        self._state_lock = threading.RLock()
        self._command_lock = threading.RLock()
        self._pending_mode_timer: Optional[threading.Timer] = None
        self._pending_welcome_timer: Optional[threading.Timer] = None
        self._ble_detected = False
        self._ble_rssi: Optional[int] = None
        self._last_ble_seen_monotonic = 0.0
        # A clear event is edge-triggered after the room has remained empty
        # for the configured timeout.  HE20 can briefly report "none" between
        # occupied samples, so a single clear poll must never turn lights off.
        self._room_clear_emitted = not self._state.mmwave.occupied
        self._owner_name = str(config.get("owner", "Shereef")).strip() or "Shereef"
        self._owner = self._owner_name.lower()
        self._owner_device_id = str(
            (config.get("esp32") or {}).get("owner_device_id", "")
        ).strip().lower()
        self._rpc_token = os.environ.get("SMART_ROOM_RPC_TOKEN", "")
        self._last_wifi_probe = 0.0
        if not self._state.mmwave.occupied and not self._state.room_empty_since:
            self._state.room_empty_since = now_iso()

        # Initialize components (lazily — some need hardware present)
        self._mqtt = None
        self._tuya = None
        self._scheduler = None
        self._router = None
        self._sound_events = None
        self._rpc_thread: Optional[threading.Thread] = None
        self._poll_thread: Optional[threading.Thread] = None

        # Scene definitions from config
        configured_scenes = config.get("scenes") if isinstance(config.get("scenes"), dict) else {}
        self._scenes = {
            name: {**defaults, **(configured_scenes.get(name) or {})}
            for name, defaults in _DEFAULT_SCENES.items()
        }

    def start(self) -> None:
        """Start all runtime components."""
        logger.info("Smart room runtime starting...")

        # Initialize Tuya controller (fallback path)
        try:
            from plugins.smart_room.runtime.tuya.controller import TuyaController
            self._tuya = TuyaController(self._config)
        except Exception as e:
            logger.warning("Tuya controller init failed (non-fatal): %s", e)

        # Initialize MQTT client
        try:
            from plugins.smart_room.runtime.mqtt.client import MQTTClient
            self._mqtt = MQTTClient(
                config=self._config,
                on_presence=self._on_ble_presence,
                on_geofence=self._on_geofence,
                on_command=self._on_mqtt_command,
                on_node_status=self._on_esp32_status,
            )
            self._mqtt.start()
        except Exception as e:
            logger.warning("MQTT client init failed (non-fatal): %s", e)

        # Initialize scheduler
        self._scheduler = Scheduler(
            self._config,
            self._emit_event,
            get_alarms=self.list_alarms,
            get_active_alarm=self.get_active_alarm,
        )
        self._scheduler.start()

        # Initialize command router
        self._router = CommandRouter(self._state, self._config, self)

        # Start RPC server
        self._running = True
        self._start_rpc_server()

        # Start device poller
        self._poll_thread = threading.Thread(target=self._device_poll_loop, daemon=True, name="smart_room_poll")
        self._poll_thread.start()

        # Optional plugin-local clap detector. It owns its microphone and model;
        # the Marvi core audio/STT path is intentionally not involved.
        try:
            from plugins.smart_room.runtime.sound_events import SoundEventListener

            self._sound_events = SoundEventListener(
                self._config.get("sound_events") or {}, self._on_sound_action
            )
            self._sound_events.start()
        except Exception as e:
            logger.warning("Sound events init failed (non-fatal): %s", e)

        self._resume_active_alarm()
        self._resume_pending_mode()

        logger.info("Smart room runtime started — RPC on port %d", _rpc_port())

        # Wait for shutdown
        self._exit_event.wait()
        self._cleanup()

    def stop(self) -> None:
        """Signal the runtime to stop."""
        self._running = False
        self._exit_event.set()

    # -------------------------------------------------------------------
    # RPC server
    # -------------------------------------------------------------------

    def _start_rpc_server(self) -> None:
        """Start the JSON-RPC server in a background thread."""
        self._rpc_thread = threading.Thread(target=self._rpc_loop, daemon=True, name="smart_room_rpc")
        self._rpc_thread.start()

    def _rpc_loop(self) -> None:
        """Listen for JSON-RPC requests on localhost TCP."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if sys.platform == "win32":
            server.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind(("127.0.0.1", _rpc_port()))
        except OSError:
            logger.exception("Unable to bind Smart Room RPC port %d", _rpc_port())
            server.close()
            self.stop()
            return
        server.listen(5)
        server.settimeout(1.0)

        while self._running:
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            thread = threading.Thread(target=self._handle_rpc_conn, args=(conn,), daemon=True)
            thread.start()

        server.close()

    def _handle_rpc_conn(self, conn: socket.socket) -> None:
        """Handle a single RPC connection."""
        try:
            conn.settimeout(5.0)
            buf = b""
            while b"\n" not in buf:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
                if len(buf) > 1_048_576:
                    raise ValueError("RPC request exceeds 1 MiB")

            if not buf:
                return

            request = json.loads(buf.decode("utf-8").strip())
            supplied_token = str(request.get("auth", ""))
            if not self._rpc_token or not hmac.compare_digest(supplied_token, self._rpc_token):
                raise PermissionError("unauthorized runtime request")
            method = request.get("method", "")
            params = request.get("params", {})
            request_id = str(request.get("id", ""))
            result = self._router.dispatch(method, params, request_id=request_id)

            response = {"jsonrpc": "2.0", "id": request_id, "result": result}
            conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
        except Exception as e:
            logger.error("RPC handler error: %s", e)
            try:
                err = {"jsonrpc": "2.0", "id": "", "error": str(e)}
                conn.sendall((json.dumps(err) + "\n").encode("utf-8"))
            except Exception:
                pass
        finally:
            conn.close()

    # -------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event to the bus and evaluate automations."""
        with self._state_lock:
            if event_type == "schedule_alarm":
                alarm_data = data.get("alarm") or {}
                for alarm in self._state.alarms:
                    if alarm.id == alarm_data.get("id"):
                        alarm.last_fired_at = now_iso()
                        if alarm.recurrence == "once":
                            alarm.enabled = False
                        break
            elif event_type == "alarm_flash_finished" and self._state.active_alarm:
                self._state.active_alarm.phase = "steady"
            self._state.event_id += 1
            self._state.last_updated = now_iso()
            event = {
                "id": self._state.event_id,
                "at": self._state.last_updated,
                "type": event_type,
                **data,
            }
            self._bus.publish_all(event)
            if event_type in {
                "mode_changed",
                "presence_detected",
                "presence_cleared",
                "phone_location_changed",
                "device_offline",
                "sleep_cancelled",
                "alarm_acknowledged",
            }:
                event["summary"] = data.get("summary") or event_type.replace("_", " ")
                append_transition(event)
            actions = evaluate_automations(self._state, event, self._config)
            save_state(self._state)
        for action in actions:
            self._execute_action(action)

    def _on_ble_presence(
        self, detected: bool, rssi: Optional[int], identity: Optional[str] = None
    ) -> None:
        """Handle BLE presence event from ESPresense."""
        with self._state_lock:
            identity = (identity or self._owner).lower()
            esp32 = self._state.devices.setdefault("esp32", DeviceHealth())
            esp32.online = True
            esp32.last_seen = now_iso()
            owner_ids = {self._owner, f"apple:{self._owner}"}
            if self._owner_device_id:
                owner_ids.add(self._owner_device_id)
            is_owner = identity in owner_ids
            if is_owner:
                self._ble_detected = detected
                self._ble_rssi = rssi
                if detected:
                    self._last_ble_seen_monotonic = time.monotonic()
            was_present = self._state.presence.detected
            self._update_presence(other_identity_detected=detected and not is_owner)
            present = self._state.presence.detected
            source = self._state.presence.source
            stale_geofence = self._location_signal_stale()
        if not was_present and present:
            self._emit_event("presence_detected", {"source": source})
            if stale_geofence:
                self._emit_event("ble_arrive_fallback", {"source": "ble"})
        elif was_present and not present:
            self._emit_event("presence_cleared", {})

    def _on_geofence(self, action: str, zone: str) -> None:
        """Handle optional OwnTracks events through the same location path."""
        if action == "sync":
            with self._state_lock:
                if self._state.location.zone == zone and self._state.location.source == "owntracks":
                    return
                stamp = now_iso()
                self._state.location.zone = zone
                self._state.location.home = zone == "home"
                self._state.location.since = stamp
                self._state.location.source = "owntracks"
                self._state.location.last_geofence_at = stamp
                self._update_presence(geofence_zone=zone)
            return
        transition = "arrive" if action == "enter" else "leave"
        self.phone_location_changed(
            who=self._owner,
            transition=transition,
            zone=zone,
            at=now_iso(),
            delivery_id=f"owntracks:{transition}:{zone}:{int(time.time())}",
            source="owntracks",
        )

    @_state_locked
    def _on_esp32_status(self, online: bool, ip: Optional[str] = None) -> None:
        """Update node health from ESPresense retained status/telemetry."""
        esp32 = self._state.devices.setdefault("esp32", DeviceHealth())
        esp32.online = online
        if ip:
            esp32.ip = str(ip)
        elif not esp32.ip:
            esp32.ip = (self._config.get("esp32") or {}).get("ip")
        if online:
            esp32.last_seen = now_iso()
        save_state(self._state)

    def _location_signal_stale(self) -> bool:
        value = self._state.location.last_geofence_at
        if not value:
            return True
        try:
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return (datetime.now(timezone.utc) - stamp.astimezone(timezone.utc)).total_seconds() > 86400
        except (TypeError, ValueError):
            return True

    @_state_locked
    def _update_presence(
        self,
        *,
        geofence_zone: Optional[str] = None,
        wifi_detected: bool = False,
        other_identity_detected: bool = False,
    ) -> tuple[bool, bool]:
        missing_timeout = int((self._config.get("esp32") or {}).get("missing_timeout_seconds", 30))
        if (
            self._ble_detected
            and self._last_ble_seen_monotonic
            and time.monotonic() - self._last_ble_seen_monotonic > missing_timeout
        ):
            self._ble_detected = False
            self._ble_rssi = None
        self._state.presence, self._state.mmwave, self._state.location, light_on, light_off = fuse(
            presence=self._state.presence,
            mmwave=self._state.mmwave,
            location=self._state.location,
            ble_detected=self._ble_detected,
            ble_rssi=self._ble_rssi,
            mmwave_occupied=self._state.mmwave.occupied,
            geofence_zone=geofence_zone,
            exit_timeout_elapsed=self._check_exit_timeout(),
            wifi_detected=wifi_detected,
            other_identity_detected=other_identity_detected,
        )
        self._state.last_updated = now_iso()
        save_state(self._state)
        return light_on, light_off

    def phone_location_changed(
        self,
        *,
        who: str,
        transition: str,
        zone: str,
        at: str,
        delivery_id: str,
        source: str,
    ) -> Dict[str, Any]:
        """Apply one validated, idempotent phone-location transition."""
        with self._state_lock:
            who = str(who).strip().lower()
            transition = str(transition).strip().lower()
            zone = str(zone).strip().lower()
            source = str(source).strip().lower()
            event_key = f"{who}|{transition}|{zone}|{at}"
            if self._state.location.last_event_key == event_key:
                return {"success": True, "duplicate": True, "event_id": self._state.event_id}

            was_present = self._state.presence.detected
            self._state.location.last_event_key = event_key
            self._state.location.source = source
            self._state.location.since = at
            if source == "owntracks":
                self._state.location.last_geofence_at = at

            if transition == "arrive":
                self._state.location.zone = zone
                self._state.location.home = zone == "home"
            elif zone == "home":
                self._state.location.zone = "away"
                self._state.location.home = False
                self._cancel_pending_mode()
            elif self._state.location.zone == zone:
                self._state.location.zone = "away"

            self._update_presence(geofence_zone=self._state.location.zone)
            present = self._state.presence.detected
        self._emit_event(
            "phone_location_changed",
            {
                "who": who,
                "transition": transition,
                "zone": zone,
                "source": source,
                "delivery_id": delivery_id,
            },
        )
        if transition == "arrive" and zone == "home":
            self._emit_event("geofence_arrive_home", {"zone": zone, "source": source})
        elif transition == "leave" and zone == "home":
            self._emit_event("geofence_leave_home", {"zone": zone, "source": source})
        if was_present and not present:
            self._emit_event("presence_cleared", {"source": "geofence"})
        with self._state_lock:
            return {"success": True, "duplicate": False, "state": self._state.to_dict()}

    def _on_mqtt_command(self, payload: Dict[str, Any]) -> None:
        """Handle a command from MQTT (alternative to RPC)."""
        action = payload.get("action", "")
        if action == "light_on":
            self.set_light(on=True)
        elif action == "light_off":
            self.set_light(on=False)
        elif action == "set_mode":
            self.set_mode(payload.get("mode", "off"))
        elif action == "set_brightness":
            self.set_light(brightness=payload.get("value", 50))
        elif action == "set_color":
            self.set_light(rgb=payload.get("rgb"))

    def _on_sound_action(self, action: str) -> None:
        """Apply a locally detected clap sequence without entering Marvi core."""
        if action == "toggle_light":
            current = self._state.light.on
            if self._tuya:
                status = self._tuya.get_light_status()
                if status.get("success"):
                    current = bool(status.get("on"))
            self.set_light(on=not current, manual=True)
            logger.info("Double clap toggled the light %s", "off" if current else "on")
        elif action == "sleep":
            if self._state.modes.active_mode != "sleep":
                self.set_mode("sleep", reason="sound_event")
                logger.info("Triple clap activated sleep mode")
        else:
            logger.warning("Unknown sound action: %s", action)

    # -------------------------------------------------------------------
    # Device polling
    # -------------------------------------------------------------------

    def _device_poll_loop(self) -> None:
        """Poll Tuya devices for status updates every N seconds."""
        while self._running:
            try:
                self._poll_devices()
            except Exception as e:
                logger.error("Device poll failed: %s", e)
            self._exit_event.wait(_STATE_POLL_INTERVAL)

    def _poll_devices(self) -> None:
        """Poll blocking devices outside the shared presence/state lock."""
        bulb_status = self._tuya.get_light_status() if self._tuya else None
        he20_status = self._tuya.get_mmwave_status() if self._tuya else None
        worker_health = self._tuya.health() if self._tuya else {}
        offline_events = []
        with self._state_lock:
            was_present = self._state.presence.detected
            was_occupied = self._state.mmwave.occupied
            if bulb_status is not None:
                bulb = self._state.devices.setdefault("tuya_bulb", DeviceHealth())
                if bulb_status.get("success"):
                    self._state.light.on = bulb_status.get("on", False)
                    self._state.light.brightness = bulb_status.get("brightness", 0)
                    self._state.light.confirmed = True
                    bulb.online = True
                    bulb.ip = (self._config.get("tuya") or {}).get("bulb", {}).get("ip")
                    bulb.last_poll = bulb.last_success = now_iso()
                    bulb.consecutive_failures = 0
                    bulb.last_command = "get_status"
                else:
                    if bulb.online:
                        offline_events.append("tuya_bulb")
                    bulb.online = False
                    bulb.consecutive_failures += 1
            if he20_status is not None:
                he20 = self._state.devices.setdefault("tuya_he20", DeviceHealth())
                if he20_status.get("success"):
                    self._state.mmwave.occupied = he20_status.get("occupied", False)
                    if self._state.mmwave.occupied:
                        self._state.mmwave.last_seen = now_iso()
                    he20.online = True
                    he20.ip = (self._config.get("tuya") or {}).get("he20", {}).get("ip")
                    he20.last_poll = he20.last_success = now_iso()
                    he20.consecutive_failures = 0
                    he20.last_command = "get_status"
                else:
                    if he20.online:
                        offline_events.append("tuya_he20")
                    he20.online = False
                    he20.consecutive_failures += 1
            for name, state_name in (("bulb", "tuya_bulb"), ("he20", "tuya_he20")):
                metrics = worker_health.get(name, {})
                device = self._state.devices.setdefault(state_name, DeviceHealth())
                device.queue_depth = int(metrics.get("queue_depth", 0))
                device.circuit_open = bool(metrics.get("circuit_open", False))
            occupied = self._state.mmwave.occupied
            save_state(self._state)

        for device in offline_events:
            self._emit_event("device_offline", {"device": device})
        self._handle_welcome_transition(was_occupied, occupied)

        wifi_detected = self._probe_wifi_presence()
        _, light_should_off = self._update_presence(wifi_detected=wifi_detected)
        with self._state_lock:
            if self._state.mmwave.occupied:
                self._room_clear_emitted = False
            present = self._state.presence.detected
            source = self._state.presence.source
            should_clear = light_should_off and not self._room_clear_emitted
            if should_clear:
                self._room_clear_emitted = True
            snapshot = self._state.to_dict()
        if not was_present and present:
            self._emit_event("presence_detected", {"source": source})
        elif should_clear:
            self._emit_event("presence_cleared", {"source": "mmwave"})

        # Publish state via MQTT
        if self._mqtt:
            self._mqtt.publish_state(snapshot)

    def _handle_welcome_transition(self, was_occupied: bool, occupied: bool) -> None:
        welcome = self._config.get("welcome") or {}
        if not welcome.get("enabled", True):
            return
        if not occupied:
            if was_occupied:
                self._state.room_empty_since = now_iso()
                save_state(self._state)
            return
        if was_occupied:
            return

        empty_since = self._state.room_empty_since
        self._state.room_empty_since = None
        save_state(self._state)
        try:
            empty_at = datetime.fromisoformat(str(empty_since).replace("Z", "+00:00"))
            empty_seconds = (datetime.now(timezone.utc) - empty_at.astimezone(timezone.utc)).total_seconds()
        except (TypeError, ValueError):
            return
        if empty_seconds < max(60, int(welcome.get("reset_after_seconds", 3600))):
            return

        if self._pending_welcome_timer:
            self._pending_welcome_timer.cancel()
        delay = max(0, int(welcome.get("identity_grace_seconds", 4)))
        self._pending_welcome_timer = threading.Timer(delay, self._deliver_welcome)
        self._pending_welcome_timer.daemon = True
        self._pending_welcome_timer.start()

    def _deliver_welcome(self) -> None:
        self._pending_welcome_timer = None
        with self._state_lock:
            if not self._state.mmwave.occupied or self._state.modes.active_mode == "sleep":
                return
            owner_detected = self._ble_detected

        self._publish_welcome(owner_detected, self._owner_name, record_arrival=True)

    def test_welcome(self, audience: str) -> None:
        """Generate a real welcome preview without changing arrival state."""
        owner_detected = audience == "owner"
        owner_name = str(load_config().get("owner", self._owner_name)).strip() or self._owner_name
        thread = threading.Thread(
            target=self._publish_welcome,
            args=(owner_detected, owner_name),
            kwargs={"record_arrival": False},
            daemon=True,
            name=f"smart_room_welcome_test_{audience}",
        )
        thread.start()

    def _publish_welcome(
        self, owner_detected: bool, owner_name: str, *, record_arrival: bool
    ) -> None:
        """Generate through auxiliary.voice_instant and publish to the TTS lane."""

        fallback = f"Welcome back, {owner_name}." if owner_detected else "Welcome."
        try:
            from agent.auxiliary_client import call_llm
            from agent.message_content import flatten_message_text
            from agent.prompt_builder import load_soul_md

            identity = (
                f"The detected person is the owner, named {owner_name}. Include that exact name."
                if owner_detected
                else "The detected person is a guest whose name is unknown. Do not invent a name."
            )
            soul = (load_soul_md() or "")[:4000]
            response = call_llm(
                task="voice_instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Write exactly one short, natural spoken welcome in Marvi's personality. "
                            "Vary the wording, use at most two brief sentences, and do not ask a question."
                            + (f"\n\nMarvi personality:\n{soul}" if soul else "")
                        ),
                    },
                    {"role": "user", "content": identity},
                ],
                temperature=0.9,
                max_tokens=80,
                timeout=20,
            )
            greeting = " ".join(flatten_message_text(response.choices[0].message.content).split()).strip(' "')
            if not greeting:
                greeting = fallback
            if owner_detected and owner_name.casefold() not in greeting.casefold():
                greeting = f"{owner_name}, {greeting[:1].lower()}{greeting[1:]}"
            greeting = greeting[:300]
        except Exception:
            logger.warning("Could not generate room welcome; using fallback", exc_info=True)
            greeting = fallback

        publish_welcome(greeting)
        if record_arrival:
            with self._state_lock:
                self._state.last_welcome_at = now_iso()
                save_state(self._state)

    def _probe_wifi_presence(self) -> bool:
        config = (self._config.get("presence") or {}).get("wifi_ping") or {}
        if not config.get("enabled", False):
            return False
        now = time.monotonic()
        interval = max(10, int(config.get("interval_seconds", 60)))
        if now - self._last_wifi_probe < interval:
            return False
        self._last_wifi_probe = now
        address = str(config.get("ip", "")).strip()
        if not address:
            return False
        args = ["ping", "-n", "1", "-w", "1000", address] if sys.platform == "win32" else ["ping", "-c", "1", "-W", "1", address]
        try:
            return subprocess.run(
                args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False,
            ).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def _check_exit_timeout(self) -> bool:
        """Check whether mmWave has remained clear for the exit timeout."""
        exit_timeout = self._config.get("esp32", {}).get("exit_timeout", 60)
        if self._state.mmwave.occupied:
            return False
        last_seen = self._state.mmwave.last_seen
        if last_seen is None:
            return True
        try:
            from datetime import datetime, timezone
            last = datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return (now - last).total_seconds() > exit_timeout
        except Exception:
            return True

    # -------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------

    def _execute_action(self, action: Action) -> None:
        """Execute an automation action."""
        logger.info("Executing action: %s %s", action.type, action.params)

        if action.type == "turn_on":
            if action.params.get("restore_scene"):
                # Restore last scene mode
                mode_name = self._state.modes.active_mode
                if mode_name in self._scenes and mode_name != "alarm":
                    scene = self._scenes[mode_name]
                    self.set_light(
                        on=True,
                        brightness=scene.get("brightness", 70),
                        color_temp=scene.get("color_temp"),
                        rgb=scene.get("rgb"),
                        transition=scene.get("transition"),
                    )
                    return
            self.set_light(on=True)

        elif action.type == "turn_off":
            self.set_light(on=False)

        elif action.type == "set_light":
            self.set_light(**action.params)

        elif action.type == "set_mode":
            mode = action.params.get("mode", "off")
            delay = action.params.get("delay")
            if delay:
                self._schedule_mode(
                    mode,
                    int(delay),
                    reason=action.params.get("reason"),
                )
            else:
                self.set_mode(
                    mode,
                    reason=action.params.get("reason"),
                    alarm_id=action.params.get("alarm_id"),
                    alarm_name=action.params.get("alarm_name"),
                    duration_minutes=action.params.get("duration_minutes"),
                )

        elif action.type == "ack_alarm":
            self.acknowledge_alarm(reason=action.params.get("reason", "acknowledged"))

        elif action.type == "set_flag":
            with self._state_lock:
                for key, val in action.params.items():
                    if hasattr(self._state.flags, key):
                        setattr(self._state.flags, key, val)
                self._state.flags.last_reset = now_iso()
                self._state.last_updated = now_iso()
                self._state.event_id += 1
                save_state(self._state)

    def _cancel_pending_mode(self, *, clear_state: bool = True) -> None:
        timer = self._pending_mode_timer
        if timer:
            timer.cancel()
        self._pending_mode_timer = None
        if clear_state:
            with self._state_lock:
                self._state.pending_mode = None
                save_state(self._state)

    def _schedule_mode(
        self,
        mode: str,
        delay: int,
        *,
        reason: Optional[str],
        due_at: Optional[str] = None,
    ) -> None:
        self._cancel_pending_mode(clear_state=False)
        due = due_at or (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()
        with self._state_lock:
            self._state.pending_mode = {"mode": mode, "reason": reason, "due_at": due}
            save_state(self._state)

        def apply_mode() -> None:
            self._pending_mode_timer = None
            with self._state_lock:
                self._state.pending_mode = None
                save_state(self._state)
            self.set_mode(mode, reason=reason)

        self._pending_mode_timer = threading.Timer(delay, apply_mode)
        self._pending_mode_timer.daemon = True
        self._pending_mode_timer.start()

    def _resume_pending_mode(self) -> None:
        with self._state_lock:
            pending = dict(self._state.pending_mode or {})
        if not pending:
            return
        try:
            due = datetime.fromisoformat(str(pending["due_at"]).replace("Z", "+00:00")).astimezone(timezone.utc)
            delay = max(0, int((due - datetime.now(timezone.utc)).total_seconds()))
        except (KeyError, TypeError, ValueError):
            self._cancel_pending_mode()
            return
        if delay == 0:
            with self._state_lock:
                self._state.pending_mode = None
                save_state(self._state)
            self.set_mode(str(pending.get("mode", "off")), reason=pending.get("reason"))
        else:
            self._schedule_mode(
                str(pending.get("mode", "off")),
                delay,
                reason=pending.get("reason"),
                due_at=str(pending["due_at"]),
            )

    def set_mode(
        self,
        mode: str,
        *,
        reason: Optional[str] = None,
        alarm_id: Optional[str] = None,
        alarm_name: Optional[str] = None,
        duration_minutes: Optional[int] = None,
    ) -> None:
        """Set the room mode."""
        with self._command_lock:
            if mode not in VALID_MODES:
                raise ValueError(f"invalid room mode: {mode}")
            with self._state_lock:
                active_alarm = self._state.active_alarm
            if mode == "alarm" and active_alarm:
                return
            if active_alarm and mode != "alarm" and reason != "alarm_restore":
                self.acknowledge_alarm(reason=f"mode_changed:{mode}")
            if mode != "sleep" or reason != "work_return":
                self._cancel_pending_mode()

            with self._state_lock:
                previous_mode = self._state.modes.active_mode
                if mode == "sleep" and previous_mode != "sleep":
                    self._state.sleep_restore = {
                        "light": self._state.light.__dict__.copy(),
                        "mode": previous_mode,
                    }
                self._state.modes.active_mode = mode
                self._state.modes.work_return = mode == "sleep" and reason == "work_return"
                if reason == "work_return":
                    self._state.flags.work_sleep_done_today = True
                elif reason == "evening":
                    self._state.flags.evening_sleep_done_today = True

                if mode == "alarm":
                    now = datetime.now(timezone.utc)
                    minutes = max(1, min(180, int(duration_minutes or 30)))
                    self._state.alarm_restore = {
                        "mode": previous_mode if previous_mode != "alarm" else "off",
                        "light": self._state.light.__dict__.copy(),
                    }
                    self._state.active_alarm = ActiveAlarm(
                        id=alarm_id or uuid.uuid4().hex[:12],
                        name=(alarm_name or "Alarm").strip()[:80] or "Alarm",
                        started_at=now.isoformat(),
                        flash_until=(now + timedelta(seconds=60)).isoformat(),
                        expires_at=(now + timedelta(minutes=minutes)).isoformat(),
                    )
                    active_alarm = self._state.active_alarm
                else:
                    active_alarm = None
                scene = dict(self._scenes.get(mode, {}))
                save_state(self._state)

            if mode in {"off", "sleep"}:
                self.set_light(on=False)
            elif mode == "alarm" and active_alarm:
                publish_alarm(active_alarm.id, f"Your alarm {active_alarm.name}. Are you awake?", active=True)
            elif scene:
                self.set_light(
                    on=True,
                    brightness=scene.get("brightness", 70),
                    color_temp=scene.get("color_temp"),
                    rgb=scene.get("rgb"),
                    transition=scene.get("transition"),
                )
                with self._state_lock:
                    self._state.light.scene = mode
                    save_state(self._state)

            self._emit_event("mode_changed", {"mode": mode, "reason": reason})

    def set_light(
        self,
        on: Optional[bool] = None,
        brightness: Optional[int] = None,
        color_temp: Optional[int] = None,
        rgb: Optional[list] = None,
        flash: bool = False,
        flash_interval_ms: int = 500,
        transition: Optional[float] = None,
        manual: bool = False,
    ) -> Dict[str, Any]:
        """Set the light state."""
        if brightness is not None and not 0 <= int(brightness) <= 100:
            raise ValueError("brightness must be between 0 and 100")
        if color_temp is not None and not 2200 <= int(color_temp) <= 6500:
            raise ValueError("color_temp must be between 2200 and 6500")
        if rgb is not None and (
            len(rgb) != 3 or any(not 0 <= int(value) <= 255 for value in rgb)
        ):
            raise ValueError("rgb must contain three values from 0 to 255")
        if manual:
            self._cancel_pending_mode()
        with self._state_lock:
            if on is not None:
                self._state.light.on = on
            if brightness is not None:
                self._state.light.brightness = brightness
            if color_temp is not None:
                self._state.light.color_temp = color_temp
                self._state.light.rgb = None
            if rgb is not None:
                self._state.light.rgb = rgb
            self._state.light.scene = "off" if not self._state.light.on else "alarm" if flash else "custom"
            self._state.light.confirmed = False
            save_state(self._state)

        # Control the physical device
        if self._tuya:
            device_result = self._tuya.set_light(
                on=on, brightness=brightness, color_temp=color_temp, rgb=rgb,
                flash=flash, flash_interval_ms=flash_interval_ms, transition=transition,
            )
        else:
            logger.warning("No Tuya controller — light command not sent to device")
            device_result = {
                "success": False,
                "code": "DEVICE_UNAVAILABLE",
                "error": "Tuya controller is not configured",
            }

        with self._state_lock:
            self._state.light.confirmed = bool(device_result.get("success"))
            self._state.light.last_error = None if device_result.get("success") else str(device_result.get("error", "device command failed"))
            self._state.last_updated = now_iso()
            self._state.event_id += 1
            save_state(self._state)
        return {
            "success": bool(device_result.get("success")),
            "logical_applied": True,
            "device": device_result,
        }

    def cancel_sleep(self) -> None:
        """Cancel sleep mode and restore previous state."""
        with self._command_lock:
            self._cancel_pending_mode()
            with self._state_lock:
                was_evening = self._state.flags.evening_sleep_done_today
                was_work = self._state.flags.work_sleep_done_today
                self._state.modes.active_mode = "off"
                self._state.modes.work_return = False
                if was_work:
                    self._state.flags.work_sleep_cancel_today = True
                elif was_evening:
                    self._state.flags.evening_sleep_cancel_today = True
                restore = dict(self._state.sleep_restore or {})
                occupied = self._state.mmwave.occupied or self._state.presence.detected
                save_state(self._state)

            if was_work:
                self._emit_event("sleep_cancelled", {"reason": "work_return"})
            elif was_evening:
                self._emit_event("sleep_cancelled", {"reason": "evening"})

            light = restore.get("light") if isinstance(restore.get("light"), dict) else {}
            if occupied:
                self.set_light(
                    on=bool(light.get("on", True)),
                    brightness=light.get("brightness"),
                    color_temp=light.get("color_temp"),
                    rgb=light.get("rgb"),
                )
            else:
                self.set_light(on=False)
            with self._state_lock:
                if occupied and light.get("scene"):
                    self._state.light.scene = str(light["scene"])
                previous_mode = restore.get("mode")
                if previous_mode in {"reading", "focus", "relax", "night"}:
                    self._state.modes.active_mode = previous_mode
                self._state.sleep_restore = {}
                save_state(self._state)
            logger.info("Sleep mode cancelled")

    @_state_locked
    def set_override(self, mode: str) -> None:
        if mode not in {"none", "hold_on", "hold_off"}:
            raise ValueError("override must be none, hold_on, or hold_off")
        if mode != "none":
            self._cancel_pending_mode()
        self._state.modes.manual_override = mode
        self._emit_event("mode_changed", {"mode": "manual_override", "override": mode})

    @_state_locked
    def list_alarms(self) -> list[Dict[str, Any]]:
        return [asdict(alarm) for alarm in self._state.alarms]

    @_state_locked
    def get_active_alarm(self) -> Optional[Dict[str, Any]]:
        return asdict(self._state.active_alarm) if self._state.active_alarm else None

    @_state_locked
    def upsert_alarm(self, data: Dict[str, Any]) -> Dict[str, Any]:
        alarm_id = str(data.get("id") or uuid.uuid4().hex[:12]).strip()
        name = str(data.get("name") or "Alarm").strip()[:80] or "Alarm"
        alarm_time = str(data.get("time") or "").strip()
        try:
            datetime.strptime(alarm_time, "%H:%M")
        except ValueError as exc:
            raise ValueError("alarm time must be HH:MM") from exc
        recurrence = str(data.get("recurrence") or "daily").lower()
        if recurrence not in {"once", "daily"}:
            raise ValueError("alarm recurrence must be once or daily")
        date = str(data.get("date") or "").strip() or None
        if recurrence == "once":
            try:
                datetime.strptime(date or "", "%Y-%m-%d")
            except ValueError as exc:
                raise ValueError("one-time alarms require a YYYY-MM-DD date") from exc
        duration = max(1, min(180, int(data.get("duration_minutes", 30))))
        existing = next((alarm for alarm in self._state.alarms if alarm.id == alarm_id), None)
        values = {
            "name": name,
            "time": alarm_time,
            "recurrence": recurrence,
            "date": date if recurrence == "once" else None,
            "enabled": bool(data.get("enabled", True)),
            "duration_minutes": duration,
        }
        if existing:
            for key, value in values.items():
                setattr(existing, key, value)
            alarm = existing
        else:
            alarm = Alarm(id=alarm_id, **values)
            self._state.alarms.append(alarm)
        save_state(self._state)
        return asdict(alarm)

    def delete_alarm(self, alarm_id: str) -> bool:
        with self._state_lock:
            before = len(self._state.alarms)
            self._state.alarms = [alarm for alarm in self._state.alarms if alarm.id != alarm_id]
            active = bool(self._state.active_alarm and self._state.active_alarm.id == alarm_id)
            save_state(self._state)
        if active:
            self.acknowledge_alarm(reason="deleted")
        return len(self._state.alarms) != before

    def acknowledge_alarm(self, *, reason: str = "awake") -> Dict[str, Any]:
        with self._command_lock:
            with self._state_lock:
                active = self._state.active_alarm
                if not active:
                    return {"success": True, "active": False}
                restore = dict(self._state.alarm_restore or {})
                self._state.active_alarm = None
                self._state.alarm_restore = {}
                self._state.modes.active_mode = "off"
                save_state(self._state)
            if self._tuya:
                self._tuya.stop_flash()
            previous_mode = str(restore.get("mode") or "off")
            light = restore.get("light") if isinstance(restore.get("light"), dict) else {}
            if previous_mode in self._scenes:
                self.set_mode(previous_mode, reason="alarm_restore")
            elif previous_mode == "sleep":
                self.set_mode("sleep", reason="alarm_restore")
            elif light.get("on"):
                self.set_light(
                    on=True,
                    brightness=light.get("brightness"),
                    color_temp=light.get("color_temp"),
                    rgb=light.get("rgb"),
                )
            else:
                self.set_light(on=False)
            publish_alarm(active.id, "", active=False)
            self._emit_event("alarm_acknowledged", {"alarm_id": active.id, "reason": reason})
            return {"success": True, "active": False, "alarm_id": active.id}

    def _resume_active_alarm(self) -> None:
        active = self._state.active_alarm
        if not active:
            return
        now = datetime.now(timezone.utc)
        try:
            expires = datetime.fromisoformat(active.expires_at.replace("Z", "+00:00"))
            flash_until = datetime.fromisoformat(active.flash_until.replace("Z", "+00:00"))
        except ValueError:
            self.acknowledge_alarm(reason="invalid_state")
            return
        self._state.modes.active_mode = "alarm"
        if now >= expires:
            self.acknowledge_alarm(reason="expired_during_restart")
        elif now >= flash_until:
            active.phase = "steady"
            self.set_light(on=True, brightness=100, color_temp=6500)
        else:
            self.set_light(on=True, brightness=100, color_temp=6500, flash=True)

    def get_status(self) -> Dict[str, Any]:
        """Get runtime status for diagnostics."""
        return {
            "running": self._running,
            "rpc_port": _rpc_port(),
            "mqtt_connected": self._mqtt.connected if self._mqtt else False,
            "tuya_available": bool(self._tuya and self._tuya.available),
            "state_event_id": self._state.event_id,
            "active_alarm": self.get_active_alarm(),
            "alarms": self.list_alarms(),
            "tuya": self._tuya.health() if self._tuya else {},
            "sound_events": self._sound_events.status() if self._sound_events else {
                "enabled": False,
                "running": False,
            },
        }

    def run_diagnostic(self) -> Dict[str, Any]:
        from plugins.smart_room.runtime.command_router import _redact_config

        bulb = self._tuya.get_light_status() if self._tuya else {"success": False, "error": "not configured"}
        he20 = self._tuya.get_mmwave_status() if self._tuya else {"success": False, "error": "not configured"}
        scene_errors = []
        for name, scene in self._scenes.items():
            brightness = scene.get("brightness", 0)
            color_temp = scene.get("color_temp")
            rgb = scene.get("rgb")
            if not 0 <= int(brightness) <= 100:
                scene_errors.append(f"{name}: brightness out of range")
            if color_temp is not None and not 2200 <= int(color_temp) <= 6500:
                scene_errors.append(f"{name}: color temperature out of range")
            if rgb is not None and (len(rgb) != 3 or any(not 0 <= int(v) <= 255 for v in rgb)):
                scene_errors.append(f"{name}: invalid RGB")
        with self._state_lock:
            state = self._state.to_dict()
            health = check_device_health(self._state, self._config)
        runtime = self.get_status()
        health["mqtt"]["connected"] = bool(runtime.get("mqtt_connected"))
        return {
            "state": state,
            "config": _redact_config(self._config),
            "health": health,
            "runtime": runtime,
            "checks": {
                "mqtt": {"success": bool(runtime.get("mqtt_connected"))},
                "bulb": bulb,
                "he20": he20,
                "scenes": {"success": not scene_errors, "errors": scene_errors},
            },
        }

    def _cleanup(self) -> None:
        """Clean shutdown of all components."""
        logger.info("Smart room runtime shutting down...")
        if self._scheduler:
            self._scheduler.stop()
        if self._sound_events:
            self._sound_events.stop()
        self._cancel_pending_mode(clear_state=False)
        if self._pending_welcome_timer:
            self._pending_welcome_timer.cancel()
            self._pending_welcome_timer = None
        if self._tuya:
            self._tuya.stop()
        if self._mqtt:
            self._mqtt.stop()
        if self._poll_thread and self._poll_thread is not threading.current_thread():
            self._poll_thread.join(timeout=2)
        if self._rpc_thread and self._rpc_thread is not threading.current_thread():
            self._rpc_thread.join(timeout=2)
        save_state(self._state)
        logger.info("Smart room runtime stopped")


# ---------------------------------------------------------------------------
# Default scenes (used when config doesn't define them)
# ---------------------------------------------------------------------------

_DEFAULT_SCENES = {
    "reading": {"color_temp": 3000, "brightness": 70, "transition": 2},
    "focus": {"color_temp": 5000, "brightness": 100, "transition": 2},
    "relax": {"color_temp": 2700, "rgb": [255, 180, 80], "brightness": 40, "transition": 3},
    "night": {"color_temp": 2200, "rgb": [255, 120, 40], "brightness": 15, "transition": 3},
    "alarm": {"color_temp": 6500, "brightness": 100, "flash": True, "flash_interval": 500},
}


def _rpc_port() -> int:
    return int(os.environ.get("SMART_ROOM_RPC_PORT", _DEFAULT_PORT))


def main() -> None:
    """Entry point for the runtime process."""
    from hermes_constants import get_hermes_home

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "component": record.name,
                "event": record.getMessage(),
            }
            if record.exc_info:
                payload["error"] = self.formatException(record.exc_info)
            return json.dumps(payload, ensure_ascii=False)

    log_path = Path(get_hermes_home()) / "smart_room" / "runtime.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)

    # Load config
    config = load_config()
    if not config:
        logger.warning("No smart_room config found — using defaults")
        config = {}

    runtime = Runtime(config)

    # Handle SIGTERM for clean shutdown
    def _sigterm(signum, frame):
        logger.info("Received SIGTERM — stopping")
        runtime.stop()

    signal.signal(signal.SIGTERM, _sigterm)

    try:
        runtime.start()
    except KeyboardInterrupt:
        runtime.stop()


if __name__ == "__main__":
    main()
