"""Smart room runtime — main daemon process.

Runs as a child process spawned by the Marvi gateway (via process_manager).
Provides a JSON-RPC server on localhost for tool handlers to call.

Architecture:
  - JSON-RPC server (localhost TCP) — receives commands from bridge.py
  - MQTT client — subscribes to OwnTracks + ESPresense, publishes state
  - Tuya controller — direct LAN control of bulb + HE20 (fallback path)
  - Presence fusion — BLE + mmWave + geofence → presence state
  - Automation engine — evaluates rules on state changes
  - Scheduler — time-based triggers (alarm, evening sleep, daily reset)
  - State store — atomic JSON persistence

The runtime NEVER writes memory/Honcho. It supplies raw transitions;
the subconscious proposes durable patterns.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from typing import Any, Dict, Optional

# Add parent paths when run as __main__
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plugins.smart_room.runtime.models import RoomState, LightState, Modes, now_iso, DeviceHealth
from plugins.smart_room.runtime.state_store import load_state, save_state, load_config
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


class Runtime:
    """Main smart room runtime."""

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._state = load_state()
        self._bus = EventBus()
        self._running = False
        self._exit_event = threading.Event()

        # Initialize components (lazily — some need hardware present)
        self._mqtt = None
        self._tuya = None
        self._scheduler = None
        self._router = None

        # Scene definitions from config
        self._scenes = config.get("scenes", _DEFAULT_SCENES)

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
            )
            self._mqtt.start()
        except Exception as e:
            logger.warning("MQTT client init failed (non-fatal): %s", e)

        # Initialize scheduler
        self._scheduler = Scheduler(self._config, self._emit_event)
        self._scheduler.start()

        # Initialize command router
        self._router = CommandRouter(self._state, self._config, self)

        # Start RPC server
        self._running = True
        self._start_rpc_server()

        # Start device poller
        poll_thread = threading.Thread(target=self._device_poll_loop, daemon=True, name="smart_room_poll")
        poll_thread.start()

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
        thread = threading.Thread(target=self._rpc_loop, daemon=True, name="smart_room_rpc")
        thread.start()

    def _rpc_loop(self) -> None:
        """Listen for JSON-RPC requests on localhost TCP."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", _rpc_port()))
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

            if not buf:
                return

            request = json.loads(buf.decode("utf-8").strip())
            method = request.get("method", "")
            params = request.get("params", {})
            result = self._router.dispatch(method, params)

            response = {"jsonrpc": "2.0", "id": request.get("id", ""), "result": result}
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
        event = {"type": event_type, **data}
        self._bus.publish_all(event)
        actions = evaluate_automations(self._state, event, self._config)
        for action in actions:
            self._execute_action(action)

    def _on_ble_presence(self, detected: bool, rssi: Optional[int]) -> None:
        """Handle BLE presence event from ESPresense."""
        was_present = self._state.presence.detected

        # Fuse signals
        exit_timeout = self._check_exit_timeout()
        self._state.presence, self._state.mmwave, self._state.location, light_on, light_off = fuse(
            presence=self._state.presence,
            mmwave=self._state.mmwave,
            location=self._state.location,
            ble_detected=detected,
            ble_rssi=rssi,
            mmwave_occupied=self._state.mmwave.occupied,
            geofence_zone=None,
            exit_timeout_elapsed=exit_timeout,
        )

        self._state.last_updated = now_iso()
        self._state.event_id += 1
        save_state(self._state)

        if not was_present and self._state.presence.detected:
            self._emit_event("presence_detected", {"source": self._state.presence.source})
        elif was_present and not self._state.presence.detected:
            self._emit_event("presence_cleared", {})

    def _on_geofence(self, action: str, zone: str) -> None:
        """Handle OwnTracks geofence event."""
        if action == "enter" and zone == "home":
            self._state.location.home = True
            self._state.location.zone = "home"
            self._state.location.since = now_iso()
            self._emit_event("geofence_arrive_home", {"zone": zone})
        elif action == "leave" and zone == "home":
            self._state.location.home = False
            self._state.location.zone = "away"
            self._state.location.since = now_iso()
            self._emit_event("geofence_leave_home", {"zone": zone})
        elif action == "enter":
            self._state.location.zone = zone
            self._state.location.home = (zone == "home")
            self._state.location.since = now_iso()
            self._emit_event("geofence_enter_zone", {"zone": zone})
        elif action == "leave":
            self._emit_event("geofence_leave_zone", {"zone": zone})

        self._state.last_updated = now_iso()
        self._state.event_id += 1
        save_state(self._state)

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
        """Poll Tuya devices and update state."""
        if self._tuya:
            # Poll bulb
            bulb_status = self._tuya.get_light_status()
            if bulb_status.get("success"):
                self._state.light.on = bulb_status.get("on", False)
                self._state.light.brightness = bulb_status.get("brightness", 0)
                if "bulb" not in self._state.devices:
                    self._state.devices["tuya_bulb"] = DeviceHealth()
                self._state.devices["tuya_bulb"].online = True
                self._state.devices["tuya_bulb"].last_poll = now_iso()
            else:
                if "tuya_bulb" in self._state.devices:
                    self._state.devices["tuya_bulb"].online = False

            # Poll HE20
            he20_status = self._tuya.get_mmwave_status()
            if he20_status.get("success"):
                self._state.mmwave.occupied = he20_status.get("occupied", False)
                if he20_status.get("occupied"):
                    self._state.mmwave.last_seen = now_iso()
                if "he20" not in self._state.devices:
                    self._state.devices["tuya_he20"] = DeviceHealth()
                self._state.devices["tuya_he20"].online = True
                self._state.devices["tuya_he20"].last_poll = now_iso()
            else:
                if "tuya_he20" in self._state.devices:
                    self._state.devices["tuya_he20"].online = False

        # Publish state via MQTT
        if self._mqtt:
            self._mqtt.publish_state(self._state.to_dict())

    def _check_exit_timeout(self) -> bool:
        """Check if presence has been absent long enough to clear."""
        exit_timeout = self._config.get("esp32", {}).get("exit_timeout", 60)
        if self._state.presence.last_seen is None:
            return True
        try:
            from datetime import datetime, timezone
            last = datetime.fromisoformat(self._state.presence.last_seen.replace("Z", "+00:00"))
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
                for mode_name in ("reading", "focus", "relax"):
                    if getattr(self._state.modes, mode_name, False):
                        scene = self._scenes.get(mode_name, {})
                        self.set_light(
                            on=True,
                            brightness=scene.get("brightness", 70),
                            color_temp=scene.get("color_temp"),
                            rgb=scene.get("rgb"),
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
                # Delayed mode change (work return settle)
                threading.Timer(delay, lambda: self.set_mode(mode)).start()
            else:
                self.set_mode(mode)

        elif action.type == "set_flag":
            for key, val in action.params.items():
                if hasattr(self._state.flags, key):
                    setattr(self._state.flags, key, val)
            self._state.flags.last_reset = now_iso()
            self._state.last_updated = now_iso()
            self._state.event_id += 1
            save_state(self._state)

    def set_mode(self, mode: str) -> None:
        """Set the room mode."""
        # Cancel all scene modes first
        for m in ("reading", "focus", "relax", "sleep", "alarm"):
            setattr(self._state.modes, m, False)

        if mode == "off":
            self.set_light(on=False)
        elif mode == "sleep":
            self._state.modes.sleep = True
            self.set_light(on=False)
            self._state.flags.evening_sleep_done_today = True
        elif mode == "alarm":
            self._state.modes.alarm = True
            scene = self._scenes.get("alarm", {})
            self.set_light(
                on=True, brightness=100, color_temp=6500,
                flash=scene.get("flash", True),
                flash_interval_ms=scene.get("flash_interval", 500),
            )
            if self._scheduler:
                self._scheduler.notify_alarm_started()
            self._emit_event("mode_changed", {"mode": "alarm"})
            self._state.last_updated = now_iso()
            self._state.event_id += 1
            save_state(self._state)
            return
        elif mode in self._scenes:
            setattr(self._state.modes, mode, True)
            scene = self._scenes[mode]
            self.set_light(
                on=True,
                brightness=scene.get("brightness", 70),
                color_temp=scene.get("color_temp"),
                rgb=scene.get("rgb"),
            )
        else:
            logger.warning("Unknown mode: %s", mode)

        self._emit_event("mode_changed", {"mode": mode})
        self._state.last_updated = now_iso()
        self._state.event_id += 1
        save_state(self._state)

    def set_light(
        self,
        on: Optional[bool] = None,
        brightness: Optional[int] = None,
        color_temp: Optional[int] = None,
        rgb: Optional[list] = None,
        flash: bool = False,
        flash_interval_ms: int = 500,
    ) -> None:
        """Set the light state."""
        if on is not None:
            self._state.light.on = on
        if brightness is not None:
            self._state.light.brightness = brightness
        if color_temp is not None:
            self._state.light.color_temp = color_temp
            self._state.light.rgb = None  # color temp mode
        if rgb is not None:
            self._state.light.rgb = rgb

        # Determine scene name
        if not self._state.light.on:
            self._state.light.scene = "off"
        elif flash:
            self._state.light.scene = "alarm"
        elif rgb is not None:
            self._state.light.scene = "custom"
        else:
            self._state.light.scene = "custom"

        # Control the physical device
        if self._tuya:
            self._tuya.set_light(
                on=on, brightness=brightness, color_temp=color_temp, rgb=rgb,
                flash=flash, flash_interval_ms=flash_interval_ms,
            )
        else:
            logger.warning("No Tuya controller — light command not sent to device")

        self._state.last_updated = now_iso()
        self._state.event_id += 1
        save_state(self._state)

    def cancel_sleep(self) -> None:
        """Cancel sleep mode and restore previous state."""
        was_evening = self._state.flags.evening_sleep_done_today
        was_work = self._state.flags.work_sleep_done_today

        self._state.modes.sleep = False

        if was_work:
            self._state.flags.work_sleep_cancel_today = True
            self._emit_event("sleep_cancelled", {"reason": "work_return"})
        elif was_evening:
            self._state.flags.evening_sleep_cancel_today = True
            self._emit_event("sleep_cancelled", {"reason": "evening"})

        self.set_light(on=True)
        self._state.last_updated = now_iso()
        self._state.event_id += 1
        save_state(self._state)
        logger.info("Sleep mode cancelled")

    def get_status(self) -> Dict[str, Any]:
        """Get runtime status for diagnostics."""
        return {
            "running": self._running,
            "rpc_port": _rpc_port(),
            "mqtt_connected": self._mqtt.connected if self._mqtt else False,
            "tuya_available": self._tuya is not None,
            "state_event_id": self._state.event_id,
        }

    def _cleanup(self) -> None:
        """Clean shutdown of all components."""
        logger.info("Smart room runtime shutting down...")
        if self._scheduler:
            self._scheduler.stop()
        if self._mqtt:
            self._mqtt.stop()
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

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
