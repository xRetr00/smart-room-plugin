"""Health monitoring for smart_room devices.

Checks MQTT broker connectivity, ESP32 staleness, Tuya device availability,
and runtime process health. Used by smart_room_health and smart_room_diagnostic tools.
"""

from __future__ import annotations

import logging
import importlib.util
import os
from typing import Any, Dict

from plugins.smart_room.runtime.models import RoomState, now_iso

logger = logging.getLogger(__name__)


def _dependency_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def check_device_health(state: RoomState, config: Dict[str, Any]) -> Dict[str, Any]:
    """Build a health report from current state + config."""
    report: Dict[str, Any] = {
        "timestamp": now_iso(),
        "devices": {},
        "mqtt": {"connected": False},
        "runtime": {"alive": True},
        "dependencies": {
            "paho_mqtt": _dependency_available("paho.mqtt.client"),
            "tinytuya": _dependency_available("tinytuya"),
            "yaml": _dependency_available("yaml"),
        },
    }

    mqtt_cfg = config.get("mqtt", {})
    report["mqtt"].update({
        "broker": mqtt_cfg.get("broker", "127.0.0.1"),
        "port": mqtt_cfg.get("port", 1883),
        "auth_configured": bool(os.getenv("SMART_ROOM_MQTT_USERNAME")),
    })

    # Device health from state
    for name, dev in state.devices.items():
        report["devices"][name] = {
            "online": dev.online,
            "ip": dev.ip,
            "last_seen": dev.last_seen,
            "last_poll": dev.last_poll,
        }

    # ESP32 staleness check
    esp32_cfg = config.get("esp32", {})
    report["devices"]["esp32"] = report["devices"].get("esp32", {
        "online": False,
        "ip": None,
        "last_seen": None,
        "last_poll": None,
    })
    report["devices"]["esp32"]["configured"] = bool(
        esp32_cfg.get("presence_topic") and esp32_cfg.get("owner_device_id")
    )
    esp32_state = state.devices.get("esp32")
    if esp32_state:
        staleness_threshold = int(esp32_cfg.get("stale_seconds", 120))
        if esp32_state.last_seen:
            try:
                from datetime import datetime
                last = datetime.fromisoformat(esp32_state.last_seen.replace("Z", "+00:00"))
                elapsed = (datetime.now(last.tzinfo) - last).total_seconds()
                report["devices"]["esp32"]["staleness_seconds"] = elapsed
                report["devices"]["esp32"]["stale"] = elapsed > staleness_threshold
            except Exception:
                pass

    # Tuya device availability
    tuya_cfg = config.get("tuya", {})
    for name in ("bulb", "he20"):
        dev_key = f"tuya_{name}"
        dev_state = state.devices.get(dev_key)
        if dev_state:
            report["devices"][dev_key] = {
                "online": dev_state.online,
                "ip": dev_state.ip,
                "last_poll": dev_state.last_poll,
            }
        else:
            report["devices"][dev_key] = {
                "online": False,
                "ip": None,
                "last_poll": None,
            }
        # Check if configured
        dev_cfg = tuya_cfg.get(name, {})
        secret = os.getenv(
            "SMART_ROOM_TUYA_BULB_KEY" if name == "bulb" else "SMART_ROOM_TUYA_HE20_KEY",
            "",
        )
        report["devices"][dev_key]["configured"] = bool(
            dev_cfg.get("ip") and dev_cfg.get("device_id") and secret
        )
        report["devices"][dev_key]["ip"] = dev_cfg.get("ip") or report["devices"][dev_key].get("ip")

    return report
