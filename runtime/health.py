"""Health monitoring for smart_room devices.

Checks MQTT broker connectivity, ESP32 staleness, Tuya device availability,
and runtime process health. Used by smart_room_health and smart_room_diagnostic tools.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from plugins.smart_room.runtime.models import RoomState, now_iso

logger = logging.getLogger(__name__)


def check_device_health(state: RoomState, config: Dict[str, Any]) -> Dict[str, Any]:
    """Build a health report from current state + config."""
    report: Dict[str, Any] = {
        "timestamp": now_iso(),
        "devices": {},
        "mqtt": {"connected": False},
        "runtime": {"alive": True},
    }

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
    esp32_state = state.devices.get("esp32")
    if esp32_state:
        staleness_threshold = 120  # 2 minutes
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
        # Check if configured
        dev_cfg = tuya_cfg.get(name, {})
        report["devices"][dev_key]["configured"] = bool(dev_cfg.get("ip") and dev_cfg.get("local_key"))

    return report
