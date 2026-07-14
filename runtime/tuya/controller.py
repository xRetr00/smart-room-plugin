"""Tuya LAN device control via tinytuya.

Fallback control path — when ESP32 is not controlling devices directly,
the PC runtime can control Tuya devices via tinytuya over local network.

Per v0.3 §E: the primary control path is ESP32 (EspTuya), the PC path
is the fallback/override. Both use the same Tuya LAN protocol.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import tinytuya
    HAS_TINYTUYA = True
except ImportError:
    HAS_TINYTUYA = False
    tinytuya = None  # type: ignore


class TuyaController:
    """Wrapper for tinytuya device control."""

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._devices: Dict[str, Any] = {}  # name -> tinytuya.Device
        self._connect_devices()

    def _connect_devices(self) -> None:
        """Initialize tinytuya device connections."""
        if not HAS_TINYTUYA:
            logger.warning("tinytuya not installed — Tuya control disabled")
            return

        tuya_cfg = self._config.get("tuya", {})
        for name in ("bulb", "he20"):
            dev_cfg = tuya_cfg.get(name, {})
            ip = dev_cfg.get("ip")
            key = dev_cfg.get("local_key")
            dev_id = dev_cfg.get("device_id", "")
            protocol = dev_cfg.get("protocol", "3.3")

            if ip and key:
                try:
                    dev = tinytuya.Device(
                        dev_id=dev_id,
                        address=ip,
                    )
                    dev.set_version(float(protocol))
                    self._devices[name] = dev
                    logger.info("Tuya device '%s' connected at %s", name, ip)
                except Exception as e:
                    logger.error("Failed to connect Tuya device '%s': %s", name, e)

    def set_light(
        self,
        on: Optional[bool] = None,
        brightness: Optional[int] = None,
        color_temp: Optional[int] = None,
        rgb: Optional[list] = None,
        flash: bool = False,
        flash_interval_ms: int = 500,
    ) -> Dict[str, Any]:
        """Control the RGBCW bulb."""
        dev = self._devices.get("bulb")
        if dev is None:
            return {"success": False, "error": "bulb not connected or not configured"}

        try:
            dps: Dict[str, Any] = {}
            if on is not None:
                dps["1"] = on  # DP 1 = switch
            if brightness is not None:
                dps["2"] = str(int(brightness * 255 / 100))  # DP 2 = brightness (0-255)
            if color_temp is not None:
                # Tuya color temp is 0-255 (warm to cool)
                tuya_ct = int((color_temp - 2200) / (6500 - 2200) * 255)
                dps["3"] = str(tuya_ct)  # DP 3 = color temperature
            if rgb is not None:
                r, g, b = rgb[0], rgb[1], rgb[2]
                # Tuya expects "rrrgggbb" hex format
                dps["5"] = f"{r:02x}{g:02x}{b:02x}"  # DP 5 = color

            if dps:
                dev.set_multiple_dps(dps)
            return {"success": True, "dps": dps}
        except Exception as e:
            logger.error("Tuya set_light failed: %s", e)
            return {"success": False, "error": str(e)}

    def get_light_status(self) -> Dict[str, Any]:
        """Poll the bulb for current state."""
        dev = self._devices.get("bulb")
        if dev is None:
            return {"success": False, "error": "bulb not connected"}
        try:
            status = dev.status()
            dps = status.get("dps", {})
            return {
                "success": True,
                "on": dps.get("1", False),
                "brightness": int(int(dps.get("2", "0")) * 100 / 255) if "2" in dps else 0,
                "online": True,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "online": False}

    def get_mmwave_status(self) -> Dict[str, Any]:
        """Poll HE20 sensor for presence."""
        dev = self._devices.get("he20")
        if dev is None:
            return {"success": False, "error": "he20 not connected"}
        try:
            status = dev.status()
            dps = status.get("dps", {})
            # HE20 presence DP is typically DP 1 (occupancy)
            return {
                "success": True,
                "occupied": dps.get("1", False),
                "online": True,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "online": False}

    def refresh(self) -> None:
        """Reconnect all devices (call on config change)."""
        self._devices.clear()
        self._connect_devices()
