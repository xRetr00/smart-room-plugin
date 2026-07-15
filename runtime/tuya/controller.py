"""Tuya LAN device control via tinytuya.

Fallback control path — when ESP32 is not controlling devices directly,
the PC runtime can control Tuya devices via tinytuya over local network.

Per v0.3 §E: the primary control path is ESP32 (EspTuya), the PC path
is the fallback/override. Both use the same Tuya LAN protocol.
"""

from __future__ import annotations

import colorsys
import logging
import os
import threading
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
        self._flash_stop = threading.Event()
        self._flash_thread: Optional[threading.Thread] = None
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
            key = os.getenv(
                "SMART_ROOM_TUYA_BULB_KEY" if name == "bulb" else "SMART_ROOM_TUYA_HE20_KEY",
                "",
            )
            dev_id = dev_cfg.get("device_id", "")
            protocol = dev_cfg.get("protocol", "3.3")

            if ip and key and dev_id:
                try:
                    dev = tinytuya.Device(
                        dev_id=dev_id,
                        address=ip,
                        local_key=key,
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
            if not flash:
                self._flash_stop.set()
            bulb_cfg = (self._config.get("tuya", {}).get("bulb", {}))
            dp = bulb_cfg.get("dps", {})
            switch_dp = str(dp.get("switch", 1))
            mode_dp = str(dp.get("mode", 21))
            dps: Dict[str, Any] = {}
            if on is not None:
                dps[switch_dp] = on
            if brightness is not None and rgb is None:
                brightness_max = int(bulb_cfg.get("brightness_max", 255))
                dps[str(dp.get("brightness", 2))] = int(brightness * brightness_max / 100)
            if color_temp is not None and rgb is None:
                color_temp_max = int(bulb_cfg.get("color_temp_max", 255))
                tuya_ct = int((color_temp - 2200) / (6500 - 2200) * color_temp_max)
                dps[str(dp.get("color_temp", 3))] = tuya_ct
                dps[mode_dp] = "white"
            if rgb is not None:
                r, g, b = rgb[0], rgb[1], rgb[2]
                hue, saturation, value = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
                if brightness is not None:
                    value = brightness / 100
                dps[mode_dp] = "colour"
                dps[str(dp.get("color", 5))] = (
                    f"{round(hue * 360):04x}{round(saturation * 1000):04x}"
                    f"{round(value * 1000):04x}"
                )

            if dps:
                setter = getattr(dev, "set_multiple_values", None) or getattr(
                    dev, "set_multiple_dps", None
                )
                if setter is None:
                    raise RuntimeError("installed tinytuya has no multi-DP setter")
                response = setter(dps)
                if isinstance(response, dict) and (
                    response.get("Error") or response.get("Err")
                ):
                    raise RuntimeError(str(response.get("Error") or response.get("Err")))
            if flash:
                self._start_flash(dev, switch_dp, flash_interval_ms)
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
            if not isinstance(dps, dict) or not dps:
                return {"success": False, "error": status.get("Error", "empty DPS response"), "online": False}
            mapping = self._config.get("tuya", {}).get("bulb", {}).get("dps", {})
            switch_dp = str(mapping.get("switch", 1))
            brightness_dp = str(mapping.get("brightness", 2))
            return {
                "success": True,
                "on": dps.get(switch_dp, dps.get(int(switch_dp), False)),
                "brightness": int(
                    int(dps.get(brightness_dp, dps.get(int(brightness_dp), "0")))
                    * 100
                    / int(self._config.get("tuya", {}).get("bulb", {}).get("brightness_max", 255))
                ),
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
            if not isinstance(dps, dict) or not dps:
                return {"success": False, "error": status.get("Error", "empty DPS response"), "online": False}
            presence_dp = str(self._config.get("tuya", {}).get("he20", {}).get("presence_dp", 1))
            raw = dps.get(presence_dp, dps.get(int(presence_dp), False))
            if isinstance(raw, str):
                occupied_values = self._config.get("tuya", {}).get("he20", {}).get(
                    "occupied_values", ["true", "1", "presence", "occupied", "pir", "human"]
                )
                occupied = raw.strip().lower() in {str(value).lower() for value in occupied_values}
            else:
                occupied = bool(raw)
            # HE20 presence DP is typically DP 1 (occupancy)
            return {
                "success": True,
                "occupied": occupied,
                "online": True,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "online": False}

    def refresh(self) -> None:
        """Reconnect all devices (call on config change)."""
        self._devices.clear()
        self._connect_devices()

    def stop(self) -> None:
        """Stop any active alarm flash loop."""
        self._flash_stop.set()
        if self._flash_thread and self._flash_thread.is_alive():
            self._flash_thread.join(timeout=1)

    def _start_flash(self, dev: Any, switch_dp: str, interval_ms: int) -> None:
        self._flash_stop.set()
        if self._flash_thread and self._flash_thread.is_alive():
            self._flash_thread.join(timeout=1)
        self._flash_stop = threading.Event()

        def flash_loop() -> None:
            value = False
            while not self._flash_stop.wait(max(0.1, interval_ms / 1000)):
                value = not value
                try:
                    dev.set_value(int(switch_dp) if switch_dp.isdigit() else switch_dp, value)
                except Exception:
                    logger.debug("Tuya alarm flash failed", exc_info=True)
                    break

        self._flash_thread = threading.Thread(
            target=flash_loop, name="smart_room_alarm_flash", daemon=True
        )
        self._flash_thread.start()

    @property
    def available(self) -> bool:
        return bool(self._devices)
