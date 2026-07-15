"""Command router — handles RPC requests from the bridge.

Routes method calls to the appropriate runtime component:
  get_state, set_mode, set_light, cancel_sleep, set_override,
  get_health, get_diagnostic
"""

from __future__ import annotations

import logging
import threading
from contextlib import nullcontext
from typing import Any, Dict

from plugins.smart_room.runtime.models import RoomState
from plugins.smart_room.runtime.health import check_device_health

logger = logging.getLogger(__name__)


def _redact_config(value: Any, key: str = "") -> Any:
    if any(marker in key.lower() for marker in ("key", "secret", "password", "token")):
        return "***"
    if isinstance(value, dict):
        return {name: _redact_config(item, str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_redact_config(item) for item in value]
    return value


class CommandRouter:
    """Routes RPC method calls to runtime actions."""

    def __init__(self, state: RoomState, config: Dict[str, Any], runtime: "Runtime"):
        self._state = state
        self._config = config
        self._runtime = runtime

    def dispatch(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Route an RPC method to the handler."""
        handler = getattr(self, f"_handle_{method}", None)
        if handler is None:
            return {"success": False, "error": f"unknown method: {method}"}
        try:
            if method == "ping":
                return handler(params)
            lock = getattr(self._runtime, "_state_lock", None)
            with lock if lock is not None else nullcontext():
                return handler(params)
        except Exception as e:
            logger.error("Command %s failed: %s", method, e)
            return {"success": False, "error": str(e)}

    def _handle_get_state(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True, "state": self._state.to_dict()}

    def _handle_ping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True}

    def _handle_set_mode(self, params: Dict[str, Any]) -> Dict[str, Any]:
        mode = params.get("mode", "off")
        if mode not in {"reading", "focus", "relax", "sleep", "alarm", "off"}:
            return {"success": False, "error": f"invalid mode: {mode}"}
        self._runtime.set_mode(mode)
        return {"success": True, "mode": mode, "state": self._state.to_dict()}

    def _handle_set_light(self, params: Dict[str, Any]) -> Dict[str, Any]:
        result = self._runtime.set_light(**params, manual=True)
        ack = result if isinstance(result, dict) else {"success": True}
        return {**ack, "light": self._state.light.__dict__, "state": self._state.to_dict()}

    def _handle_cancel_sleep(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._runtime.cancel_sleep()
        return {"success": True, "state": self._state.to_dict()}

    def _handle_test_welcome(self, params: Dict[str, Any]) -> Dict[str, Any]:
        audience = str(params.get("audience", ""))
        if audience not in {"owner", "guest"}:
            return {"success": False, "error": "audience must be owner or guest"}
        self._runtime.test_welcome(audience)
        return {"success": True, "audience": audience}

    def _handle_set_override(self, params: Dict[str, Any]) -> Dict[str, Any]:
        enabled = bool(params.get("enabled", False))
        self._runtime.set_override(enabled)
        logger.info("Manual override %s", "enabled" if enabled else "disabled")
        return {"success": True, "override": enabled, "state": self._state.to_dict()}

    def _handle_get_health(self, params: Dict[str, Any]) -> Dict[str, Any]:
        health = check_device_health(self._state, self._config)
        runtime = self._runtime.get_status()
        health["mqtt"]["connected"] = bool(runtime.get("mqtt_connected"))
        health["sound_events"] = runtime.get(
            "sound_events", {"enabled": False, "running": False}
        )
        return {"success": True, "health": health}

    def _handle_phone_location_changed(self, params: Dict[str, Any]) -> Dict[str, Any]:
        required = {"who", "transition", "zone", "at", "delivery_id", "source"}
        missing = sorted(required.difference(params))
        if missing:
            return {"success": False, "error": f"missing fields: {', '.join(missing)}"}
        return self._runtime.phone_location_changed(**{key: params[key] for key in required})

    def _handle_shutdown(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Let the RPC thread flush this acknowledgement before the daemon's
        # main thread begins teardown.
        timer = threading.Timer(0.1, self._runtime.stop)
        timer.daemon = True
        timer.start()
        return {"success": True}

    def _handle_get_diagnostic(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "diagnostic": {
                "state": self._state.to_dict(),
                "config": _redact_config(self._config),
                "health": check_device_health(self._state, self._config),
                "runtime": self._runtime.get_status(),
            },
        }
