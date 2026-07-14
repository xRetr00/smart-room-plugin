"""Command router — handles RPC requests from the bridge.

Routes method calls to the appropriate runtime component:
  get_state, set_mode, set_light, cancel_sleep, set_override,
  get_health, get_diagnostic
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from plugins.smart_room.runtime.models import RoomState, now_iso
from plugins.smart_room.runtime.state_store import save_state
from plugins.smart_room.runtime.health import check_device_health

logger = logging.getLogger(__name__)


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
            return handler(params)
        except Exception as e:
            logger.error("Command %s failed: %s", method, e)
            return {"success": False, "error": str(e)}

    def _handle_get_state(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._state.last_updated = now_iso()
        self._state.event_id += 1
        save_state(self._state)
        return {"success": True, "state": self._state.to_dict()}

    def _handle_set_mode(self, params: Dict[str, Any]) -> Dict[str, Any]:
        mode = params.get("mode", "off")
        self._runtime.set_mode(mode)
        return {"success": True, "mode": mode, "state": self._state.to_dict()}

    def _handle_set_light(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._runtime.set_light(**params)
        return {"success": True, "light": self._state.light.__dict__, "state": self._state.to_dict()}

    def _handle_cancel_sleep(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._runtime.cancel_sleep()
        return {"success": True, "state": self._state.to_dict()}

    def _handle_set_override(self, params: Dict[str, Any]) -> Dict[str, Any]:
        enabled = params.get("enabled", False)
        self._state.modes.manual_override = enabled
        self._state.last_updated = now_iso()
        self._state.event_id += 1
        save_state(self._state)
        logger.info("Manual override %s", "enabled" if enabled else "disabled")
        return {"success": True, "override": enabled, "state": self._state.to_dict()}

    def _handle_get_health(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True, "health": check_device_health(self._state, self._config)}

    def _handle_get_diagnostic(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "diagnostic": {
                "state": self._state.to_dict(),
                "config": {k: "***" if "key" in k.lower() else v for k, v in self._config.items()},
                "health": check_device_health(self._state, self._config),
                "runtime": self._runtime.get_status(),
            },
        }
