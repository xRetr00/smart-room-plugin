"""smart_room plugin — world-aware room control for Marvi.

Ships as plugins/smart_room/ in the Marvi repo. The runtime runs as ONE
child process spawned and supervised by the plugin when the gateway starts
(mirrors google_meet's process_manager pattern).

Key design principles (from v0.3 spec):
1. Room state is world-awareness context, NOT memory.
2. The runtime NEVER writes memory/Honcho.
3. BLE silence is NOT absence (iPhone deep-sleep fusion axioms).
4. Marvi controls the room via tool handlers → runtime RPC bridge.
"""

from __future__ import annotations

import platform
import logging

from plugins.smart_room.tools import (
    SMART_ROOM_STATE_SCHEMA,
    SMART_ROOM_SET_MODE_SCHEMA,
    SMART_ROOM_SET_LIGHT_SCHEMA,
    SMART_ROOM_CANCEL_SLEEP_SCHEMA,
    SMART_ROOM_OVERRIDE_SCHEMA,
    SMART_ROOM_HEALTH_SCHEMA,
    SMART_ROOM_DIAGNOSTIC_SCHEMA,
    SMART_ROOM_ALARM_SCHEMA,
    check_smart_room_requirements,
    handle_smart_room_state,
    handle_smart_room_set_mode,
    handle_smart_room_set_light,
    handle_smart_room_cancel_sleep,
    handle_smart_room_override,
    handle_smart_room_health,
    handle_smart_room_diagnostic,
    handle_smart_room_alarm,
)
from plugins.smart_room.context import build_context_line
from plugins.smart_room import process_manager
from plugins.smart_room.runtime.state_store import load_config

logger = logging.getLogger(__name__)

_TOOLS = (
    ("smart_room_state",       SMART_ROOM_STATE_SCHEMA,       handle_smart_room_state,       "🏠"),
    ("smart_room_set_mode",    SMART_ROOM_SET_MODE_SCHEMA,    handle_smart_room_set_mode,    "🎨"),
    ("smart_room_set_light",   SMART_ROOM_SET_LIGHT_SCHEMA,   handle_smart_room_set_light,   "💡"),
    ("smart_room_cancel_sleep", SMART_ROOM_CANCEL_SLEEP_SCHEMA, handle_smart_room_cancel_sleep, "🌙"),
    ("smart_room_override",    SMART_ROOM_OVERRIDE_SCHEMA,    handle_smart_room_override,    "🔧"),
    ("smart_room_health",      SMART_ROOM_HEALTH_SCHEMA,      handle_smart_room_health,      "💚"),
    ("smart_room_diagnostic",  SMART_ROOM_DIAGNOSTIC_SCHEMA,  handle_smart_room_diagnostic,  "🔍"),
    ("smart_room_alarm",       SMART_ROOM_ALARM_SCHEMA,       handle_smart_room_alarm,       "⏰"),
)


def _on_gateway_start(**_kwargs) -> None:
    config = load_config()
    if config.get("enabled", False):
        process_manager.start_supervisor(config)


def _on_gateway_stop(**_kwargs) -> None:
    process_manager.stop_supervisor()


def register(ctx) -> None:
    """Register tools, CLI, and lifecycle hooks.

    Called once by the plugin loader when the plugin is enabled via
    ``plugins.enabled`` in config.yaml.
    """
    system = platform.system().lower()
    if system != "windows":
        logger.info("smart_room plugin: platform=%s not supported (Windows only)", system)
        return

    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="smart_room",
            schema=schema,
            handler=handler,
            check_fn=check_smart_room_requirements,
            emoji=emoji,
        )

    # Register context provider for ambient session context
    ctx.register_context_provider(
        name="smart_room",
        handler=build_context_line,
    )
    ctx.register_hook("on_gateway_start", _on_gateway_start)
    ctx.register_hook("on_gateway_stop", _on_gateway_stop)
