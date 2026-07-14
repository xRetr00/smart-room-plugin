"""Agent-facing tools for the smart_room plugin.

Tools:
  smart_room_state         — full room snapshot (presence, light, modes, devices)
  smart_room_set_mode      — set mode: reading, focus, relax, sleep, alarm, off
  smart_room_set_light     — direct light control (on/off, brightness, color)
  smart_room_cancel_sleep  — cancel sleep mode, restore previous state
  smart_room_override      — toggle manual override (disables presence automations)
  smart_room_health        — device health check (online/offline, last seen)
  smart_room_diagnostic    — full diagnostic dump for troubleshooting

All handlers are non-blocking: they call the runtime via the bridge and return
structured JSON. If the runtime is down, they return DEVICE_TIMEOUT per v0.3.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from plugins.smart_room.bridge import call_runtime


# ---------------------------------------------------------------------------
# Runtime gate
# ---------------------------------------------------------------------------

def check_smart_room_requirements() -> bool:
    """Return True when the smart_room plugin can run."""
    import platform as _p
    if _p.system().lower() not in {"windows", "linux", "darwin"}:
        return False
    try:
        import paho.mqtt.client  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

SMART_ROOM_STATE_SCHEMA: Dict[str, Any] = {
    "name": "smart_room_state",
    "description": (
        "Get the full smart room state snapshot: presence (BLE+mmWave+geofence "
        "fusion), light state (on/off, brightness, color, scene), active mode, "
        "device health (bulb, HE20 sensor, ESP32), phone location, and flags."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

SMART_ROOM_SET_MODE_SCHEMA: Dict[str, Any] = {
    "name": "smart_room_set_mode",
    "description": (
        "Set the smart room mode. Modes are mutually exclusive. "
        "reading=warm 3000K 70%, focus=cool 5000K 100%, relax=warm amber 40%, "
        "sleep=lights off darkness enforced, alarm=flash bright white auto-expire, "
        "off=lights off no mode."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["reading", "focus", "relax", "sleep", "alarm", "off"],
                "description": "The mode to activate.",
            },
        },
        "required": ["mode"],
        "additionalProperties": False,
    },
}

SMART_ROOM_SET_LIGHT_SCHEMA: Dict[str, Any] = {
    "name": "smart_room_set_light",
    "description": (
        "Direct light control bypassing mode scenes. Set on/off, brightness "
        "(0-100), color temperature (2200K-6500K), or RGB color. "
        "At least one parameter must be provided."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "on": {"type": "boolean", "description": "Turn the light on or off."},
            "brightness": {"type": "integer", "description": "Brightness 0-100.", "minimum": 0, "maximum": 100},
            "color_temp": {"type": "integer", "description": "Color temperature in Kelvin (2200=warm, 6500=cool).", "minimum": 2200, "maximum": 6500},
            "rgb": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 255}, "description": "RGB color [R, G, B]."},
        },
        "additionalProperties": False,
    },
}

SMART_ROOM_CANCEL_SLEEP_SCHEMA: Dict[str, Any] = {
    "name": "smart_room_cancel_sleep",
    "description": "Cancel sleep mode and restore the previous state. Also marks sleep_cancel_today flags.",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}

SMART_ROOM_OVERRIDE_SCHEMA: Dict[str, Any] = {
    "name": "smart_room_override",
    "description": "Toggle manual override. When ON, presence-based automations are suppressed.",
    "parameters": {
        "type": "object",
        "properties": {"enabled": {"type": "boolean", "description": "True to enable override, false to disable."}},
        "required": ["enabled"],
        "additionalProperties": False,
    },
}

SMART_ROOM_HEALTH_SCHEMA: Dict[str, Any] = {
    "name": "smart_room_health",
    "description": "Check device health: online/offline status, last seen, MQTT connectivity, ESP32 staleness.",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}

SMART_ROOM_DIAGNOSTIC_SCHEMA: Dict[str, Any] = {
    "name": "smart_room_diagnostic",
    "description": "Full diagnostic dump: runtime state, MQTT connection, Tuya DPS, ESP32 telemetry, automation engine, recent events.",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}


# ---------------------------------------------------------------------------
# Handlers — all non-blocking via bridge.call_runtime()
# ---------------------------------------------------------------------------

def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _err(msg: str, **extra) -> str:
    return _json({"success": False, "error": msg, **extra})


def handle_smart_room_state(args: Dict[str, Any], **_kw) -> str:
    try:
        return _json(call_runtime("get_state", {}))
    except RuntimeError as e:
        return _err(str(e), code="RUNTIME_UNAVAILABLE")


def handle_smart_room_set_mode(args: Dict[str, Any], **_kw) -> str:
    mode = (args.get("mode") or "").strip().lower()
    if mode not in {"reading", "focus", "relax", "sleep", "alarm", "off"}:
        return _err(f"invalid mode: {mode!r}")
    try:
        return _json(call_runtime("set_mode", {"mode": mode}))
    except RuntimeError as e:
        return _err(str(e), code="RUNTIME_UNAVAILABLE")


def handle_smart_room_set_light(args: Dict[str, Any], **_kw) -> str:
    params: Dict[str, Any] = {}
    if "on" in args:
        params["on"] = bool(args["on"])
    if "brightness" in args:
        params["brightness"] = int(args["brightness"])
    if "color_temp" in args:
        params["color_temp"] = int(args["color_temp"])
    if "rgb" in args:
        params["rgb"] = list(args["rgb"])
    if not params:
        return _err("at least one parameter is required (on, brightness, color_temp, or rgb)")
    try:
        return _json(call_runtime("set_light", params))
    except RuntimeError as e:
        return _err(str(e), code="RUNTIME_UNAVAILABLE")


def handle_smart_room_cancel_sleep(args: Dict[str, Any], **_kw) -> str:
    try:
        return _json(call_runtime("cancel_sleep", {}))
    except RuntimeError as e:
        return _err(str(e), code="RUNTIME_UNAVAILABLE")


def handle_smart_room_override(args: Dict[str, Any], **_kw) -> str:
    enabled = bool(args.get("enabled"))
    try:
        return _json(call_runtime("set_override", {"enabled": enabled}))
    except RuntimeError as e:
        return _err(str(e), code="RUNTIME_UNAVAILABLE")


def handle_smart_room_health(args: Dict[str, Any], **_kw) -> str:
    try:
        return _json(call_runtime("get_health", {}))
    except RuntimeError as e:
        return _err(str(e), code="RUNTIME_UNAVAILABLE")


def handle_smart_room_diagnostic(args: Dict[str, Any], **_kw) -> str:
    try:
        return _json(call_runtime("get_diagnostic", {}))
    except RuntimeError as e:
        return _err(str(e), code="RUNTIME_UNAVAILABLE")
