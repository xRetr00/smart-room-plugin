"""Automation engine for adaptive light, modes, schedules, and daily flags.

Each automation is a rule evaluated by the engine when events arrive.
Rules are pure functions: (state, event) → list[actions].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from plugins.smart_room.runtime.models import RoomState, now_iso

logger = logging.getLogger(__name__)


def _inside_window(start: str, end: str, now: Optional[datetime] = None) -> bool:
    current = now or datetime.now()
    try:
        start_h, start_m = map(int, start.split(":"))
        end_h, end_m = map(int, end.split(":"))
    except (AttributeError, TypeError, ValueError):
        return True
    minute = current.hour * 60 + current.minute
    lower, upper = start_h * 60 + start_m, end_h * 60 + end_m
    return lower <= minute <= upper if lower <= upper else minute >= lower or minute <= upper


@dataclass
class Action:
    """An action the engine wants the runtime to execute."""
    type: str  # "set_light", "set_mode", "set_flag", "turn_off", "turn_on"
    params: Dict[str, Any]


def evaluate_automations(
    state: RoomState,
    event: Dict[str, Any],
    config: Dict[str, Any],
) -> List[Action]:
    """Evaluate all automations against the current state + event.

    Returns a list of actions to execute. Actions are deduplicated by type
    (last one wins per type).
    """
    actions: List[Action] = []
    event_type = event.get("type", "")
    auto_cfg = config.get("automations", {})

    # Don't run presence-based automations in sleep/alarm mode or with override
    suppress_on = state.modes.active_mode in {"sleep", "alarm"} or state.modes.manual_override == "hold_off"
    suppress_off = state.modes.active_mode in {"sleep", "alarm"} or state.modes.manual_override == "hold_on"

    # --- 4.1 Adaptive Light On ---
    if auto_cfg.get("adaptive_light", {}).get("enabled", True):
        if event_type == "presence_detected" and not suppress_on:
            actions.append(Action(type="turn_on", params={"restore_scene": True}))

    # --- 4.2 Light Off After Clear ---
    if auto_cfg.get("adaptive_light", {}).get("enabled", True):
        if event_type == "presence_cleared" and not suppress_off:
            actions.append(Action(type="turn_off", params={}))

    # --- 4.3 Sleep Mode Enforce Darkness ---
    if event_type == "mode_changed" and event.get("mode") == "sleep":
        actions.append(Action(type="turn_off", params={"reason": "sleep_mode"}))

    # --- 4.4 Alarm Mode Force Bright Light ---
    if event_type == "mode_changed" and event.get("mode") == "alarm":
        alarm_cfg = auto_cfg.get("alarm", {})
        actions.append(Action(type="set_light", params={
            "on": True,
            "brightness": 100,
            "color_temp": 6500,
            "flash": True,
            "flash_interval_ms": alarm_cfg.get("flash_interval_ms", 500),
        }))

    # --- 4.6 Clear Alarm After Duration ---
    if event_type == "schedule_alarm":
        alarm = event.get("alarm") or {}
        actions.append(Action(type="set_mode", params={
            "mode": "alarm",
            "reason": "schedule",
            "alarm_id": alarm.get("id"),
            "alarm_name": alarm.get("name"),
            "duration_minutes": alarm.get("duration_minutes", 30),
        }))

    if event_type == "alarm_flash_finished":
        actions.append(Action(type="set_light", params={
            "on": True, "brightness": 100, "color_temp": 6500, "flash": False,
        }))

    if event_type == "alarm_duration_expired":
        actions.append(Action(type="ack_alarm", params={"reason": "expired"}))

    # --- 4.7 Work Return Sleep ---
    work_cfg = auto_cfg.get("work_return", {})
    if work_cfg.get("enabled", True) and event_type in {"geofence_arrive_home", "ble_arrive_fallback"}:
        start = work_cfg.get("arrival_window_start", work_cfg.get("work_hours_start", "00:00"))
        end = work_cfg.get("arrival_window_end", work_cfg.get("work_hours_end", "23:59"))
        if (
            _inside_window(start, end)
            and not state.flags.work_sleep_done_today
            and not state.flags.work_sleep_cancel_today
        ):
            actions.append(Action(type="set_mode", params={
                "mode": "sleep",
                "reason": "work_return",
                "delay": work_cfg.get("settle_delay", 300),
            }))

    # --- 4.8 Evening Sleep ---
    evening_cfg = auto_cfg.get("evening_sleep", {})
    if evening_cfg.get("enabled", True) and event_type == "schedule_evening_sleep":
        if not state.flags.evening_sleep_done_today and not state.flags.evening_sleep_cancel_today:
            actions.append(Action(type="set_mode", params={"mode": "sleep", "reason": "evening"}))

    # --- 4.12 Reset Daily Mode Flags ---
    if event_type == "schedule_daily_reset":
        actions.append(Action(type="set_flag", params={
            "work_sleep_done_today": False,
            "work_sleep_cancel_today": False,
            "evening_sleep_done_today": False,
            "evening_sleep_cancel_today": False,
        }))

    # --- 4.13/4.14 Cancel sleep flags when closed early ---
    if event_type == "sleep_cancelled":
        # Determine which sleep type to cancel
        if event.get("reason") == "work_return":
            actions.append(Action(type="set_flag", params={"work_sleep_cancel_today": True}))
        elif event.get("reason") == "evening":
            actions.append(Action(type="set_flag", params={"evening_sleep_cancel_today": True}))

    # Deduplicate: last action per type wins
    seen: Dict[str, Action] = {}
    for a in actions:
        seen[a.type] = a

    return list(seen.values())
