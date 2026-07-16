"""Small persistent-aware scheduler for room alarms and automations."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Optional

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(
        self,
        config: Dict[str, Any],
        emit_event: Callable[[str, Dict[str, Any]], None],
        get_alarms: Optional[Callable[[], Iterable[Dict[str, Any]]]] = None,
        get_active_alarm: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
    ):
        self._config = config
        self._emit = emit_event
        self._get_alarms = get_alarms or (lambda: ())
        self._get_active_alarm = get_active_alarm or (lambda: None)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_fired: Dict[str, str] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="smart_room_scheduler")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._check_triggers()
            except Exception:
                logger.exception("Scheduler check failed")
            self._stop.wait(5)

    def _check_triggers(self) -> None:
        now = datetime.now().astimezone()
        minute_key = now.strftime("%Y-%m-%d %H:%M")

        for alarm in self._get_alarms():
            if not alarm.get("enabled", True) or alarm.get("time") != now.strftime("%H:%M"):
                continue
            recurrence = alarm.get("recurrence", "daily")
            if recurrence == "once" and alarm.get("date") != now.strftime("%Y-%m-%d"):
                continue
            alarm_id = str(alarm.get("id", ""))
            if not alarm_id or self._last_fired.get(alarm_id) == minute_key:
                continue
            try:
                last_fired = datetime.fromisoformat(str(alarm.get("last_fired_at")).replace("Z", "+00:00"))
                if last_fired.astimezone(now.tzinfo).strftime("%Y-%m-%d %H:%M") == minute_key:
                    continue
            except (TypeError, ValueError):
                pass
            self._last_fired[alarm_id] = minute_key
            self._emit("schedule_alarm", {"alarm": alarm})

        auto = self._config.get("automations", {})

        def fire_once(name: str, configured_time: str, event: str) -> None:
            try:
                hour, minute = map(int, configured_time.split(":"))
            except (AttributeError, TypeError, ValueError):
                return
            if now.hour == hour and now.minute == minute and self._last_fired.get(name) != minute_key:
                self._last_fired[name] = minute_key
                self._emit(event, {"time": configured_time})

        evening = auto.get("evening_sleep", {})
        if evening.get("enabled", True):
            fire_once("evening_sleep", evening.get("time", "18:00"), "schedule_evening_sleep")
        reset = auto.get("daily_reset", "00:00")
        fire_once("daily_reset", reset.get("time", "00:00") if isinstance(reset, dict) else reset, "schedule_daily_reset")

        active = self._get_active_alarm()
        if not active:
            return
        utc_now = datetime.now(timezone.utc)

        def reached(value: Any) -> bool:
            try:
                return utc_now >= datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
            except (TypeError, ValueError):
                return True

        if reached(active.get("expires_at")):
            self._emit("alarm_duration_expired", {"alarm_id": active.get("id")})
        elif active.get("phase") == "flashing" and reached(active.get("flash_until")):
            self._emit("alarm_flash_finished", {"alarm_id": active.get("id")})

    # Compatibility no-ops: active lifecycle is persisted in RoomState now.
    def notify_alarm_started(self) -> None:
        pass

    def notify_alarm_stopped(self) -> None:
        pass
