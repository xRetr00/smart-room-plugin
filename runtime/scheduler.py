"""Scheduler — time-based automation triggers.

Handles:
- Daily alarm at configurable time (default disabled)
- Evening sleep at 6 PM
- Daily mode flag reset at midnight
- Alarm auto-clear after duration
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class Scheduler:
    """Simple time-based scheduler using a background thread.

    No external dependency — uses threading.Timer and time comparisons.
    For production, APScheduler could be used, but this keeps deps minimal.
    """

    def __init__(self, config: Dict[str, Any], emit_event: Callable[[str, Dict[str, Any]], None]):
        self._config = config
        self._emit = emit_event
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._alarm_active_since: Optional[float] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="smart_room_scheduler")
        self._thread.start()
        logger.info("Scheduler started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self) -> None:
        """Check every 30 seconds for time-based triggers."""
        while not self._stop.is_set():
            try:
                self._check_triggers()
            except Exception as e:
                logger.error("Scheduler check failed: %s", e)
            self._stop.wait(30)  # Check every 30s

    def _check_triggers(self) -> None:
        now = datetime.now()
        auto = self._config.get("automations", {})

        # Daily alarm (default disabled — alarm_time must be explicitly set)
        alarm_time = auto.get("alarm", {}).get("daily_time")
        alarm_enabled = auto.get("alarm", {}).get("enabled", False)
        if alarm_enabled and alarm_time:
            h, m = alarm_time.split(":")
            if now.hour == int(h) and now.minute == int(m):
                self._emit("schedule_alarm", {"time": alarm_time})

        # Evening sleep
        evening_time = auto.get("evening_sleep", {}).get("time", "18:00")
        evening_enabled = auto.get("evening_sleep", {}).get("enabled", True)
        if evening_enabled:
            h, m = evening_time.split(":")
            if now.hour == int(h) and now.minute == int(m):
                self._emit("schedule_evening_sleep", {"time": evening_time})

        # Daily reset at midnight
        reset_time = auto.get("daily_reset", "00:00")
        h, m = reset_time.split(":")
        if now.hour == int(h) and now.minute == int(m):
            self._emit("schedule_daily_reset", {})

        # Alarm auto-clear
        alarm_duration = auto.get("alarm", {}).get("duration_minutes", 30)
        if self._alarm_active_since is not None:
            if time.time() - self._alarm_active_since >= alarm_duration * 60:
                self._emit("alarm_duration_expired", {})
                self._alarm_active_since = None

    def notify_alarm_started(self) -> None:
        """Called by the engine when alarm mode is activated."""
        self._alarm_active_since = time.time()

    def notify_alarm_stopped(self) -> None:
        self._alarm_active_since = None
