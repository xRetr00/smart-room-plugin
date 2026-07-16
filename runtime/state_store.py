"""Atomic JSON state persistence for the smart_room runtime.

State is written to state.json atomically (write tmp + rename).
Read on startup, written on every meaningful state change.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
from pathlib import Path
from typing import Any, Dict

from hermes_constants import get_hermes_home

from plugins.smart_room.runtime.models import RoomState

logger = logging.getLogger(__name__)
_events_lock = threading.Lock()


def state_path() -> Path:
    return Path(get_hermes_home()) / "smart_room" / "state.json"


def load_state() -> RoomState:
    """Load state, recovering the previous atomic snapshot when necessary."""
    p = state_path()
    if not p.is_file():
        return RoomState()
    backup = p.with_suffix(".json.bak")
    for candidate in (p, backup):
        try:
            state = RoomState.from_dict(json.loads(candidate.read_text(encoding="utf-8")))
            if candidate == backup:
                shutil.copy2(backup, p)
                logger.error("Recovered corrupt Smart Room state from %s", backup)
            return state
        except Exception as exc:
            logger.warning("Failed to load state from %s: %s", candidate, exc)
    try:
        p.replace(p.with_suffix(".json.corrupt"))
    except OSError:
        pass
    return RoomState()


def save_state(state: RoomState) -> None:
    """Atomically write state to disk."""
    p = state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    backup = p.with_suffix(".json.bak")
    data = state.to_dict()
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    if p.is_file():
        shutil.copy2(p, backup)
    tmp.replace(p)
    logger.debug("State saved (event_id=%d)", state.event_id)


def load_config() -> Dict[str, Any]:
    """Load smart_room config from config.yaml smart_room section.

    Falls back to defaults if not configured.
    """
    try:
        import yaml
        from hermes_cli.config import cfg_get, load_config as load_hermes_config

        return cfg_get(load_hermes_config(), "smart_room", default={}) or {}
    except Exception:
        return {}


def events_path() -> Path:
    return Path(get_hermes_home()) / "smart_room" / "events.jsonl"


def append_transition(event: Dict[str, Any]) -> None:
    """Append one meaningful transition and mirror it to the Mind activity feed."""
    path = events_path()
    with _events_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) > 500:
            tmp = path.with_suffix(".jsonl.tmp")
            tmp.write_text("\n".join(lines[-500:]) + "\n", encoding="utf-8")
            tmp.replace(path)
    try:
        from cron.scheduler import record_subconscious_activity

        record_subconscious_activity(
            source="world",
            outcome="diff_silent",
            summary=str(event.get("summary") or event.get("type") or "Room changed"),
            diff=json.dumps(event, ensure_ascii=False),
        )
    except Exception:
        logger.debug("Failed to append smart-room activity", exc_info=True)


def publish_welcome(message: str) -> None:
    """Send one room greeting through Marvi's existing proactive delivery lane."""
    try:
        from cron.scheduler import record_subconscious_activity

        record_subconscious_activity(
            source="world",
            outcome="message",
            summary="Smart Room welcome",
            thought=message,
        )
    except Exception:
        logger.debug("Failed to publish smart-room welcome", exc_info=True)


def publish_alarm(alarm_id: str, message: str, *, active: bool) -> None:
    """Surface alarm speech/session lifecycle through Desktop's proactive lane."""
    try:
        from cron.scheduler import record_subconscious_activity

        record_subconscious_activity(
            source="smart_room_alarm",
            job_id=alarm_id,
            outcome="message" if active else "diff_silent",
            summary="Smart Room alarm" if active else "Alarm acknowledged",
            thought=message if active else None,
        )
    except Exception:
        logger.debug("Failed to publish smart-room alarm", exc_info=True)


def load_transition_events(after_id: int = 0) -> list[Dict[str, Any]]:
    path = events_path()
    if not path.exists():
        return []
    events: list[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and int(event.get("id", 0)) > after_id:
            events.append(event)
    return events
