"""Atomic JSON state persistence for the smart_room runtime.

State is written to state.json atomically (write tmp + rename).
Read on startup, written on every meaningful state change.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any, Dict

from .models import RoomState
from .paths import config_path, smart_room_home

logger = logging.getLogger(__name__)
_events_lock = threading.Lock()
_locations_lock = threading.Lock()
_last_location_keys: Dict[str, str] = {}


def state_path() -> Path:
    return smart_room_home() / "state.json"


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


#: How hard to try before giving up on one state write. Short: the caller is
#: usually an MQTT callback, and the next tick writes the same state anyway.
SAVE_ATTEMPTS = 5
SAVE_RETRY_SECONDS = 0.05


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

    # Retried, because on Windows `os.replace` fails outright when anything
    # else holds the target open -- a reader, an indexer, antivirus -- and the
    # lock is gone milliseconds later.
    #
    # It is not a theoretical race. The unhandled PermissionError came back up
    # through the ESPresense handler and killed the paho client thread, which
    # takes *every* subscription down with it: OwnTracks stopped arriving for
    # a fortnight because a state file was busy for one moment.
    last: Optional[OSError] = None
    for attempt in range(SAVE_ATTEMPTS):
        try:
            tmp.replace(p)
            logger.debug("State saved (event_id=%d)", state.event_id)
            return
        except PermissionError as error:
            last = error
            time.sleep(SAVE_RETRY_SECONDS * (attempt + 1))
    # Still busy. The next tick writes the same state again, so losing this one
    # costs nothing -- and raising here would kill whichever thread called us.
    logger.warning("Could not replace state file after %d attempts: %s", SAVE_ATTEMPTS, last)


def load_config() -> Dict[str, Any]:
    """Load the sidecar's own config, defaulting to enabled when installed."""
    try:
        import yaml

        raw = yaml.safe_load(config_path().read_text(encoding="utf-8")) or {}
        config = raw.get("smart_room", raw) if isinstance(raw, dict) else {}
        return {"enabled": True, **config}
    except (OSError, ValueError, TypeError):
        return {"enabled": True}


def save_config(config: Dict[str, Any]) -> None:
    """Persist the behavior config Marvi supplied to the independent sidecar."""
    import yaml

    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def events_path() -> Path:
    return smart_room_home() / "events.jsonl"


def locations_path() -> Path:
    return smart_room_home() / "locations.jsonl"


def append_location_report(topic: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Preserve one OwnTracks report as append-only local JSONL."""
    received_at = datetime.now(timezone.utc).isoformat()
    try:
        reported_at = datetime.fromtimestamp(float(payload.get("tst")), timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        reported_at = received_at
    regions = payload.get("inregions")
    zone = str(payload.get("desc") or (regions[0] if isinstance(regions, list) and regions else "")).strip().lower()
    record = {
        "received_at": received_at,
        "reported_at": reported_at,
        "topic": str(topic),
        "type": str(payload.get("_type") or "unknown"),
        "event": str(payload.get("event") or ""),
        "zone": zone,
        "latitude": payload.get("lat"),
        "longitude": payload.get("lon"),
        "accuracy_m": payload.get("acc"),
        "altitude_m": payload.get("alt"),
        "velocity_kmh": payload.get("vel"),
        "course": payload.get("cog"),
        "battery_percent": payload.get("batt"),
        "trigger": payload.get("t"),
        "connection": payload.get("conn"),
        "data": payload,
    }
    path = locations_path()
    key = json.dumps([str(topic), reported_at, payload], ensure_ascii=False, sort_keys=True)
    with _locations_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        path_key = str(path)
        if path_key not in _last_location_keys and path.exists():
            last_line = ""
            with path.open("r", encoding="utf-8") as handle:
                for last_line in handle:
                    pass
            try:
                previous = json.loads(last_line)
                _last_location_keys[path_key] = json.dumps(
                    [previous.get("topic", ""), previous.get("reported_at", ""), previous.get("data", {})],
                    ensure_ascii=False,
                    sort_keys=True,
                )
            except (json.JSONDecodeError, AttributeError):
                pass
        if _last_location_keys.get(path_key) == key:
            return {**record, "duplicate": True}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        _last_location_keys[path_key] = key
    return record


def load_location_reports(
    *, limit: int = 20, since: str = "", until: str = "", zone: str = ""
) -> list[Dict[str, Any]]:
    """Read the latest matching OwnTracks reports without loading the whole log."""
    path = locations_path()
    if not path.exists():
        return []
    # ponytail: JSONL scan is O(n); move to SQLite only when this log becomes measurably slow.
    matches: deque[Dict[str, Any]] = deque(maxlen=max(1, min(int(limit), 500)))
    zone = zone.strip().lower()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            stamp = str(record.get("reported_at") or record.get("received_at") or "")
            if since and stamp < since:
                continue
            if until and stamp > until:
                continue
            if zone and str(record.get("zone") or "").lower() != zone:
                continue
            matches.append(record)
    return list(matches)


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
