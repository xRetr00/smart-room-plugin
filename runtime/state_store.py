"""Atomic JSON state persistence for the smart_room runtime.

State is written to state.json atomically (write tmp + rename).
Read on startup, written on every meaningful state change.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from hermes_constants import get_hermes_home

from plugins.smart_room.runtime.models import RoomState

logger = logging.getLogger(__name__)


def state_path() -> Path:
    return Path(get_hermes_home()) / "smart_room" / "state.json"


def load_state() -> RoomState:
    """Load state from disk, or return fresh default."""
    p = state_path()
    if not p.is_file():
        return RoomState()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return RoomState.from_dict(data)
    except Exception as e:
        logger.warning("Failed to load state from %s: %s — using fresh state", p, e)
        return RoomState()


def save_state(state: RoomState) -> None:
    """Atomically write state to disk."""
    p = state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    data = state.to_dict()
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
    logger.debug("State saved (event_id=%d)", state.event_id)


def load_config() -> Dict[str, Any]:
    """Load smart_room config from config.yaml smart_room section.

    Falls back to defaults if not configured.
    """
    try:
        import yaml
        from hermes_cli.config import cfg_get
        return cfg_get("smart_room", {}) or {}
    except Exception:
        return {}
