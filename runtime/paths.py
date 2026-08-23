"""Marvi-owned filesystem paths for the independent Smart Room sidecar."""

from __future__ import annotations

import os
from pathlib import Path


def marvi_home() -> Path:
    configured = os.environ.get("MARVI_HOME", "").strip()
    if configured:
        return Path(configured)
    local = os.environ.get("LOCALAPPDATA", "").strip()
    return (Path(local) if local else Path.home() / "AppData" / "Local") / "Marvi-OS"


def smart_room_home() -> Path:
    configured = os.environ.get("MARVI_SMART_ROOM_HOME", "").strip()
    if configured:
        return Path(configured)
    plugin_data = os.environ.get("MARVI_PLUGIN_DATA", "").strip()
    base = Path(plugin_data) if plugin_data else marvi_home() / "plugin-data"
    return base / "smart_room"


def config_path() -> Path:
    configured = os.environ.get("MARVI_SMART_ROOM_CONFIG", "").strip()
    return Path(configured) if configured else smart_room_home() / "config.yaml"


def secrets_path() -> Path:
    configured = os.environ.get("MARVI_SMART_ROOM_SECRETS", "").strip()
    return Path(configured) if configured else smart_room_home() / "secrets.env"


def rpc_token_path() -> Path:
    return smart_room_home() / ".rpc-token"


def vision_home() -> Path:
    return smart_room_home() / "vision"


def vision_models_home() -> Path:
    return vision_home() / "models"
