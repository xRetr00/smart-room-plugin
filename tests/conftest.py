"""Load the standalone plugin package and isolate its Marvi-owned data."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest


_ROOT = Path(__file__).resolve().parents[1]
if "plugins" not in sys.modules:
    namespace = types.ModuleType("plugins")
    namespace.__path__ = []
    sys.modules["plugins"] = namespace
if "plugins.smart_room" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "plugins.smart_room",
        _ROOT / "__init__.py",
        submodule_search_locations=[str(_ROOT)],
    )
    assert spec and spec.loader
    package = importlib.util.module_from_spec(spec)
    sys.modules["plugins.smart_room"] = package
    spec.loader.exec_module(package)
    # Pytest imports the repository root as ``__init__`` because the checkout
    # folder contains a hyphen. Point that collection name at the same package.
    sys.modules.setdefault("__init__", package)


@pytest.fixture(autouse=True)
def isolated_marvi_smart_room_home(tmp_path, monkeypatch):
    monkeypatch.setenv("MARVI_SMART_ROOM_HOME", str(tmp_path / "marvi-smart-room"))
