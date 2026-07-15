"""Keep smart-room tests out of the user's real Marvi profile."""

from __future__ import annotations

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture(autouse=True)
def isolated_hermes_home(tmp_path):
    token = set_hermes_home_override(tmp_path / "hermes-home")
    try:
        yield
    finally:
        reset_hermes_home_override(token)
