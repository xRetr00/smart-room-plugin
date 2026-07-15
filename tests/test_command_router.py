"""Tests for the command router — RPC method dispatch."""

import sys
import os
import json
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugins.smart_room.runtime.models import RoomState, LightState, Modes, now_iso
from plugins.smart_room.runtime.command_router import CommandRouter
from plugins.smart_room.runtime.state_store import save_state, load_state


class TestCommandRouter:
    """Test RPC method routing."""

    def setup_method(self):
        self.state = RoomState()
        self.config = {"scenes": {"reading": {"brightness": 70, "color_temp": 3000}}}
        self.runtime = MagicMock()
        self.runtime.set_mode = MagicMock()
        self.runtime.set_light = MagicMock()
        self.runtime.cancel_sleep = MagicMock()
        self.runtime.test_welcome = MagicMock()
        self.runtime.get_status = MagicMock(
            return_value={
                "running": True,
                "sound_events": {"enabled": True, "running": True},
            }
        )
        self.router = CommandRouter(self.state, self.config, self.runtime)

    def test_get_state_returns_snapshot(self):
        result = self.router.dispatch("get_state", {})
        assert result["success"] is True
        assert "state" in result
        assert result["state"]["last_updated"] is None
        assert self.state.event_id == 0

    def test_set_mode_calls_runtime(self):
        result = self.router.dispatch("set_mode", {"mode": "reading"})
        assert result["success"] is True
        self.runtime.set_mode.assert_called_once_with("reading")

    def test_set_light_calls_runtime(self):
        result = self.router.dispatch("set_light", {"on": True, "brightness": 50})
        assert result["success"] is True
        self.runtime.set_light.assert_called_once_with(on=True, brightness=50, manual=True)

    def test_set_override_updates_state(self):
        result = self.router.dispatch("set_override", {"enabled": True})
        assert result["success"] is True
        self.runtime.set_override.assert_called_once_with(True)

    def test_cancel_sleep_calls_runtime(self):
        result = self.router.dispatch("cancel_sleep", {})
        assert result["success"] is True
        self.runtime.cancel_sleep.assert_called_once()

    def test_welcome_preview_calls_runtime_for_valid_audience(self):
        result = self.router.dispatch("test_welcome", {"audience": "guest"})
        assert result == {"success": True, "audience": "guest"}
        self.runtime.test_welcome.assert_called_once_with("guest")

    def test_get_health_returns_report(self):
        result = self.router.dispatch("get_health", {})
        assert result["success"] is True
        assert "health" in result
        assert result["health"]["sound_events"]["running"] is True

    def test_get_diagnostic_returns_full_dump(self):
        result = self.router.dispatch("get_diagnostic", {})
        assert result["success"] is True
        assert "diagnostic" in result
        assert "state" in result["diagnostic"]

    def test_unknown_method_returns_error(self):
        result = self.router.dispatch("nonexistent", {})
        assert result["success"] is False
        assert "unknown method" in result["error"]
