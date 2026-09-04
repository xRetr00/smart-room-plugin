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
        self.runtime.run_diagnostic = MagicMock(return_value={"state": self.state.to_dict()})
        self.runtime.list_alarms = MagicMock(return_value=[])
        self.runtime.get_active_alarm = MagicMock(return_value=None)
        self.runtime.upsert_alarm = MagicMock(return_value={"id": "wake", "name": "Work"})
        self.runtime.delete_alarm = MagicMock(return_value=True)
        self.runtime.acknowledge_alarm = MagicMock(return_value={"success": True, "active": False})
        self.runtime.vision_observe = MagicMock(return_value={"success": True, "vision": {}})
        self.runtime.vision_describe = MagicMock(return_value={"success": True, "description": "owner visible"})
        self.runtime.vision_people = MagicMock(return_value={"success": True, "people": []})
        self.runtime.vision_visitors = MagicMock(return_value={"success": True, "visitors": []})
        self.runtime.vision_enroll_owner = MagicMock(return_value={"success": True, "name": "Retro"})
        self.runtime.vision_approve = MagicMock(return_value={"success": True, "name": "Guest"})
        self.runtime.vision_reject = MagicMock(return_value={"success": True, "sighting_id": 7})
        self.runtime.vision_reject_all = MagicMock(return_value={"success": True, "rejected": 4})
        self.runtime.vision_set_owner = MagicMock(return_value={"success": True, "name": "Retro"})
        self.runtime.get_status = MagicMock(
            return_value={
                "running": True,
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
        self.runtime.set_mode.assert_called_once_with("reading", reason="manual")

    def test_set_light_calls_runtime(self):
        result = self.router.dispatch("set_light", {"on": True, "brightness": 50})
        assert result["success"] is True
        self.runtime.set_light.assert_called_once_with(on=True, brightness=50, manual=True)

    def test_set_override_updates_state(self):
        result = self.router.dispatch("set_override", {"mode": "hold_on"})
        assert result["success"] is True
        self.runtime.set_override.assert_called_once_with("hold_on")

    def test_cancel_sleep_calls_runtime(self):
        result = self.router.dispatch("cancel_sleep", {})
        assert result["success"] is True
        self.runtime.cancel_sleep.assert_called_once()

    def test_welcome_preview_calls_runtime_for_valid_audience(self):
        result = self.router.dispatch("test_welcome", {"audience": "guest"})
        assert result["success"] is True
        assert result["audience"] == "guest"
        assert result["schema_version"] == 1
        self.runtime.test_welcome.assert_called_once_with("guest")

    def test_get_health_returns_report(self):
        result = self.router.dispatch("get_health", {})
        assert result["success"] is True
        assert "health" in result
        assert "mqtt" in result["health"]

    def test_get_diagnostic_returns_full_dump(self):
        result = self.router.dispatch("get_diagnostic", {})
        assert result["success"] is True
        assert "diagnostic" in result
        assert "state" in result["diagnostic"]

    def test_unknown_method_returns_error(self):
        result = self.router.dispatch("nonexistent", {})
        assert result["success"] is False
        assert "unknown method" in result["error"]

    def test_alarm_management_routes_to_runtime(self):
        created = self.router.dispatch("upsert_alarm", {"name": "Work", "time": "08:00", "recurrence": "daily"}, request_id="cmd-1")
        assert created["alarm"]["id"] == "wake"
        assert created["request_id"] == "cmd-1"
        acknowledged = self.router.dispatch("acknowledge_alarm", {})
        assert acknowledged["active"] is False

    def test_vision_reads_route_to_the_sidecar_worker(self):
        assert self.router.dispatch("vision_observe", {})["success"] is True
        assert self.router.dispatch("vision_describe", {})["description"] == "owner visible"
        assert self.router.dispatch("vision_people", {})["people"] == []
        assert self.router.dispatch("vision_visitors", {})["visitors"] == []

    def test_vision_identity_changes_route_to_the_sidecar_worker(self):
        enrolled = self.router.dispatch("vision_enroll_owner", {"name": "Retro", "seconds": 3})
        approved = self.router.dispatch(
            "vision_approve", {"sighting_id": 7, "name": "Guest", "owner": False}
        )
        rejected = self.router.dispatch("vision_reject", {"sighting_id": 7})
        rejected_all = self.router.dispatch("vision_reject_all", {})
        owner = self.router.dispatch("vision_set_owner", {"name": "Retro"})

        assert enrolled["name"] == "Retro"
        self.runtime.vision_enroll_owner.assert_called_once_with("Retro", 3.0)
        assert approved["name"] == "Guest"
        assert rejected["sighting_id"] == 7
        assert rejected_all["rejected"] == 4
        assert owner["name"] == "Retro"
