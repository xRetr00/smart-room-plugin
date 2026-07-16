"""Tests for the automation engine — rule evaluation."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugins.smart_room.runtime.models import RoomState, Modes, DailyFlags
from plugins.smart_room.runtime.automation_engine import evaluate_automations, Action


class TestAutomationEngine:
    """Test automation rule evaluation."""

    def setup_method(self):
        self.state = RoomState()
        self.config = {
            "automations": {
                "adaptive_light": {"enabled": True},
                "alarm": {"enabled": True, "flash_interval_ms": 500, "duration_minutes": 30},
                "evening_sleep": {"enabled": True, "time": "18:00"},
                "work_return": {"enabled": True, "settle_delay": 300},
                "daily_reset": {"enabled": True, "time": "00:00"},
            }
        }

    def test_presence_detected_triggers_light_on(self):
        actions = evaluate_automations(self.state, {"type": "presence_detected"}, self.config)
        assert any(a.type == "turn_on" for a in actions)

    def test_presence_cleared_triggers_light_off(self):
        actions = evaluate_automations(self.state, {"type": "presence_cleared"}, self.config)
        assert any(a.type == "turn_off" for a in actions)

    def test_sleep_mode_triggers_darkness(self):
        actions = evaluate_automations(self.state, {"type": "mode_changed", "mode": "sleep"}, self.config)
        assert any(a.type == "turn_off" for a in actions)

    def test_alarm_mode_triggers_bright_flash(self):
        actions = evaluate_automations(self.state, {"type": "mode_changed", "mode": "alarm"}, self.config)
        light_action = next(a for a in actions if a.type == "set_light")
        assert light_action.params["on"] is True
        assert light_action.params["brightness"] == 100
        assert light_action.params["color_temp"] == 6500
        assert light_action.params["flash"] is True

    def test_alarm_duration_expired_restores_previous_state(self):
        actions = evaluate_automations(self.state, {"type": "alarm_duration_expired"}, self.config)
        assert any(a.type == "ack_alarm" for a in actions)

    def test_alarm_flash_becomes_steady(self):
        actions = evaluate_automations(self.state, {"type": "alarm_flash_finished"}, self.config)
        light = next(a for a in actions if a.type == "set_light")
        assert light.params == {"on": True, "brightness": 100, "color_temp": 6500, "flash": False}

    def test_evening_sleep_triggers_sleep_mode(self):
        self.state.flags.evening_sleep_done_today = False
        actions = evaluate_automations(self.state, {"type": "schedule_evening_sleep"}, self.config)
        assert any(a.type == "set_mode" and a.params["mode"] == "sleep" for a in actions)

    def test_evening_sleep_skipped_if_already_done(self):
        self.state.flags.evening_sleep_done_today = True
        actions = evaluate_automations(self.state, {"type": "schedule_evening_sleep"}, self.config)
        assert not any(a.type == "set_mode" for a in actions)

    def test_work_return_triggers_delayed_sleep(self):
        self.state.flags.work_sleep_done_today = False
        actions = evaluate_automations(self.state, {"type": "geofence_arrive_home"}, self.config)
        sleep_action = next(a for a in actions if a.type == "set_mode")
        assert sleep_action.params["mode"] == "sleep"
        assert sleep_action.params["delay"] == 300

    def test_daily_reset_clears_flags(self):
        actions = evaluate_automations(self.state, {"type": "schedule_daily_reset"}, self.config)
        flag_action = next(a for a in actions if a.type == "set_flag")
        assert flag_action.params["work_sleep_done_today"] is False
        assert flag_action.params["evening_sleep_done_today"] is False

    def test_automations_suppressed_in_sleep_mode(self):
        self.state.modes.active_mode = "sleep"
        actions = evaluate_automations(self.state, {"type": "presence_detected"}, self.config)
        assert not any(a.type == "turn_on" for a in actions)

    def test_automations_suppressed_with_manual_override(self):
        self.state.modes.manual_override = "hold_off"
        actions = evaluate_automations(self.state, {"type": "presence_detected"}, self.config)
        assert not any(a.type == "turn_on" for a in actions)
