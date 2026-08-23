from __future__ import annotations

import threading
import time

from plugins.smart_room.runtime import app as app_module
from plugins.smart_room.runtime.app import Runtime
from plugins.smart_room.runtime.state_store import load_state, save_state, state_path


class FakeTuya:
    def __init__(self):
        self.commands = []
        self.flash_stopped = False

    def set_light(self, **kwargs):
        self.commands.append(kwargs)
        return {"success": True}

    def stop_flash(self):
        self.flash_stopped = True

    def health(self):
        return {}


def test_named_alarm_lifecycle_and_night_mode(monkeypatch):
    runtime = Runtime({})
    runtime._tuya = FakeTuya()
    emitted = []
    monkeypatch.setattr(runtime, "_emit_event", lambda event, data: emitted.append((event, data)))

    alarm = runtime.upsert_alarm({
        "name": "Work", "time": "08:00", "recurrence": "once", "date": "2026-07-16",
    })
    assert alarm["id"] and alarm["recurrence"] == "once"

    runtime.set_mode("night")
    assert runtime._state.modes.active_mode == "night"
    runtime.set_mode("alarm", alarm_id=alarm["id"], alarm_name=alarm["name"], duration_minutes=10)
    assert runtime._state.active_alarm.name == "Work"
    alarm_event = next(data for event, data in emitted if event == "alarm_requested")
    assert alarm_event["active"] is True

    result = runtime.acknowledge_alarm()
    assert result["success"] is True
    assert runtime._state.active_alarm is None
    assert runtime._state.modes.active_mode == "night"
    assert runtime._tuya.flash_stopped is True


def test_cancel_sleep_does_not_restore_light_in_empty_room():
    runtime = Runtime({})
    runtime._tuya = FakeTuya()
    runtime._state.modes.active_mode = "sleep"
    runtime._state.sleep_restore = {"mode": "reading", "light": {"on": True, "brightness": 70}}

    runtime.cancel_sleep()

    assert runtime._tuya.commands[-1]["on"] is False
    assert runtime._state.light.on is False


def test_runtime_start_discards_persisted_sleep_without_touching_light():
    state = load_state()
    state.modes.active_mode = "sleep"
    state.light.on = True
    state.sleep_restore = {"mode": "reading", "light": {"on": True}}
    save_state(state)

    runtime = Runtime({})

    assert runtime._state.modes.active_mode == "reading"
    assert runtime._state.light.on is True
    assert runtime._state.sleep_restore == {}


def test_manual_light_control_leaves_sleep_mode():
    runtime = Runtime({})
    runtime._tuya = FakeTuya()
    runtime._state.modes.active_mode = "sleep"
    runtime._state.sleep_restore = {"mode": "reading", "light": {"on": False}}

    runtime.set_light(on=True, manual=True)

    assert runtime._state.modes.active_mode == "reading"
    assert runtime._state.light.on is True


def test_setting_a_non_sleep_mode_clears_stale_restore_state():
    runtime = Runtime({})
    runtime._tuya = FakeTuya()
    runtime._state.sleep_restore = {"mode": "off", "light": {"on": False}}

    runtime.set_mode("reading")

    assert runtime._state.sleep_restore == {}


def test_normal_mode_is_neutral_white_at_seventy_percent():
    runtime = Runtime({})
    runtime._tuya = FakeTuya()

    runtime.set_mode("normal")

    assert runtime._state.modes.active_mode == "normal"
    assert runtime._tuya.commands[-1]["brightness"] == 70
    assert runtime._tuya.commands[-1]["color_temp"] == 4000


def test_off_during_alarm_acknowledges_then_stays_off(monkeypatch):
    runtime = Runtime({})
    runtime._tuya = FakeTuya()
    runtime.set_mode("reading")
    runtime.set_mode("alarm", alarm_id="wake", alarm_name="Wake", duration_minutes=10)

    runtime.set_mode("off")

    assert runtime._state.active_alarm is None
    assert runtime._state.modes.active_mode == "off"
    assert runtime._state.light.on is False


def test_corrupt_state_recovers_backup():
    runtime = Runtime({})
    runtime._state.modes.active_mode = "reading"
    save_state(runtime._state)
    runtime._state.modes.active_mode = "focus"
    save_state(runtime._state)
    state_path().write_text("not json", encoding="utf-8")

    assert load_state().modes.active_mode == "reading"


def test_slow_tuya_command_does_not_hold_presence_state_lock():
    entered = threading.Event()
    release = threading.Event()

    class SlowTuya(FakeTuya):
        def set_light(self, **kwargs):
            entered.set()
            release.wait(2)
            return {"success": True}

    runtime = Runtime({})
    runtime._tuya = SlowTuya()
    command = threading.Thread(target=runtime.set_light, kwargs={"on": True})
    command.start()
    assert entered.wait(1)

    started = time.monotonic()
    runtime._on_esp32_status(True, "192.0.2.10")
    elapsed = time.monotonic() - started
    release.set()
    command.join(2)

    assert elapsed < 0.2
