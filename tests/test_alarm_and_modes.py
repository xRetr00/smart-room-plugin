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
    published = []
    monkeypatch.setattr(app_module, "publish_alarm", lambda alarm_id, message, active: published.append((alarm_id, message, active)))
    runtime = Runtime({})
    runtime._tuya = FakeTuya()

    alarm = runtime.upsert_alarm({
        "name": "Work", "time": "08:00", "recurrence": "once", "date": "2026-07-16",
    })
    assert alarm["id"] and alarm["recurrence"] == "once"

    runtime.set_mode("night")
    assert runtime._state.modes.active_mode == "night"
    runtime.set_mode("alarm", alarm_id=alarm["id"], alarm_name=alarm["name"], duration_minutes=10)
    assert runtime._state.active_alarm.name == "Work"
    assert published[-1][2] is True

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


def test_off_during_alarm_acknowledges_then_stays_off(monkeypatch):
    monkeypatch.setattr(app_module, "publish_alarm", lambda *_args, **_kwargs: None)
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
