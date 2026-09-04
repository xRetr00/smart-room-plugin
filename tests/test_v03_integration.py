"""v0.3 acceptance contracts that cross the plugin's component seams."""

from __future__ import annotations

import ast
import json
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.smart_room import process_manager
from plugins.smart_room.bridge import call_runtime
from plugins.smart_room.bridge import build_context_line
from plugins.smart_room.runtime.app import Runtime
from plugins.smart_room.runtime.models import MmWaveState, PhoneLocation, Presence, RoomState
from plugins.smart_room.runtime.presence_fusion import fuse
from plugins.smart_room.runtime.scheduler import Scheduler
from plugins.smart_room.runtime.state_store import append_location_report, append_transition, load_location_reports
from plugins.smart_room.runtime.state_store import save_state
from plugins.smart_room.tools import handle_smart_room_state


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for(predicate, timeout: float = 12.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError("condition did not become true before timeout")


def _rpc_ready() -> bool:
    try:
        return bool(call_runtime("get_state", {}).get("success"))
    except RuntimeError:
        return False


def test_geofence_home_identity_survives_until_mmwave_appears():
    presence, mmwave, location, _, _ = fuse(
        Presence(), MmWaveState(), PhoneLocation(),
        ble_detected=False, ble_rssi=None, mmwave_occupied=False,
        geofence_zone="home", exit_timeout_elapsed=False,
    )
    presence, _, _, light_on, _ = fuse(
        presence, mmwave, location,
        ble_detected=False, ble_rssi=None, mmwave_occupied=True,
        geofence_zone=None, exit_timeout_elapsed=False,
    )
    assert presence.detected is True
    assert presence.source == "geofence_mmwave"
    assert light_on is True


def test_wifi_is_positive_only_identity_evidence():
    presence, _, _, _, _ = fuse(
        Presence(), MmWaveState(), PhoneLocation(),
        ble_detected=False, ble_rssi=None, mmwave_occupied=True,
        geofence_zone=None, exit_timeout_elapsed=False, wifi_detected=True,
    )
    assert presence.detected is True
    assert presence.source == "wifi_mmwave"

    presence, _, _, _, _ = fuse(
        presence, MmWaveState(occupied=True), PhoneLocation(),
        ble_detected=False, ble_rssi=None, mmwave_occupied=True,
        geofence_zone=None, exit_timeout_elapsed=False, wifi_detected=False,
    )
    assert presence.detected is True
    assert presence.source == "wifi_mmwave"
    assert presence.identity_sticky is False


def test_exit_timeout_uses_last_mmwave_occupancy_not_phone_sighting():
    runtime = Runtime({"esp32": {"exit_timeout": 60}})
    runtime._state.presence.last_seen = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()
    runtime._state.mmwave.occupied = False
    runtime._state.mmwave.last_seen = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()

    assert runtime._check_exit_timeout() is False

    runtime._state.mmwave.last_seen = (datetime.now(timezone.utc) - timedelta(seconds=61)).isoformat()
    assert runtime._check_exit_timeout() is True


def test_he20_clear_event_is_delayed_and_edge_triggered(monkeypatch):
    runtime = Runtime({"esp32": {"exit_timeout": 60}})
    runtime._state.mmwave.occupied = True
    runtime._state.mmwave.last_seen = datetime.now(timezone.utc).isoformat()
    runtime._room_clear_emitted = False
    runtime._tuya = MagicMock()
    runtime._tuya.get_light_status.return_value = {"success": True, "on": True, "brightness": 70}
    runtime._tuya.get_mmwave_status.return_value = {"success": True, "occupied": False}
    runtime._mqtt = None
    monkeypatch.setattr(runtime, "_probe_wifi_presence", lambda: False)
    emitted = []
    monkeypatch.setattr(runtime, "_emit_event", lambda event, data: emitted.append((event, data)))

    runtime._poll_devices()
    assert emitted == []

    runtime._state.mmwave.last_seen = (datetime.now(timezone.utc) - timedelta(seconds=61)).isoformat()
    runtime._poll_devices()
    runtime._poll_devices()
    assert emitted == [("presence_cleared", {"source": "mmwave"})]


def test_he20_single_occupied_pulse_does_not_create_room_entry(monkeypatch):
    runtime = Runtime({"esp32": {"exit_timeout": 60}})
    runtime._state.mmwave.occupied = False
    runtime._room_clear_emitted = True
    runtime._tuya = MagicMock()
    runtime._tuya.get_light_status.return_value = {"success": True, "on": True, "brightness": 70}
    runtime._tuya.get_mmwave_status.side_effect = [
        {"success": True, "occupied": True},
        {"success": True, "occupied": False},
    ]
    runtime._tuya.health.return_value = {}
    runtime._mqtt = None
    transition = MagicMock()
    monkeypatch.setattr(runtime, "_probe_wifi_presence", lambda: False)
    monkeypatch.setattr(runtime, "_handle_welcome_transition", transition)

    runtime._poll_devices()
    runtime._poll_devices()

    transition.assert_not_called()
    runtime._tuya.get_mmwave_status.side_effect = None
    runtime._tuya.get_mmwave_status.return_value = {"success": True, "occupied": True}
    runtime._poll_devices()
    runtime._mmwave_occupied_since -= 4
    runtime._poll_devices()
    transition.assert_called_once_with(False, True)


def test_entry_is_tracked_but_welcome_requires_one_hour_empty(monkeypatch):
    runtime = Runtime({
        "owner": "Shereef",
        "welcome": {"enabled": True, "reset_after_seconds": 3600, "identity_grace_seconds": 0},
    })
    runtime._state.room_empty_since = (datetime.now(timezone.utc) - timedelta(minutes=59)).isoformat()
    timer = MagicMock()
    monkeypatch.setattr(threading, "Timer", lambda *_args, **_kwargs: timer)
    runtime._handle_welcome_transition(False, True)
    timer.start.assert_called_once()
    assert runtime._pending_entry_should_welcome is False

    timer.reset_mock()
    runtime._state.room_empty_since = (datetime.now(timezone.utc) - timedelta(hours=1, seconds=1)).isoformat()
    runtime._handle_welcome_transition(False, True)
    timer.start.assert_called_once()
    assert runtime._pending_entry_should_welcome is True


def test_welcome_preview_emits_structured_event_without_recording_arrival(monkeypatch):
    runtime = Runtime({"owner": "Shereef"})
    runtime._state.last_welcome_at = None
    emitted = []
    monkeypatch.setattr(runtime, "_emit_event", lambda event, data: emitted.append((event, data)))

    runtime._publish_welcome(True, "Shereef", record_arrival=False)

    assert emitted == [("room_welcome", {
        "audience": "owner",
        "owner_name": "Shereef",
        "visitor_entries": [],
        "message": "Welcome back, Shereef.",
        "summary": "Smart Room welcome",
    })]
    assert runtime._state.last_welcome_at is None


def test_guest_welcome_is_structured_for_host_delivery(monkeypatch):
    runtime = Runtime({"owner": "Shereef"})
    emitted = []
    monkeypatch.setattr(runtime, "_emit_event", lambda event, data: emitted.append((event, data)))

    runtime._publish_welcome(False, "Shereef", record_arrival=False)

    assert emitted[0][0] == "room_welcome"
    assert emitted[0][1]["audience"] == "guest"
    assert emitted[0][1]["message"] == "Welcome. Shereef isn't here right now."


def test_recent_owntracks_home_arrival_identifies_owner_when_ble_sleeps(monkeypatch):
    runtime = Runtime({"owner": "Shereef", "welcome": {"owner_evidence_window_seconds": 3600}})
    runtime._state.mmwave.occupied = True
    runtime._state.presence.detected = True
    runtime._state.presence.source = "geofence_mmwave"
    runtime._state.location.home = True
    runtime._state.location.last_geofence_at = "2026-07-19T09:57:00+00:00"
    runtime._ble_detected = False
    runtime._pending_entry_at = "2026-07-19T10:00:00+00:00"
    runtime._pending_entry_should_welcome = False
    published = MagicMock()
    emitted = MagicMock()
    monkeypatch.setattr(runtime, "_publish_welcome", published)
    monkeypatch.setattr(runtime, "_emit_event", emitted)

    runtime._deliver_welcome()

    published.assert_not_called()
    assert runtime._state.unreported_visitor_entries == []
    assert emitted.call_args.args[1]["classification"] == "owner"
    assert emitted.call_args.args[1]["identity_reason"] == "owntracks_recent"


def test_stale_home_phone_without_owner_evidence_is_still_a_guest(monkeypatch):
    runtime = Runtime({"owner": "Shereef", "welcome": {"owner_evidence_window_seconds": 3600}})
    runtime._state.mmwave.occupied = True
    runtime._state.location.home = True
    runtime._state.location.last_geofence_at = "2026-07-19T08:00:00+00:00"
    runtime._pending_entry_at = "2026-07-19T10:00:00+00:00"
    runtime._pending_entry_should_welcome = False
    emitted = MagicMock()
    monkeypatch.setattr(runtime, "_emit_event", emitted)

    runtime._deliver_welcome()

    assert runtime._state.unreported_visitor_entries[-1]["classification"] == "guest"
    assert emitted.call_args.args[1]["identity_reason"] == "phone_home_without_recent_owner_evidence"


def test_next_confirmed_owner_welcome_reports_and_clears_visitor_entries(monkeypatch):
    runtime = Runtime({"owner": "Shereef"})
    runtime._state.mmwave.occupied = True
    runtime._state.location.home = True
    runtime._state.unreported_visitor_entries = [{
        "at": "2026-07-19T10:00:00+00:00",
        "classification": "unknown_visitor",
        "owner_phone_home": False,
    }]
    runtime._ble_detected = True
    runtime._pending_entry_should_welcome = True
    published = MagicMock()
    monkeypatch.setattr(runtime, "_publish_welcome", published)
    monkeypatch.setattr(runtime, "_emit_event", MagicMock())

    runtime._deliver_welcome()

    published.assert_called_once_with(
        True,
        "Shereef",
        record_arrival=True,
        visitor_entries=[{
            "at": "2026-07-19T10:00:00+00:00",
            "classification": "unknown_visitor",
            "owner_phone_home": False,
        }],
    )
    assert runtime._state.unreported_visitor_entries == []


def test_confirmed_owner_reports_visitors_even_inside_welcome_cooldown(monkeypatch):
    runtime = Runtime({"owner": "Shereef"})
    runtime._state.mmwave.occupied = True
    runtime._state.location.home = True
    runtime._state.unreported_visitor_entries = [{
        "at": "2026-07-19T10:00:00+00:00",
        "classification": "unknown_visitor",
        "owner_phone_home": False,
    }]
    runtime._ble_detected = True
    runtime._pending_entry_should_welcome = False
    emitted = []
    monkeypatch.setattr(runtime, "_emit_event", lambda event, data: emitted.append((event, data)))
    published = MagicMock()
    monkeypatch.setattr(runtime, "_publish_welcome", published)

    runtime._deliver_welcome()

    published.assert_not_called()
    report = next(data for event, data in emitted if event == "visitor_report")
    assert "entered the room" in report["message"]
    assert runtime._state.unreported_visitor_entries == []


def test_owntracks_history_preserves_details_and_filters():
    location = {
        "_type": "location", "lat": 41.1, "lon": 29.2, "acc": 12,
        "batt": 87, "inregions": ["Home"], "tst": 1784358126,
    }
    append_location_report("owntracks/smart_room/iphone", location)
    duplicate = append_location_report("owntracks/smart_room/iphone", location)
    append_location_report("owntracks/smart_room/iphone", {
        "_type": "transition", "event": "leave", "desc": "Home", "tst": 1784359000,
    })

    reports = load_location_reports(limit=10, zone="home")

    assert len(reports) == 2
    assert duplicate["duplicate"] is True
    assert reports[0]["latitude"] == 41.1
    assert reports[0]["accuracy_m"] == 12
    assert reports[0]["data"]["batt"] == 87
    assert reports[1]["event"] == "leave"


def test_owntracks_heartbeat_uses_phone_report_time_not_mqtt_delivery_time():
    runtime = Runtime({"owner": "Shereef"})
    runtime._state.location.home = True
    runtime._state.location.zone = "home"
    runtime._state.location.source = "owntracks"
    reported = int(time.time()) - 120

    runtime._on_owntracks("owntracks/smart_room/iphone", {
        "_type": "location", "inregions": ["Home"], "tst": reported,
    })
    runtime._on_geofence("sync", "home")

    stamp = datetime.fromisoformat(runtime._state.location.last_geofence_at)
    assert int(stamp.timestamp()) == reported


def test_location_event_is_idempotent_and_schedules_work_return(monkeypatch):
    runtime = Runtime({
        "owner": "shereef",
        "automations": {
            "work_return": {
                "enabled": True,
                "work_hours_start": "00:00",
                "work_hours_end": "23:59",
                "settle_delay": 300,
            },
            "adaptive_light": {"enabled": False},
            "evening_sleep": {"enabled": False},
        },
    })
    runtime._state.mmwave.occupied = True
    scheduled = MagicMock()
    monkeypatch.setattr(runtime, "_schedule_mode", scheduled)
    runtime._on_geofence("sync", "home")
    assert runtime._state.location.zone == "home"
    assert runtime._state.location.source == "owntracks"
    scheduled.assert_not_called()
    at = datetime.now(timezone.utc).isoformat()
    params = {
        "who": "Shereef", "transition": "ARRIVE", "zone": "Home",
        "at": at, "delivery_id": "delivery-1", "source": "OwnTracks",
    }
    first = runtime.phone_location_changed(**params)
    second = runtime.phone_location_changed(**params)
    assert first["success"] is True and first["duplicate"] is False
    assert second["success"] is True and second["duplicate"] is True
    assert runtime._state.location.zone == "home"
    assert runtime._state.location.source == "owntracks"
    assert runtime._state.location.last_geofence_at == at
    scheduled.assert_called_once_with("sleep", 300, reason="work_return")


def test_scheduler_fires_each_daily_trigger_once(monkeypatch):
    import plugins.smart_room.runtime.scheduler as scheduler_module

    fixed = datetime(2026, 7, 15, 18, 0)

    class FakeDateTime:
        @classmethod
        def now(cls):
            return fixed

    monkeypatch.setattr(scheduler_module, "datetime", FakeDateTime)
    emitted = []
    scheduler = Scheduler({
        "automations": {
            "alarm": {"enabled": False},
            "evening_sleep": {"enabled": True, "time": "18:00"},
            "daily_reset": "00:00",
        }
    }, lambda event, data: emitted.append((event, data)))
    scheduler._check_triggers()
    scheduler._check_triggers()
    assert [event for event, _ in emitted] == ["schedule_evening_sleep"]


def test_scheduler_supports_one_day_and_daily_named_alarms(monkeypatch):
    import plugins.smart_room.runtime.scheduler as scheduler_module

    fixed = datetime(2026, 7, 15, 8, 0)

    class FakeDateTime:
        @classmethod
        def now(cls, *_args):
            return fixed

        @classmethod
        def fromisoformat(cls, value):
            return datetime.fromisoformat(value)

    alarms = [
        {"id": "once", "name": "Once", "time": "08:00", "recurrence": "once", "date": "2026-07-15", "enabled": True},
        {"id": "daily", "name": "Daily", "time": "08:00", "recurrence": "daily", "enabled": True},
    ]
    monkeypatch.setattr(scheduler_module, "datetime", FakeDateTime)
    emitted = []
    scheduler = Scheduler(
        {"automations": {"evening_sleep": {"enabled": False}}},
        lambda event, data: emitted.append((event, data)),
        get_alarms=lambda: alarms,
    )

    scheduler._check_triggers()
    scheduler._check_triggers()

    assert [data["alarm"]["id"] for event, data in emitted if event == "schedule_alarm"] == ["once", "daily"]


def test_plugin_registers_tools_lifecycle_and_context(monkeypatch):
    import plugins.smart_room as plugin

    class FakeContext:
        def __init__(self):
            self.tools = []
            self.hooks = []
            self.context = []

        def register_tool(self, **kwargs):
            self.tools.append(kwargs)

        def register_hook(self, name, callback):
            self.hooks.append((name, callback))

        def register_context_provider(self, name, handler):
            self.context.append((name, handler))

    monkeypatch.setattr(plugin.platform, "system", lambda: "Windows")
    ctx = FakeContext()
    plugin.register(ctx)
    assert len(ctx.tools) == 10
    assert {name for name, _ in ctx.hooks} == {
        "on_gateway_start", "on_gateway_stop",
    }
    assert [name for name, _ in ctx.context] == ["smart_room"]


def test_context_line_is_config_gated_and_uses_fused_source():
    import yaml
    from plugins.smart_room.runtime.paths import config_path

    config_path().parent.mkdir(parents=True, exist_ok=True)
    config_path().write_text(yaml.safe_dump({
        "enabled": True, "context": {"enabled": True}, "owner": "shereef"
    }), encoding="utf-8")
    state = RoomState()
    state.presence = Presence(detected=True, source="ble_sticky_mmwave", confidence=0.6)
    state.light.on = True
    state.light.brightness = 70
    state.light.scene = "reading"
    state.modes.active_mode = "reading"
    save_state(state)
    line = build_context_line()
    assert line is not None
    assert "Shereef present (ble_sticky_mmwave, conf 0.60)" in line
    assert "light 70% (reading) @3000K" in line

    config_path().write_text(yaml.safe_dump({
        "enabled": True, "context": {"enabled": False}
    }), encoding="utf-8")
    assert build_context_line() is None


def test_runtime_has_no_memory_writer_imports():
    runtime_root = Path(__file__).resolve().parents[1] / "runtime"
    imports = set()
    for path in runtime_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
    assert not any("memory_tool" in name or "honcho" in name for name in imports)


def test_tuya_v35_bulb_uses_standard_hsv_dps(monkeypatch):
    from types import SimpleNamespace
    from plugins.smart_room.runtime.tuya import controller

    class Device:
        values = None

        def __init__(self, **_kwargs):
            pass

        def set_version(self, _version):
            pass

        def set_multiple_values(self, values):
            Device.values = values
            return {"dps": values}

    monkeypatch.setattr(controller, "HAS_TINYTUYA", True)
    monkeypatch.setattr(controller, "tinytuya", SimpleNamespace(Device=Device))
    monkeypatch.setenv("SMART_ROOM_TUYA_BULB_KEY", "test-key")
    room = controller.TuyaController({"tuya": {"bulb": {
        "ip": "192.0.2.1", "device_id": "bulb", "protocol": "3.5",
        "brightness_max": 1000, "color_temp_max": 1000,
        "dps": {"switch": 20, "mode": 21, "brightness": 22, "color_temp": 23, "color": 24},
    }}})

    expected = {
        "20": True, "21": "colour", "24": "000003e80190",
    }
    assert room.set_light(on=True, brightness=40, rgb=[255, 0, 0])["dps"] == expected
    assert Device.values == expected


def test_tuya_reconnects_after_the_circuit_breaker_backoff(monkeypatch):
    from plugins.smart_room.runtime.tuya import controller

    room = controller.TuyaController({})
    reconnect = MagicMock()
    monkeypatch.setattr(room, "_connect_device", reconnect)
    room._health["bulb"]["consecutive_failures"] = 3
    room._health["bulb"]["circuit_open_until"] = time.monotonic() - 1
    try:
        result = room._run("bulb", "get_status", lambda: {"success": True})

        assert result["success"] is True
        reconnect.assert_called_once_with("bulb")
        assert room._health["bulb"]["consecutive_failures"] == 0
    finally:
        room.stop()


def test_manual_device_refresh_reconnects_then_polls():
    runtime = Runtime({})
    runtime._tuya = MagicMock()
    runtime._poll_devices = MagicMock()

    result = runtime.refresh_devices()

    assert result["success"] is True
    runtime._tuya.refresh.assert_called_once_with()
    runtime._poll_devices.assert_called_once_with()


@pytest.mark.skipif(sys.platform != "win32", reason="Smart Room runtime is Windows-only")
def test_runtime_crash_is_restarted_and_shutdown_is_clean():
    import yaml

    port = _free_port()
    home = process_manager._root().parent
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"smart_room": {
            "enabled": True,
            "runtime": {"rpc_port": port},
            "vision": {"enabled": False},
            "mqtt": {"broker": "127.0.0.1", "port": 1},
            "automations": {"evening_sleep": {"enabled": False}},
        }}),
        encoding="utf-8",
    )
    config = {
        "enabled": True,
        "runtime": {"rpc_port": port},
        "mqtt": {"port": 1},
        "vision": {"enabled": False},
    }
    process_manager.stop_supervisor()
    try:
        started = process_manager.start_supervisor(config)
        first_pid = int(started["pid"])
        _wait_for(_rpc_ready)

        # A raw local request without the supervisor token is rejected.
        with socket.create_connection(("127.0.0.1", port), timeout=2) as sock:
            sock.sendall(b'{"jsonrpc":"2.0","id":"x","method":"get_state","params":{}}\n')
            response = json.loads(sock.makefile("rb").readline())
        assert "unauthorized" in response["error"]

        subprocess.run(
            ["taskkill", "/PID", str(first_pid), "/F"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        unavailable = json.loads(handle_smart_room_state({}))
        assert unavailable["success"] is False
        assert unavailable["code"] == "DEVICE_TIMEOUT"
        restarted = _wait_for(
            lambda: (
                current if (current := process_manager.status()).get("alive")
                and int(current.get("pid", 0)) != first_pid else None
            )
        )
        _wait_for(_rpc_ready)
        assert restarted["restart_count"] >= 1
    finally:
        process_manager.stop_supervisor()
    assert not process_manager.status().get("alive")
