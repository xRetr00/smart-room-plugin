"""v0.3 acceptance contracts that cross the plugin's component seams."""

from __future__ import annotations

import ast
import json
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from plugins.smart_room import process_manager
from plugins.smart_room.bridge import call_runtime
from plugins.smart_room.bridge import build_context_line
from plugins.smart_room.runtime.app import Runtime
from plugins.smart_room.runtime.models import MmWaveState, PhoneLocation, Presence, RoomState
from plugins.smart_room.runtime.presence_fusion import fuse
from plugins.smart_room.runtime.scheduler import Scheduler
from plugins.smart_room.runtime.state_store import append_transition
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
    assert presence.source == "ble_sticky_mmwave"


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
    assert len(ctx.tools) == 7
    assert {name for name, _ in ctx.hooks} == {
        "on_gateway_start", "on_gateway_stop",
    }
    assert [name for name, _ in ctx.context] == ["smart_room"]


def test_context_provider_host_contract():
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    manager = PluginManager()
    context = PluginContext(PluginManifest(name="context-test"), manager)
    context.register_context_provider("room", lambda: "Room: focus.")
    assert manager.build_context_blocks() == ["Room: focus."]


def test_context_line_is_config_gated_and_uses_fused_source():
    import yaml
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    config_path = home / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "smart_room": {"enabled": True, "context": {"enabled": True}, "owner": "shereef"}
    }), encoding="utf-8")
    state = RoomState()
    state.presence = Presence(detected=True, source="ble_sticky_mmwave", confidence=0.6)
    state.light.on = True
    state.light.brightness = 70
    state.light.scene = "reading"
    state.modes.reading = True
    save_state(state)
    line = build_context_line()
    assert line is not None
    assert "Shereef present (ble_sticky_mmwave, conf 0.60)" in line
    assert "light 70% (reading) @3000K" in line

    config_path.write_text(yaml.safe_dump({
        "smart_room": {"enabled": True, "context": {"enabled": False}}
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


def test_subconscious_fetcher_baselines_then_returns_only_new_events():
    from cron.scripts.subconscious.smart_room import fetch_delta
    from cron.scripts.subconscious.snapshot_store import SurfaceStore

    append_transition({"id": 1, "at": "2026-07-15T10:00:00Z", "type": "mode_changed", "summary": "focus mode"})
    store = SurfaceStore("smart_room", min_interval_seconds=0)
    assert fetch_delta(store) is None
    store.save()

    append_transition({"id": 2, "at": "2026-07-15T10:05:00Z", "type": "presence_detected", "summary": "owner arrived"})
    store = SurfaceStore("smart_room", min_interval_seconds=0)
    delta = fetch_delta(store)
    assert delta is not None
    assert "owner arrived" in delta
    assert "focus mode" not in delta


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
            "mqtt": {"broker": "127.0.0.1", "port": 1},
            "automations": {"evening_sleep": {"enabled": False}},
        }}),
        encoding="utf-8",
    )
    config = {"enabled": True, "runtime": {"rpc_port": port}, "mqtt": {"port": 1}}
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
