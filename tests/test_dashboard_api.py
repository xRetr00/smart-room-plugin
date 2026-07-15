"""Desktop Smart Room API contracts."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from plugins.smart_room.dashboard import plugin_api


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(plugin_api.router, prefix="/api/plugins/smart_room")
    return TestClient(app)


def test_status_reports_starting_runtime_without_failing(monkeypatch):
    monkeypatch.setattr(plugin_api, "status", lambda: {"alive": True, "pid": 42})
    monkeypatch.setattr(
        plugin_api,
        "call_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("starting")),
    )
    response = _client().get("/api/plugins/smart_room/status")
    assert response.status_code == 200
    assert response.json()["runtime"]["ready"] is False
    assert response.json()["state"] is None


def test_mode_calls_private_runtime_bridge(monkeypatch):
    calls = []
    monkeypatch.setattr(
        plugin_api,
        "call_runtime",
        lambda method, params: calls.append((method, params)) or {"success": True},
    )
    response = _client().post("/api/plugins/smart_room/mode", json={"mode": "focus"})
    assert response.status_code == 200
    assert calls == [("set_mode", {"mode": "focus"})]


def test_welcome_preview_calls_private_runtime_bridge(monkeypatch):
    calls = []
    monkeypatch.setattr(
        plugin_api,
        "call_runtime",
        lambda method, params: calls.append((method, params)) or {"success": True},
    )
    response = _client().post(
        "/api/plugins/smart_room/welcome/test", json={"audience": "owner"}
    )
    assert response.status_code == 200
    assert calls == [("test_welcome", {"audience": "owner"})]


def test_secret_fields_are_written_to_env_store(monkeypatch):
    saved = []
    monkeypatch.setattr(plugin_api, "save_env_value", lambda key, value: saved.append((key, value)))
    response = _client().put(
        "/api/plugins/smart_room/secrets",
        json={"bulb_key": "bulb-secret", "mqtt_password": "mqtt-secret"},
    )
    assert response.status_code == 200
    assert saved == [
        ("SMART_ROOM_TUYA_BULB_KEY", "bulb-secret"),
        ("SMART_ROOM_MQTT_PASSWORD", "mqtt-secret"),
    ]


def test_apply_restarts_only_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(plugin_api, "load_config", lambda: {"enabled": True})
    monkeypatch.setattr(plugin_api, "stop_supervisor", lambda: calls.append("stop"))
    monkeypatch.setattr(
        plugin_api,
        "start_supervisor",
        lambda config: calls.append(("start", config)) or {"pid": 7},
    )
    response = _client().post("/api/plugins/smart_room/apply")
    assert response.status_code == 200
    assert calls == ["stop", ("start", {"enabled": True})]
