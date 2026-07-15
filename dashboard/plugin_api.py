"""Authenticated desktop API for Smart Room settings and controls."""

from __future__ import annotations

import asyncio
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from hermes_cli.config import save_env_value
from plugins.smart_room.bridge import call_runtime
from plugins.smart_room.process_manager import start_supervisor, status, stop_supervisor
from plugins.smart_room.runtime.state_store import load_config

router = APIRouter()


class ModeBody(BaseModel):
    mode: str


class OverrideBody(BaseModel):
    enabled: bool


class WelcomeTestBody(BaseModel):
    audience: Literal["owner", "guest"]


class LightBody(BaseModel):
    on: Optional[bool] = None
    brightness: Optional[int] = Field(default=None, ge=0, le=100)
    color_temp: Optional[int] = Field(default=None, ge=2200, le=6500)
    rgb: Optional[List[int]] = None


class SecretsBody(BaseModel):
    bulb_key: Optional[str] = Field(default=None, max_length=256)
    he20_key: Optional[str] = Field(default=None, max_length=256)
    mqtt_username: Optional[str] = Field(default=None, max_length=256)
    mqtt_password: Optional[str] = Field(default=None, max_length=4096)


async def _rpc(method: str, params: dict) -> dict:
    try:
        return await asyncio.to_thread(call_runtime, method, params)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/status")
async def get_status() -> dict:
    process = await asyncio.to_thread(status)
    if not process.get("alive"):
        return {"runtime": process, "state": None, "health": None}
    state, health = await asyncio.gather(
        asyncio.to_thread(call_runtime, "get_state", {}),
        asyncio.to_thread(call_runtime, "get_health", {}),
        return_exceptions=True,
    )
    ready = not isinstance(state, Exception) and not isinstance(health, Exception)
    process["ready"] = ready
    return {
        "runtime": process,
        "state": state.get("state") if isinstance(state, dict) else None,
        "health": health.get("health") if isinstance(health, dict) else None,
    }


@router.post("/mode")
async def set_mode(body: ModeBody) -> dict:
    if body.mode not in {"reading", "focus", "relax", "sleep", "alarm", "off"}:
        raise HTTPException(status_code=400, detail="invalid mode")
    return await _rpc("set_mode", {"mode": body.mode})


@router.post("/light")
async def set_light(body: LightBody) -> dict:
    params = body.model_dump(exclude_none=True)
    if not params:
        raise HTTPException(status_code=400, detail="at least one light field is required")
    if body.rgb is not None and (len(body.rgb) != 3 or any(not 0 <= value <= 255 for value in body.rgb)):
        raise HTTPException(status_code=400, detail="rgb must contain three values from 0 to 255")
    return await _rpc("set_light", params)


@router.post("/override")
async def set_override(body: OverrideBody) -> dict:
    return await _rpc("set_override", {"enabled": body.enabled})


@router.post("/cancel-sleep")
async def cancel_sleep() -> dict:
    return await _rpc("cancel_sleep", {})


@router.post("/welcome/test")
async def test_welcome(body: WelcomeTestBody) -> dict:
    return await _rpc("test_welcome", {"audience": body.audience})


@router.post("/apply")
async def apply_config() -> dict:
    config = load_config()
    if not config.get("enabled", False):
        await asyncio.to_thread(stop_supervisor)
        return {"ok": True, "enabled": False}
    await asyncio.to_thread(stop_supervisor)
    result = await asyncio.to_thread(start_supervisor, config)
    return {"ok": True, "enabled": True, "runtime": result}


@router.put("/secrets")
async def save_secrets(body: SecretsBody) -> dict:
    mapping = {
        "bulb_key": "SMART_ROOM_TUYA_BULB_KEY",
        "he20_key": "SMART_ROOM_TUYA_HE20_KEY",
        "mqtt_username": "SMART_ROOM_MQTT_USERNAME",
        "mqtt_password": "SMART_ROOM_MQTT_PASSWORD",
    }
    values = body.model_dump(exclude_none=True)
    for field, env_name in mapping.items():
        if field in values:
            await asyncio.to_thread(save_env_value, env_name, values[field])
    return {"ok": True, "saved": sorted(values)}
