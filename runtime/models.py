"""Data models for the smart_room runtime.

Pydantic-free dataclasses for type safety without adding a dependency.
State is serialized to JSON for the state store and bridge.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


VALID_MODES = {"off", "reading", "focus", "relax", "night", "sleep", "alarm"}


@dataclass
class Presence:
    """Fused presence state from BLE + mmWave + geofence."""
    detected: bool = False
    source: str = "none"  # "ble", "mmwave", "geofence", "ble_sticky_mmwave", "none"
    confidence: float = 0.0
    rssi: Optional[int] = None
    room: str = "smart_room"
    last_seen: Optional[str] = None  # ISO timestamp
    identity_sticky: bool = False  # True when held by mmWave after BLE went silent
    sticky_since: Optional[str] = None  # When sticky identity started


@dataclass
class MmWaveState:
    """HE20 mmWave presence sensor state."""
    occupied: bool = False
    last_seen: Optional[str] = None


@dataclass
class LightState:
    """RGBCW bulb state."""
    on: bool = False
    brightness: int = 0  # 0-100
    color_temp: int = 3000  # Kelvin
    rgb: Optional[List[int]] = None  # [R, G, B] or None for white mode
    scene: str = "off"  # "reading", "focus", "relax", "night", "alarm", "off", "custom"
    confirmed: bool = False
    last_error: Optional[str] = None


@dataclass
class Modes:
    """One authoritative mode plus presence-automation override policy."""
    active_mode: str = "off"
    manual_override: str = "none"  # none | hold_on | hold_off
    work_return: bool = False

    def is_active(self, mode: str) -> bool:
        return self.active_mode == mode


@dataclass
class DailyFlags:
    """Per-day automation flags, reset at midnight."""
    work_sleep_done_today: bool = False
    work_sleep_cancel_today: bool = False
    evening_sleep_done_today: bool = False
    evening_sleep_cancel_today: bool = False
    last_reset: Optional[str] = None


@dataclass
class PhoneLocation:
    """Phone geofence state from OwnTracks."""
    home: bool = False
    zone: str = "unknown"  # "home", "university", "bakery", "away"
    since: Optional[str] = None
    source: str = "unknown"  # "owntracks", "ble", "unknown"
    last_geofence_at: Optional[str] = None
    last_event_key: Optional[str] = None


@dataclass
class DeviceHealth:
    """Device connectivity state."""
    online: bool = False
    ip: Optional[str] = None
    last_seen: Optional[str] = None
    last_poll: Optional[str] = None
    consecutive_failures: int = 0
    last_success: Optional[str] = None
    last_command: Optional[str] = None
    queue_depth: int = 0
    circuit_open: bool = False


@dataclass
class Alarm:
    id: str
    name: str
    time: str
    recurrence: str = "daily"  # once | daily
    date: Optional[str] = None
    enabled: bool = True
    duration_minutes: int = 30
    last_fired_at: Optional[str] = None


@dataclass
class ActiveAlarm:
    id: str
    name: str
    started_at: str
    flash_until: str
    expires_at: str
    phase: str = "flashing"  # flashing | steady


@dataclass
class RoomState:
    """Full room state snapshot — serialized to state.json."""
    presence: Presence = field(default_factory=Presence)
    mmwave: MmWaveState = field(default_factory=MmWaveState)
    light: LightState = field(default_factory=LightState)
    modes: Modes = field(default_factory=Modes)
    flags: DailyFlags = field(default_factory=DailyFlags)
    location: PhoneLocation = field(default_factory=PhoneLocation)
    devices: Dict[str, DeviceHealth] = field(default_factory=dict)
    weather: Dict[str, Any] = field(default_factory=dict)
    sun: Dict[str, Any] = field(default_factory=dict)
    last_updated: Optional[str] = None
    event_id: int = 0  # monotonic event counter for subconscious cursor
    sleep_restore: Dict[str, Any] = field(default_factory=dict)
    room_empty_since: Optional[str] = None
    last_welcome_at: Optional[str] = None
    alarms: List[Alarm] = field(default_factory=list)
    active_alarm: Optional[ActiveAlarm] = None
    alarm_restore: Dict[str, Any] = field(default_factory=dict)
    pending_mode: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict."""
        d = asdict(self)
        d["last_updated"] = self.last_updated
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RoomState":
        """Deserialize from dict (partial — missing fields use defaults)."""
        state = cls()
        if "presence" in d:
            state.presence = Presence(**d["presence"])
        if "mmwave" in d:
            state.mmwave = MmWaveState(**d["mmwave"])
        if "light" in d:
            state.light = LightState(**d["light"])
        if "modes" in d:
            raw_modes = dict(d["modes"])
            active = raw_modes.get("active_mode")
            if active not in VALID_MODES:
                active = next(
                    (name for name in ("reading", "focus", "relax", "night", "sleep", "alarm") if raw_modes.get(name)),
                    "off",
                )
            override = raw_modes.get("manual_override", "none")
            if isinstance(override, bool):
                override = ("hold_on" if state.light.on else "hold_off") if override else "none"
            if override not in {"none", "hold_on", "hold_off"}:
                override = "none"
            state.modes = Modes(
                active_mode=active,
                manual_override=override,
                work_return=bool(raw_modes.get("work_return", False)),
            )
        if "flags" in d:
            state.flags = DailyFlags(**d["flags"])
        if "location" in d:
            location = dict(d["location"])
            # One-time migration from the removed iOS Shortcuts design.
            location.setdefault("last_geofence_at", location.pop("last_webhook_at", None))
            state.location = PhoneLocation(**location)
        if "devices" in d:
            state.devices = {
                k: DeviceHealth(**v) for k, v in d["devices"].items()
            }
        if "weather" in d:
            state.weather = d["weather"]
        if "sun" in d:
            state.sun = d["sun"]
        if "last_updated" in d:
            state.last_updated = d["last_updated"]
        if "event_id" in d:
            state.event_id = d["event_id"]
        if "sleep_restore" in d and isinstance(d["sleep_restore"], dict):
            state.sleep_restore = d["sleep_restore"]
        state.room_empty_since = d.get("room_empty_since")
        state.last_welcome_at = d.get("last_welcome_at")
        state.alarms = [Alarm(**item) for item in d.get("alarms", []) if isinstance(item, dict)]
        if isinstance(d.get("active_alarm"), dict):
            state.active_alarm = ActiveAlarm(**d["active_alarm"])
        if isinstance(d.get("alarm_restore"), dict):
            state.alarm_restore = d["alarm_restore"]
        if isinstance(d.get("pending_mode"), dict):
            state.pending_mode = d["pending_mode"]
        return state


def now_iso() -> str:
    """Current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat()
