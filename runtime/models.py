"""Data models for the smart_room runtime.

Pydantic-free dataclasses for type safety without adding a dependency.
State is serialized to JSON for the state store and bridge.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


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


@dataclass
class Modes:
    """Active mode flags (mutually exclusive for scene modes)."""
    reading: bool = False
    focus: bool = False
    relax: bool = False
    sleep: bool = False
    alarm: bool = False
    manual_override: bool = False
    work_return: bool = False


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
    home: bool = True
    zone: str = "home"  # "home", "university", "bakery", "away"
    since: Optional[str] = None
    source: str = "owntracks"  # "owntracks", "ble", "unknown"


@dataclass
class DeviceHealth:
    """Device connectivity state."""
    online: bool = False
    ip: Optional[str] = None
    last_seen: Optional[str] = None
    last_poll: Optional[str] = None


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
            state.modes = Modes(**d["modes"])
        if "flags" in d:
            state.flags = DailyFlags(**d["flags"])
        if "location" in d:
            state.location = PhoneLocation(**d["location"])
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
        return state


def now_iso() -> str:
    """Current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat()
