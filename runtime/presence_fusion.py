"""Presence fusion engine — BLE + mmWave + geofence.

Implements the v0.3 §D fusion axioms:
1. BLE presence is strong evidence FOR identity. BLE absence is weak.
2. Identity is sticky: once BLE establishes presence, mmWave holds it.
3. Confidence decays over time (0.95 → floor 0.6 over 2h).
4. Sticky releases when: mmWave clears, geofence says leave, or another identity appears.
5. Light-off requires mmWave clear + BLE absent.
6. mmWave alone never upgrades to identity without at least one BLE/geofence signal.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from plugins.smart_room.runtime.models import Presence, MmWaveState, PhoneLocation, now_iso

logger = logging.getLogger(__name__)

# Decay: confidence goes from 0.95 to floor 0.6 over 2 hours (7200s)
_STICKY_FLOOR = 0.6
_STICKY_CEILING = 0.95
_STICKY_DECAY_SECONDS = 7200  # 2 hours


def _decay_confidence(sticky_since: str, now: Optional[str] = None) -> float:
    """Calculate decayed confidence based on how long identity has been sticky."""
    try:
        from datetime import datetime, timezone
        start = datetime.fromisoformat(sticky_since.replace("Z", "+00:00"))
        if now is None:
            current = datetime.now(timezone.utc)
        else:
            current = datetime.fromisoformat(now.replace("Z", "+00:00"))
        elapsed = (current - start).total_seconds()
    except Exception:
        return _STICKY_FLOOR

    if elapsed <= 0:
        return _STICKY_CEILING
    if elapsed >= _STICKY_DECAY_SECONDS:
        return _STICKY_FLOOR

    ratio = elapsed / _STICKY_DECAY_SECONDS
    return _STICKY_CEILING - (_STICKY_CEILING - _STICKY_FLOOR) * ratio


def fuse(
    presence: Presence,
    mmwave: MmWaveState,
    location: PhoneLocation,
    ble_detected: bool,
    ble_rssi: Optional[int],
    mmwave_occupied: bool,
    geofence_zone: Optional[str],
    exit_timeout_elapsed: bool,
    wifi_detected: bool = False,
    other_identity_detected: bool = False,
) -> tuple[Presence, MmWaveState, PhoneLocation, bool, bool]:
    """Fuse all signals into updated presence state.

    Returns (new_presence, new_mmwave, new_location, light_should_on, light_should_off).
    """
    now = now_iso()

    # Update mmWave
    mmwave.occupied = mmwave_occupied
    if mmwave_occupied:
        mmwave.last_seen = now

    # Update geofence
    if geofence_zone is not None:
        location.zone = geofence_zone
        location.home = (geofence_zone == "home")
        location.since = now

    # --- Fusion logic ---

    # Case 1: BLE detected — strong evidence FOR identity
    if ble_detected:
        presence.detected = True
        presence.source = "ble"
        presence.confidence = _STICKY_CEILING
        presence.rssi = ble_rssi
        presence.last_seen = now
        presence.identity_sticky = False
        presence.sticky_since = None
        logger.debug("BLE detected — identity established (rssi=%s)", ble_rssi)

    # Positive-only fallback identity signals require current room occupancy.
    elif wifi_detected and mmwave_occupied:
        presence.detected = True
        presence.source = "wifi_mmwave"
        presence.confidence = 0.75
        presence.last_seen = now
        presence.identity_sticky = False
        presence.sticky_since = None

    elif location.home and mmwave_occupied and not presence.detected:
        presence.detected = True
        presence.source = "geofence_mmwave"
        presence.confidence = 0.7
        presence.last_seen = now
        presence.identity_sticky = False
        presence.sticky_since = None

    # Case 2: BLE absent but mmWave occupied and identity was previously established
    elif not ble_detected and mmwave_occupied and (presence.detected or presence.identity_sticky):
        # Sticky identity — phone is probably asleep
        if not presence.identity_sticky:
            presence.identity_sticky = True
            presence.sticky_since = now
            logger.info("BLE silent, mmWave occupied — entering sticky identity")

        presence.detected = True
        presence.source = "ble_sticky_mmwave"
        presence.confidence = _decay_confidence(presence.sticky_since or now, now)
        presence.rssi = None
        logger.debug("Sticky identity held (confidence=%.2f)", presence.confidence)

    # Case 3: mmWave occupied but no identity ever established this session
    elif mmwave_occupied and not presence.detected:
        presence.detected = False  # someone is there but we don't know who
        presence.source = "mmwave_only"
        presence.confidence = 0.0
        logger.debug("mmWave occupied but no identity signal")

    # Case 4: mmWave clear and BLE absent — room is empty
    elif not mmwave_occupied and not ble_detected:
        if exit_timeout_elapsed:
            presence.detected = False
            presence.source = "none"
            presence.confidence = 0.0
            presence.identity_sticky = False
            presence.sticky_since = None
            logger.debug("Room clear — presence released")

    # Case 5: geofence says left home — release identity immediately
    # Unknown is not evidence of absence. Only an explicit non-home zone may
    # override the positive room signals above.
    if location.zone not in {"home", "unknown", ""}:
        presence.detected = False
        presence.identity_sticky = False
        presence.sticky_since = None
        presence.source = "geofence_absent"
        presence.confidence = 0.0
        logger.info("Geofence says left home (%s) — identity released", location.zone)

    if other_identity_detected and not ble_detected:
        presence.detected = False
        presence.identity_sticky = False
        presence.sticky_since = None
        presence.source = "other_identity"
        presence.confidence = 0.0

    # --- Light decisions ---
    # Occupancy controls the room, identity only personalizes it. A guest (or
    # an owner whose phone is unavailable) must still get safe automatic light.
    light_should_on = mmwave_occupied or (presence.detected and presence.confidence > 0)

    # Light off: mmWave clear for timeout AND BLE absent
    light_should_off = (not mmwave_occupied and not ble_detected and exit_timeout_elapsed)

    return presence, mmwave, location, light_should_on, light_should_off
