"""Tests for the smart_room presence fusion axioms.

Per v0.3 §G: fusion axioms are hard rules, unit-tested.
"""

import sys
import os
import pytest
from datetime import datetime, timezone, timedelta

# Add plugin to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugins.smart_room.runtime.models import Presence, MmWaveState, PhoneLocation, now_iso
from plugins.smart_room.runtime.presence_fusion import fuse, _decay_confidence


class TestFusionAxioms:
    """The 6 hard fusion axioms from v0.3 §D."""

    def test_axiom1_ble_presence_is_strong_evidence_for_identity(self):
        """BLE presence → identity established with high confidence."""
        p = Presence()
        mm = MmWaveState()
        loc = PhoneLocation()
        p, mm, loc, light_on, light_off = fuse(
            p, mm, loc, ble_detected=True, ble_rssi=-65,
            mmwave_occupied=True, geofence_zone=None, exit_timeout_elapsed=False
        )
        assert p.detected is True
        assert p.confidence >= 0.9
        assert p.source == "ble"

    def test_axiom2_ble_silence_is_weak_evidence_never_sufficient_for_absent(self):
        """BLE absent + mmWave occupied → identity held (sticky), NOT absence."""
        p = Presence(detected=True, confidence=0.95, source="ble", last_seen=now_iso())
        mm = MmWaveState(occupied=True, last_seen=now_iso())
        loc = PhoneLocation()
        p, mm, loc, light_on, light_off = fuse(
            p, mm, loc, ble_detected=False, ble_rssi=None,
            mmwave_occupied=True, geofence_zone=None, exit_timeout_elapsed=False
        )
        # Identity must be held (sticky)
        assert p.detected is True
        assert p.identity_sticky is True
        assert p.source == "ble_sticky_mmwave"
        assert p.confidence >= 0.6  # floor

    def test_axiom3_sticky_releases_on_mmwave_clear(self):
        """mmWave clear → sticky identity released."""
        p = Presence(detected=True, confidence=0.7, source="ble_sticky_mmwave",
                     identity_sticky=True, last_seen=now_iso())
        mm = MmWaveState(occupied=False)
        loc = PhoneLocation()
        p, mm, loc, light_on, light_off = fuse(
            p, mm, loc, ble_detected=False, ble_rssi=None,
            mmwave_occupied=False, geofence_zone=None, exit_timeout_elapsed=True
        )
        assert p.detected is False
        assert p.identity_sticky is False
        assert p.confidence == 0.0

    def test_axiom4_light_off_requires_mmwave_clear_and_ble_absent(self):
        """Light off only when both mmWave clear AND BLE absent."""
        p = Presence(detected=True, confidence=0.9, source="ble", last_seen=now_iso())
        mm = MmWaveState(occupied=True, last_seen=now_iso())
        loc = PhoneLocation()
        _, _, _, light_on, light_off = fuse(
            p, mm, loc, ble_detected=False, ble_rssi=None,
            mmwave_occupied=True, geofence_zone=None, exit_timeout_elapsed=False
        )
        # mmWave still occupied → light should NOT turn off
        assert light_off is False

    def test_axiom5_mmwave_alone_never_upgrades_to_identity(self):
        """mmWave occupied without prior BLE/geofence → no identity."""
        p = Presence(detected=False)
        mm = MmWaveState(occupied=False)
        loc = PhoneLocation()
        p, mm, loc, light_on, _ = fuse(
            p, mm, loc, ble_detected=False, ble_rssi=None,
            mmwave_occupied=True, geofence_zone=None, exit_timeout_elapsed=False
        )
        # Someone is there but we don't know who
        assert p.detected is False
        assert p.source == "mmwave_only"
        assert p.confidence == 0.0
        assert light_on is True

    def test_axiom5b_mmwave_alone_with_geofence_home_can_upgrade(self):
        """mmWave + geofence home → identity can be established (geofence is identity signal)."""
        # First establish via geofence
        p = Presence(detected=True, confidence=0.95, source="geofence", last_seen=now_iso())
        mm = MmWaveState(occupied=True, last_seen=now_iso())
        loc = PhoneLocation(home=True, zone="home")
        # Now BLE goes silent but mmWave stays
        p, mm, loc, _, _ = fuse(
            p, mm, loc, ble_detected=False, ble_rssi=None,
            mmwave_occupied=True, geofence_zone="home", exit_timeout_elapsed=False
        )
        assert p.detected is True


class TestStickyDecay:
    """Confidence decay curve tests."""

    def test_decay_starts_at_ceiling(self):
        """Fresh sticky identity has ceiling confidence."""
        now = now_iso()
        conf = _decay_confidence(now, now)
        assert conf == 0.95

    def test_decay_floors_at_06_after_2_hours(self):
        """After 2 hours, confidence floors at 0.6."""
        past = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        conf = _decay_confidence(past)
        assert conf == 0.6

    def test_decay_decreases_monotonically(self):
        """Confidence decreases over time, never increases."""
        now = now_iso()
        c1 = _decay_confidence(now, now)
        past_30s = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()
        c2 = _decay_confidence(past_30s, now)
        assert c2 < c1

    def test_four_hour_sleeping_phone_soak_holds_identity_and_light(self):
        """Four hours of BLE silence cannot darken an mmWave-occupied room."""
        sticky_since = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()
        p = Presence(
            detected=True,
            confidence=0.6,
            source="ble_sticky_mmwave",
            identity_sticky=True,
            sticky_since=sticky_since,
            last_seen=sticky_since,
        )
        p, _, _, light_on, light_off = fuse(
            p,
            MmWaveState(occupied=True, last_seen=now_iso()),
            PhoneLocation(),
            ble_detected=False,
            ble_rssi=None,
            mmwave_occupied=True,
            geofence_zone=None,
            exit_timeout_elapsed=False,
        )
        assert p.detected is True
        assert p.confidence == 0.6
        assert light_on is True
        assert light_off is False


class TestGeofenceRelease:
    """Geofence-based identity release tests."""

    def test_geofence_leave_releases_identity(self):
        """Geofence 'leave home' immediately releases identity."""
        p = Presence(detected=True, confidence=0.9, source="ble", identity_sticky=True)
        mm = MmWaveState(occupied=True, last_seen=now_iso())
        loc = PhoneLocation(home=True, zone="home")
        p, mm, loc, _, _ = fuse(
            p, mm, loc, ble_detected=True, ble_rssi=-60,
            mmwave_occupied=True, geofence_zone="away", exit_timeout_elapsed=False
        )
        assert p.detected is False
        assert p.identity_sticky is False
        assert loc.home is False
