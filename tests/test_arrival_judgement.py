"""Who came in, when the sensors do not simply agree.

The rules this file exercises were four fixed ones, and each believed the last
reading it had whatever its age. Two things break that: a phone can go quiet
because it is dead rather than because its owner left, and a geofence can be a
fortnight stale -- the owner's real state had `home: true` from an event
thirteen days old.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from unittest.mock import MagicMock

import pytest

from plugins.smart_room.runtime.app import Runtime


def _at(minutes_ago: float = 0.0) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _runtime(**location) -> Runtime:
    runtime = Runtime({"owner": "Shereef", "welcome": {"owner_evidence_window_seconds": 3600}})
    runtime._state.mmwave.occupied = True
    for key, value in location.items():
        setattr(runtime._state.location, key, value)
    return runtime


def test_a_recognised_face_outranks_everything() -> None:
    """It does not need the mmWave or the phone to agree, and used not to be
    consulted here at all."""
    runtime = _runtime(home=False, last_geofence_at=_at(1))
    runtime._state.vision.owner_visible = True
    runtime._state.vision.stale = False

    assert runtime._classify_entry(_at())[0] == "owner"


def test_a_dying_phone_is_not_evidence_of_a_stranger() -> None:
    """Somebody is here, the phone says away, and it was at 8% and unplugged.

    The likeliest reason it stopped talking is that it died. Calling its owner
    an intruder over a flat battery is the wrong answer.
    """
    runtime = _runtime(home=False, last_geofence_at=_at(5), battery_percent=8, battery_state=1)
    verdict, why = runtime._classify_entry(_at())
    assert verdict == "unidentified"
    assert "battery" in why


def test_a_healthy_phone_somewhere_else_is_a_stranger() -> None:
    runtime = _runtime(home=False, last_geofence_at=_at(5), battery_percent=84, battery_state=1)
    assert runtime._classify_entry(_at()) == ("unknown_visitor", "owner_phone_away")


def test_a_charging_phone_at_low_percent_is_still_believed() -> None:
    # Plugged in at 8% is a phone that is fine, not one about to die.
    runtime = _runtime(home=False, last_geofence_at=_at(5), battery_percent=8, battery_state=2)
    assert runtime._classify_entry(_at())[0] == "unknown_visitor"


def test_a_stale_away_reading_does_not_accuse_anybody() -> None:
    """The thirteen-day-old geofence, in the direction that matters.

    "Away", from a reading old enough to mean nothing, is not the same claim
    as "a stranger is here" -- and the old rules made the second one.
    """
    runtime = _runtime(home=False, last_geofence_at="2026-08-24T21:46:54+00:00")
    verdict, why = runtime._classify_entry(_at())
    assert verdict == "unidentified"
    assert "stale" in why


def test_the_deep_sleep_case_is_still_the_owner() -> None:
    """The phone said home recently and BLE has gone quiet, which is what an
    iPhone does rather than evidence of anything."""
    runtime = _runtime(home=True, last_geofence_at=_at(10))
    assert runtime._classify_entry(_at()) == ("owner", "owntracks_recent")


@pytest.mark.parametrize("battery", [None, 90, 26])
def test_a_healthy_battery_never_excuses_an_absent_phone(battery) -> None:
    runtime = _runtime(home=False, last_geofence_at=_at(5), battery_percent=battery, battery_state=1)
    assert runtime._classify_entry(_at())[0] == "unknown_visitor"


def test_the_visitor_burst_does_not_turn_the_light_on_in_sleep_mode(monkeypatch) -> None:
    """The worst outcome available here, rather than a detail.

    Somebody asleep in a dark room is a person the camera cannot see and a
    phone that has stopped advertising -- which classifies as "unidentified",
    which lands in the photo burst. Turning the light on at 80% to photograph
    the owner in his own bed is not a security feature, it is an alarm clock.
    """
    runtime = _runtime(home=True)
    runtime._state.modes.active_mode = "sleep"
    runtime._state.light.on = False

    lights = MagicMock()
    monkeypatch.setattr(runtime, "set_light", lights)
    monkeypatch.setattr(runtime, "_emit_event", MagicMock())
    runtime._vision = MagicMock()
    runtime._vision.photograph.return_value = {
        "success": True, "photos": [{"at": _at(), "path": "a.jpg", "index": 1}]
    }

    runtime._photograph_visitor(_at(), "unidentified")

    lights.assert_not_called()
    runtime._vision.photograph.assert_called_once()


def test_the_visitor_burst_lights_the_room_when_nobody_is_asleep(monkeypatch) -> None:
    # An unrecognised person at night photographs as a dark shape, and the
    # whole purpose is that the owner can look afterwards and say who it was.
    runtime = _runtime(home=False)
    runtime._state.modes.active_mode = "off"
    runtime._state.light.on = False

    lights = MagicMock()
    monkeypatch.setattr(runtime, "set_light", lights)
    monkeypatch.setattr(runtime, "_emit_event", MagicMock())
    runtime._vision = MagicMock()
    runtime._vision.photograph.return_value = {
        "success": True, "photos": [{"at": _at(), "path": "a.jpg", "index": 1}]
    }

    runtime._photograph_visitor(_at(), "unknown_visitor")

    assert lights.call_count == 2, "did not light the room and put it back"
    assert lights.call_args_list[0].args[0] is True
    assert lights.call_args_list[-1].args[0] is False
