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


def _worker(frames):
    """A VisionWorker with a fake camera, for the burst."""
    from plugins.smart_room.runtime.vision import VisionWorker

    class _Capture:
        def __init__(self):
            self.size = (640, 480)
            self.reads = 0

        def set(self, prop, value):
            # 3 is CAP_PROP_FRAME_WIDTH, 4 is CAP_PROP_FRAME_HEIGHT.
            self.size = (value, self.size[1]) if prop == 3 else (self.size[0], value)

        def get(self, prop):
            return self.size[0] if prop == 3 else self.size[1]

        def read(self):
            self.reads += 1
            return (True, frames.pop(0)) if frames else (False, None)

        def isOpened(self):
            return True

    worker = VisionWorker.__new__(VisionWorker)
    return worker, _Capture()


def test_photographs_are_taken_at_full_resolution_not_the_recognition_size(tmp_path) -> None:
    """Face recognition runs small on purpose; a photograph must not.

    Reading `_latest_frame` handed back the recognition-sized frame, which is
    exactly the wrong one -- it is looked at once, by a person, who needs to be
    able to say who that was.
    """
    import numpy as np

    from plugins.smart_room.runtime.vision import VisionWorker

    frames = [np.zeros((1080, 1920, 3), dtype="uint8") for _ in range(3)]
    worker, capture = _worker(frames)
    worker.config = {"width": 1280, "height": 720, "photo_width": 1920, "photo_height": 1080}
    worker._capture = capture
    worker._lock = __import__("threading").Lock()
    worker._latest_frame = np.zeros((480, 640, 3), dtype="uint8")
    worker._stop = __import__("threading").Event()

    class _Library:
        dir = tmp_path

    worker.library = _Library()

    result = worker.photograph(count=3, gap=0.0)

    assert result["count"] == 3
    assert all(p["width"] == 1920 and p["height"] == 1080 for p in result["photos"]), (
        "photographed at the recognition size"
    )
    # And the stream is handed back at the size recognition expects.
    assert capture.size == (1280, 720), f"left the camera at {capture.size}"


def test_a_photograph_falls_back_rather_than_failing(tmp_path) -> None:
    # No camera handle: the recognition frame is worse than a full one and
    # much better than nothing.
    import numpy as np

    from plugins.smart_room.runtime.vision import VisionWorker

    worker = VisionWorker.__new__(VisionWorker)
    worker.config = {}
    worker._capture = None
    worker._lock = __import__("threading").Lock()
    worker._latest_frame = np.zeros((480, 640, 3), dtype="uint8")
    worker._stop = __import__("threading").Event()

    class _Library:
        dir = tmp_path

    worker.library = _Library()

    result = worker.photograph(count=2, gap=0.0)
    assert result["count"] == 2
    assert result["photos"][0]["width"] == 640


def test_asking_the_phone_is_reachable_over_rpc() -> None:
    """The other direction: the whole picture is built on the phone
    volunteering, so there was no way to tell a phone that is somewhere quiet
    from one that has stopped talking altogether."""
    from plugins.smart_room.runtime.command_router import CommandRouter

    runtime = _runtime(home=True)
    runtime._mqtt = MagicMock()
    runtime._mqtt.ask_phone_to_report.return_value = True
    router = CommandRouter(runtime._state, {}, runtime)

    answer = router.dispatch("ask_phone", {})

    assert answer["status"] == "success"
    assert answer["asked"] is True
    runtime._mqtt.ask_phone_to_report.assert_called_once()


def test_asking_says_so_when_mqtt_is_down() -> None:
    from plugins.smart_room.runtime.command_router import CommandRouter

    runtime = _runtime(home=True)
    runtime._mqtt = MagicMock()
    runtime._mqtt.ask_phone_to_report.return_value = False
    router = CommandRouter(runtime._state, {}, runtime)

    answer = router.dispatch("ask_phone", {})
    assert answer["status"] == "failed"
    assert "not connected" in answer["error"]


def test_the_owner_who_never_left_is_still_the_owner() -> None:
    """mmWave lost him sitting still and found him again when he moved.

    The phone's last word predates the last time he was identified in the
    room, so nothing has reported him leaving. On 10 September this was
    "you have company, someone is in the room", eight times, at his own desk.
    """
    runtime = _runtime(home=False, last_geofence_at=_at(600))
    runtime._state.last_owner_seen_at = _at(240)

    assert runtime._classify_entry(_at()) == ("owner", "no_departure_since_owner_seen")


def test_a_phone_that_left_after_he_was_seen_still_counts() -> None:
    # The ceiling of the rule above: a departure the phone did report wins.
    runtime = _runtime(home=False, last_geofence_at=_at(90), battery_percent=80)
    runtime._state.last_owner_seen_at = _at(240)

    assert runtime._classify_entry(_at())[0] == "unidentified"


def test_turning_over_in_sleep_mode_is_not_an_arrival(monkeypatch) -> None:
    """Seventeen "Unidentified entered the room" in one night of sleep mode.

    Each one was photographed in the dark, spoken or texted, and queued as a
    visitor for the next report. Nobody came in.
    """
    runtime = _runtime(home=False, last_geofence_at="2026-08-24T21:46:54+00:00")
    runtime._state.modes.active_mode = "sleep"
    runtime._pending_entry_at = _at()
    emitted = MagicMock()
    monkeypatch.setattr(runtime, "_emit_event", emitted)
    runtime._vision = MagicMock()

    runtime._deliver_welcome()

    emitted.assert_not_called()
    runtime._vision.photograph.assert_not_called()
    assert runtime._state.unreported_visitor_entries == []
