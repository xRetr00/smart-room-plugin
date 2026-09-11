"""The lean face engine, faces followed across frames, and lighting the room to see."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from plugins.smart_room.runtime import face_engine
from plugins.smart_room.runtime.app import Runtime
from plugins.smart_room.runtime.vision import FaceLibrary, VisionWorker

FACING = [[40.0, 40.0], [80.0, 40.0], [60.0, 60.0], [45.0, 80.0], [75.0, 80.0]]
OWNER, STRANGER = [1.0, 0.0], [0.0, 1.0]


class FakeAnalyzer:
    capabilities = {"faces": True}
    face_model = "test"
    face_provider = "test"


def _face(embedding, *, box=(10.0, 10.0, 110.0, 110.0), live=None, quality=None, landmarks=FACING):
    return {
        "embedding": embedding,
        "bbox": list(box),
        "detection_score": 0.99,
        "landmarks": landmarks,
        "live": live,
        "quality": quality,
    }


def _worker(tmp_path, events=None, **config):
    worker = VisionWorker(
        {"enabled": True, "faces": {"min_face_size": 20, "min_blur_variance": 0}, **config},
        lambda _state: None,
        lambda kind, data: (events if events is not None else []).append((kind, data)),
        library=FaceLibrary(tmp_path / "vision"),
        analyzer=FakeAnalyzer(),
    )
    worker.library.enroll("Shereef", [OWNER], owner=True)
    return worker


def _see(worker, *faces, frame=None, **extra):
    worker._apply_analysis(frame, {"faces": list(faces), **extra}, moving=False)
    return worker.state


# -- the engine ---------------------------------------------------------------


def test_the_alignment_recovers_a_known_similarity() -> None:
    """The same fit scikit-image made, so enrolled embeddings stay valid."""
    angle, scale, shift = 0.3, 1.7, np.array([12.0, -5.0])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    source = face_engine.ARCFACE_POINTS.astype(np.float64)
    target = source @ (rotation * scale).T + shift

    matrix = face_engine.similarity_transform(source, target)

    assert np.allclose(matrix[:, :2], rotation * scale, atol=1e-6)
    assert np.allclose(matrix[:, 2], shift, atol=1e-6)


def test_liveness_reads_index_one_as_real() -> None:
    liveness = face_engine.Liveness.__new__(face_engine.Liveness)
    liveness.input = "x"
    seen = {}

    def run(_names, feed):
        seen["shape"] = feed["x"].shape
        return [np.array([[0.0, 4.0, 0.0]])]

    liveness.session = SimpleNamespace(run=run)
    real = liveness.real(np.zeros((480, 640, 3), dtype=np.uint8), [200, 150, 300, 280])

    assert seen["shape"] == (1, 3, 80, 80)
    assert real > 0.9


# -- faces over frames ----------------------------------------------------------


def test_the_owner_stays_the_owner_when_he_turns_away(tmp_path) -> None:
    worker = _worker(tmp_path)
    try:
        assert _see(worker, _face(OWNER)).owner_visible
        turned = _face(OWNER, landmarks=[[40, 40], [80, 40], [86, 60], [45, 80], [75, 80]])
        state = _see(worker, turned)
        assert state.owner_visible and state.owner_seen_by == "face"
    finally:
        worker.stop()


def test_one_frame_of_a_stranger_is_not_a_visitor_three_are_and_once(tmp_path) -> None:
    events: list = []
    worker = _worker(tmp_path, events)
    try:
        _see(worker, _face(STRANGER))
        assert not [kind for kind, _ in events if kind == "vision_visitor_seen"]
        for _ in range(4):
            _see(worker, _face(STRANGER))
        assert [kind for kind, _ in events].count("vision_visitor_seen") == 1
    finally:
        worker.stop()


def test_the_owners_face_on_a_screen_is_a_visitor_not_the_owner(tmp_path) -> None:
    events: list = []
    worker = _worker(tmp_path, events)
    worker.enforce_liveness = True
    try:
        for _ in range(3):
            state = _see(worker, _face(OWNER, live=0.02))
        assert not state.owner_visible
        seen = [data for kind, data in events if kind == "vision_visitor_seen"]
        assert seen and seen[0]["spoofed"] is True
    finally:
        worker.stop()


def test_the_owner_is_known_by_his_outfit_when_no_face_is_on_offer(tmp_path) -> None:
    worker = _worker(tmp_path)
    blue = [1.0] + [0.0] * 127
    red = [0.0] * 127 + [1.0]
    try:
        _see(worker, _face(OWNER), bodies=[{"box": [0, 0, 1, 1], "outfit": blue}])
        worker.tracks.tracks.clear()
        state = _see(worker, bodies=[{"box": [0, 0, 1, 1], "outfit": blue}])
        assert state.owner_visible and state.owner_seen_by == "outfit"
        assert not _see(worker, bodies=[{"box": [0, 0, 1, 1], "outfit": red}]).owner_visible
        # Any identified face outranks clothes.
        worker.tracks.tracks.clear()
        for _ in range(3):
            state = _see(worker, _face(STRANGER), bodies=[{"box": [0, 0, 1, 1], "outfit": blue}])
        assert not state.owner_visible
    finally:
        worker.stop()


def test_a_close_face_that_cannot_be_named_is_flagged(tmp_path) -> None:
    worker = _worker(tmp_path)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        close = _face([0.3, 0.954], box=(200.0, 100.0, 330.0, 260.0))
        assert _see(worker, close, frame=frame).close_face_unidentified
        far = _face([0.3, 0.954], box=(10.0, 10.0, 40.0, 40.0))
        worker.tracks.tracks.clear()
        assert not _see(worker, far, frame=frame).close_face_unidentified
    finally:
        worker.stop()


def test_movement_is_reported_per_zone(tmp_path) -> None:
    worker = _worker(tmp_path, zones={"door": "[[0.88,0.0],[1.0,0.0],[1.0,1.0],[0.88,1.0]]"})
    try:
        changed = np.zeros((120, 160), dtype=bool)
        assert worker._zone_motion(changed) == {}
        changed[:, 150:] = True
        assert "door" in worker._zone_motion(changed)
    finally:
        worker.stop()


# -- the library ------------------------------------------------------------------


def test_curation_removes_what_is_not_him_and_keeps_a_backup(tmp_path) -> None:
    library = FaceLibrary(tmp_path / "vision")
    try:
        rng = np.random.default_rng(1)
        him = [list(np.array([1.0, 0.0, 0.0]) + rng.normal(0, 0.1, 3)) for _ in range(12)]
        library.enroll("Shereef", him + [[-1.0, 0.2, 0.0]], owner=True)

        assert library.curate_owner() == 1
        assert library.people()[0]["samples"] == 12
        assert list((tmp_path / "vision").glob("faces.sqlite3.*.bak"))
    finally:
        library.close()


def test_learning_takes_only_new_clear_views_and_not_too_often(tmp_path) -> None:
    library = FaceLibrary(tmp_path / "vision")
    try:
        library.enroll("Shereef", [OWNER], owner=True)
        assert not library.learn_owner([0.99, 0.14], 0.99), "a near-duplicate is not new"
        assert not library.learn_owner([0.2, 0.98], 0.2), "not clearly him"
        assert library.learn_owner([0.6, 0.8], 0.6)
        assert not library.learn_owner([0.55, 0.835], 0.55), "twice inside the rate limit"
    finally:
        library.close()


# -- lighting the room to see -------------------------------------------------------


def _runtime(mode: str = "off") -> Runtime:
    runtime = Runtime({"owner": "Shereef"})
    runtime._state.modes.active_mode = mode
    runtime._state.light.on = False
    runtime.set_light = MagicMock(return_value={"success": True})
    return runtime


def test_in_sleep_mode_only_a_close_face_lights_the_room_and_only_faintly() -> None:
    runtime = _runtime("sleep")
    runtime._state.vision.dark = True

    runtime._state.vision.close_face_unidentified = False
    assert not runtime._wants_to_see(runtime._state.vision)

    runtime._state.vision.close_face_unidentified = True
    assert runtime._wants_to_see(runtime._state.vision)
    runtime._config = {"vision": {"see_light": {"sleep_brightness": 60}}}
    assert runtime._light_to_see()
    kwargs = runtime.set_light.call_args.kwargs
    assert kwargs["brightness"] == 5, "sleep mode is capped at 5% whatever the config says"
    assert kwargs["color_temp"] == 2200


def test_awake_the_light_is_low_and_not_switched_on_twice_in_ten_minutes() -> None:
    runtime = _runtime("off")
    assert runtime._light_to_see()
    assert runtime.set_light.call_args.kwargs["brightness"] == 15
    runtime._state.light.on = False
    assert not runtime._light_to_see()


def test_the_see_light_is_switched_off_only_if_nobody_touched_it() -> None:
    runtime = _runtime("off")
    runtime._state.light.on = True
    runtime._see_light_ours, runtime._see_light_token = True, 3
    runtime._stop_seeing(3)
    runtime.set_light.assert_called_once_with(False, purpose="see")

    runtime.set_light.reset_mock()
    runtime._see_light_ours = False  # somebody changed the light in between
    runtime._stop_seeing(3)
    runtime.set_light.assert_not_called()


def test_an_arrival_needs_the_door_to_have_moved() -> None:
    runtime = _runtime("off")
    runtime._vision = SimpleNamespace(zones={"door": [[0.88, 0], [1, 0], [1, 1], [0.88, 1]]})
    vision = runtime._state.vision
    vision.camera_open, vision.stale, vision.dark, vision.low_power = True, False, False, False
    entry = datetime.now(timezone.utc)

    vision.zone_motion = {}
    assert runtime._nobody_came_through_the_door(entry.isoformat())
    vision.zone_motion = {"door": (entry - timedelta(seconds=20)).isoformat()}
    assert not runtime._nobody_came_through_the_door(entry.isoformat())
    vision.zone_motion = {"door": (entry - timedelta(minutes=10)).isoformat()}
    assert runtime._nobody_came_through_the_door(entry.isoformat())
    # In the dark the camera cannot see the door, so it has no say.
    vision.dark = True
    assert not runtime._nobody_came_through_the_door(entry.isoformat())


def test_unproven_liveness_never_blocks_the_owner(tmp_path) -> None:
    """Until it is measured on this camera, a low score is logged, not acted on."""
    worker = _worker(tmp_path)
    try:
        for _ in range(3):
            state = _see(worker, _face(OWNER, live=0.02))
        assert state.owner_visible
    finally:
        worker.stop()
