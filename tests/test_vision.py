from __future__ import annotations

from dataclasses import asdict

from plugins.smart_room.runtime.models import RoomState, VisionState
from plugins.smart_room.runtime.vision import FaceLibrary, VisionWorker


class FakeAnalyzer:
    capabilities = {"faces": True, "gestures": True, "posture": True}
    face_model = "test-face-model"
    face_provider = "TestExecutionProvider"


def test_vision_state_round_trips_with_room_state() -> None:
    state = RoomState()
    state.vision = VisionState(
        enabled=True,
        running=True,
        camera_open=True,
        person_count=1,
        owner_visible=True,
        activity="moving",
    )

    restored = RoomState.from_dict(state.to_dict())

    assert restored.vision.camera_open is True
    assert restored.vision.owner_visible is True
    assert restored.vision.activity == "moving"
    assert restored.vision.face_model == "ArcFace R50 · SCRFD-10G"
    assert restored.vision.face_model_loaded is False


def test_face_library_matches_owner_and_folds_repeated_unknowns(tmp_path) -> None:
    library = FaceLibrary(tmp_path / "vision")
    try:
        library.enroll("Shereef", [[1.0, 0.0]], owner=True)
        assert library.match([1.0, 0.0])["status"] == "owner"
        first = library.record_sighting("unknown", "unknown", 0.1, None, [0.0, 1.0])
        second = library.record_sighting("unknown", "unknown", 0.1, None, [0.0, 1.0])
        assert first is not None
        assert second is None
        assert len(library.unreported_visitors()) == 1
    finally:
        library.close()


def test_approving_another_owner_sample_never_demotes_the_owner(tmp_path) -> None:
    library = FaceLibrary(tmp_path / "vision")
    try:
        library.enroll("Shereef", [[1.0, 0.0]], owner=True)
        sighting = library.record_sighting("unknown", "unknown", 0.2, None, [0.9, 0.1])

        result = library.approve(sighting, "Shereef", owner=False)

        assert result["owner"] is True
        assert library.owner_name() == "Shereef"
        assert library.people()[0]["owner"] is True
    finally:
        library.close()


def test_pending_faces_include_nearest_identity_and_reject_all_removes_crops(tmp_path) -> None:
    library = FaceLibrary(tmp_path / "vision", pending_similarity=0.99)
    crop = library.dir / "faces" / "candidate.jpg"
    crop.write_bytes(b"face")
    try:
        library.enroll("Shereef", [[1.0, 0.0]], owner=True)
        library.record_sighting("unknown", "unknown", 0.2, str(crop), [0.8, 0.2])

        pending = library.unreported_visitors()

        assert pending[0]["nearest"]["name"] == "Shereef"
        assert library.reject_all() == 1
        assert library.unreported_visitors() == []
        assert not crop.exists()
    finally:
        library.close()


def test_pending_limit_comes_from_configuration_and_cleans_evicted_crop(tmp_path) -> None:
    library = FaceLibrary(tmp_path / "vision", max_pending=1, pending_similarity=1.1)
    first = library.dir / "faces" / "first.jpg"
    second = library.dir / "faces" / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    try:
        library.record_sighting("unknown", "unknown", 0.1, str(first), [1.0, 0.0])
        library.record_sighting("unknown", "unknown", 0.1, str(second), [0.0, 1.0])

        assert len(library.unreported_visitors()) == 1
        assert not first.exists()
        assert second.exists()
    finally:
        library.close()


def test_pending_limit_is_enforced_when_existing_library_opens(tmp_path) -> None:
    directory = tmp_path / "vision"
    library = FaceLibrary(directory, max_pending=3, pending_similarity=1.1)
    crops = [directory / "faces" / f"candidate-{index}.jpg" for index in range(3)]
    for index, crop in enumerate(crops):
        crop.write_bytes(b"face")
        library.record_sighting("unknown", "unknown", 0.1, str(crop), [float(index), 1.0])
    library.close()

    reopened = FaceLibrary(directory, max_pending=1)
    try:
        assert len(reopened.unreported_visitors()) == 1
        assert not crops[0].exists()
        assert not crops[1].exists()
        assert crops[2].exists()
    finally:
        reopened.close()


def test_low_quality_faces_are_not_added_to_the_review_queue(tmp_path) -> None:
    worker = VisionWorker(
        {"enabled": True, "faces": {"min_face_size": 80}},
        lambda _state: None,
        lambda _kind, _data: None,
        library=FaceLibrary(tmp_path / "vision"),
        analyzer=FakeAnalyzer(),
    )
    try:
        worker._apply_analysis(
            None,
            {
                "faces": [
                    {
                        "embedding": [0.0, 1.0],
                        "bbox": [0.0, 0.0, 30.0, 30.0],
                        "detection_score": 0.99,
                    }
                ],
                "person_count": 1,
            },
            moving=False,
        )

        assert worker.library.unreported_visitors() == []
    finally:
        worker.stop()


def test_review_crop_preserves_context_and_stays_inside_the_frame() -> None:
    left, top, right, bottom = VisionWorker._review_crop_bounds(
        1280, 720, [500.0, 200.0, 605.0, 383.0]
    )

    assert 0 <= left < 500 < 605 < right <= 1280
    assert 0 <= top < 200 < 383 < bottom <= 720
    assert right - left > 105 * 3
    assert bottom - top > 183 * 2
    assert abs(((right - left) / (bottom - top)) - (4 / 3)) < 0.01


def test_review_crop_clamps_at_camera_edges() -> None:
    left, top, right, bottom = VisionWorker._review_crop_bounds(
        640, 480, [510.0, 330.0, 630.0, 470.0]
    )

    assert (right, bottom) == (640, 480)
    assert left >= 0
    assert top >= 0


def test_worker_publishes_bounded_facts_and_structured_gesture(tmp_path) -> None:
    published = []
    events = []
    worker = VisionWorker(
        {"enabled": True, "gesture_confidence": 0.6},
        published.append,
        lambda kind, data: events.append((kind, data)),
        library=FaceLibrary(tmp_path / "vision"),
        analyzer=FakeAnalyzer(),
    )
    try:
        worker._apply_analysis(
            None,
            {
                "faces": [],
                "person_count": 1,
                "gesture": "Open_Palm",
                "gesture_confidence": 0.9,
                "sleep_state": "awake",
                "capabilities": FakeAnalyzer.capabilities,
            },
            moving=True,
        )

        state = asdict(published[-1])
        assert state["person_count"] == 1
        assert state["activity"] == "moving"
        assert state["face_model"] == "test-face-model"
        assert state["face_model_loaded"] is True
        assert state["face_provider"] == "TestExecutionProvider"
        assert state["gesture"] == "Open_Palm"
        assert "frame" not in state
        # One frame's posture is not a posture: it takes STEADY_POSTURES in a row.
        assert [kind for kind, _ in events] == ["vision_gesture"]
        for _ in range(VisionWorker.STEADY_POSTURES):
            worker._apply_analysis(None, {"faces": [], "sleep_state": "awake"}, moving=False)
        assert [kind for kind, _ in events][-1] == "vision_sleep_state"
    finally:
        worker.stop()


def test_vision_description_never_returns_a_frame(tmp_path) -> None:
    worker = VisionWorker(
        {"enabled": True},
        lambda _state: None,
        lambda _kind, _data: None,
        library=FaceLibrary(tmp_path / "vision"),
        analyzer=FakeAnalyzer(),
    )
    try:
        worker._set_state(
            running=True,
            camera_open=True,
            stale=False,
            person_count=1,
            owner_visible=True,
            activity="still",
        )
        result = worker.describe()
        assert result["success"] is True
        assert "owner visible" in result["description"]
        assert "frame" not in result
    finally:
        worker.stop()


def test_nearly_the_owner_is_not_a_stranger(tmp_path) -> None:
    """The band between "not sure" and "confidently nobody known".

    Only the second is a visitor. The first is the owner looking down or half
    turned, and filing it as a stranger is what filled the visitor queue with
    his own face.
    """
    library = FaceLibrary(tmp_path / "vision")
    try:
        library.enroll("Shereef", [[1.0, 0.0]], owner=True)
        # cos = 0.37, 0.33 and 0.05 against the owner.
        assert library.match([0.37, 0.929])["status"] == "owner"
        assert library.match([0.33, 0.944])["status"] == "uncertain"
        assert library.match([0.05, 0.999])["status"] == "unknown"
    finally:
        library.close()


# Five points as insightface gives them: left eye, right eye, nose, mouth corners.
FACING = [[40.0, 40.0], [80.0, 40.0], [60.0, 60.0], [45.0, 80.0], [75.0, 80.0]]
PROFILE = [[40.0, 40.0], [80.0, 40.0], [86.0, 60.0], [45.0, 80.0], [75.0, 80.0]]
LOOKING_DOWN = [[40.0, 40.0], [80.0, 40.0], [60.0, 42.0], [45.0, 80.0], [75.0, 80.0]]


def test_only_a_face_pointed_at_the_camera_is_judged() -> None:
    assert VisionWorker._facing_camera({"landmarks": FACING})
    assert not VisionWorker._facing_camera({"landmarks": PROFILE})
    assert not VisionWorker._facing_camera({"landmarks": LOOKING_DOWN})
    # An analyzer that gives no landmarks is judged as before.
    assert VisionWorker._facing_camera({})


def test_a_turned_face_is_neither_a_visitor_nor_the_owner(tmp_path) -> None:
    """The owner's 14:47 "unknown visitor" was the owner, from behind.

    And the other direction matters as much: a stranger's profile must not be
    able to pass for the owner's.
    """
    seen = []
    worker = VisionWorker(
        {"enabled": True, "faces": {"min_face_size": 20, "min_blur_variance": 0}},
        lambda _state: None,
        lambda kind, data: seen.append(kind),
        library=FaceLibrary(tmp_path / "vision"),
        analyzer=FakeAnalyzer(),
    )
    try:
        worker.library.enroll("Shereef", [[1.0, 0.0]], owner=True)
        for embedding in ([0.0, 1.0], [1.0, 0.0]):
            worker._apply_analysis(
                None,
                {
                    "faces": [{
                        "embedding": embedding,
                        "bbox": [10.0, 10.0, 110.0, 110.0],
                        "detection_score": 0.99,
                        "landmarks": PROFILE,
                    }],
                    "person_count": 1,
                },
                moving=False,
            )
            assert worker.state.owner_visible is False

        assert worker.library.unreported_visitors() == []
        assert "vision_visitor_seen" not in seen
    finally:
        worker.stop()
