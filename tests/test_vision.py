from __future__ import annotations

from dataclasses import asdict

from plugins.smart_room.runtime.models import RoomState, VisionState
from plugins.smart_room.runtime.vision import FaceLibrary, VisionWorker


class FakeAnalyzer:
    capabilities = {"faces": True, "gestures": True, "posture": True}


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
        assert state["gesture"] == "Open_Palm"
        assert "frame" not in state
        assert [kind for kind, _ in events] == ["vision_gesture", "vision_sleep_state"]
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
