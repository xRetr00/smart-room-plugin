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
