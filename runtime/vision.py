"""Single-owner, always-on local vision for the Smart Room sidecar.

The sidecar is the only process that opens the physical camera. Marvi receives
bounded facts and invokes authenticated RPC methods; frames and embeddings stay
inside this module.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
from typing import Any, Callable, Dict, Optional
from urllib.request import urlopen

from .models import VisionState
from .paths import vision_home, vision_models_home

logger = logging.getLogger(__name__)

OWNER_THRESHOLD = 0.42
KNOWN_THRESHOLD = 0.38
PENDING_SIMILARITY = 0.45
MAX_PENDING = 40
DETECT_SIZE = (640, 640)

GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    owner INTEGER NOT NULL DEFAULT 0,
    at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    vector TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sightings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    identity TEXT NOT NULL,
    status TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 0,
    thumbnail TEXT,
    reported INTEGER NOT NULL DEFAULT 0,
    vector TEXT
);
CREATE INDEX IF NOT EXISTS sightings_unreported ON sightings(reported, status);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    ln = sum(a * a for a in left) ** 0.5
    rn = sum(b * b for b in right) ** 0.5
    return 0.0 if ln == 0 or rn == 0 else dot / (ln * rn)


class FaceLibrary:
    """Sidecar-owned identity and visitor database."""

    def __init__(self, directory: Optional[Path] = None) -> None:
        self.dir = directory or vision_home()
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "faces").mkdir(exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.dir / "faces.sqlite3", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def enroll(self, name: str, embeddings: list[list[float]], owner: bool = False) -> Dict[str, Any]:
        clean = name.strip()[:80]
        if not clean:
            raise ValueError("a person needs a name")
        if not embeddings:
            raise ValueError("a person needs at least one face sample")
        with self._lock:
            if owner:
                self._db.execute("UPDATE people SET owner = 0")
            row = self._db.execute("SELECT id FROM people WHERE name = ?", (clean,)).fetchone()
            if row:
                person_id = int(row["id"])
                self._db.execute(
                    "UPDATE people SET owner = ? WHERE id = ?",
                    (1 if owner else 0, person_id),
                )
            else:
                cursor = self._db.execute(
                    "INSERT INTO people (name, owner, at) VALUES (?, ?, ?)",
                    (clean, 1 if owner else 0, _now_iso()),
                )
                person_id = int(cursor.lastrowid or 0)
            self._db.executemany(
                "INSERT INTO embeddings (person_id, vector) VALUES (?, ?)",
                [(person_id, json.dumps(list(map(float, item)))) for item in embeddings],
            )
            self._db.commit()
        return {"name": clean, "owner": owner, "samples": len(embeddings)}

    def people(self) -> list[Dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT p.name, p.owner, p.at, COUNT(e.id) AS samples FROM people p "
                "LEFT JOIN embeddings e ON e.person_id = p.id "
                "GROUP BY p.id ORDER BY p.owner DESC, p.name"
            ).fetchall()
        return [
            {"name": row["name"], "owner": bool(row["owner"]), "samples": int(row["samples"]), "at": row["at"]}
            for row in rows
        ]

    def owner_name(self) -> Optional[str]:
        with self._lock:
            row = self._db.execute("SELECT name FROM people WHERE owner = 1 LIMIT 1").fetchone()
        return str(row["name"]) if row else None

    def match(self, embedding: list[float]) -> Dict[str, Any]:
        with self._lock:
            rows = self._db.execute(
                "SELECT p.name, p.owner, e.vector FROM embeddings e "
                "JOIN people p ON p.id = e.person_id"
            ).fetchall()
        best_name, best_score, best_owner = "", 0.0, False
        for row in rows:
            score = cosine(embedding, json.loads(row["vector"]))
            if score > best_score:
                best_name, best_score, best_owner = str(row["name"]), score, bool(row["owner"])
        if best_owner and best_score >= OWNER_THRESHOLD:
            return {"identity": best_name, "status": "owner", "score": round(best_score, 4)}
        if best_name and best_score >= KNOWN_THRESHOLD:
            return {"identity": best_name, "status": "known", "score": round(best_score, 4)}
        return {"identity": "unknown", "status": "unknown", "score": round(best_score, 4)}

    def record_sighting(
        self,
        identity: str,
        status: str,
        score: float,
        thumbnail: Optional[str],
        embedding: Optional[list[float]],
    ) -> Optional[int]:
        with self._lock:
            if status == "unknown" and embedding is not None:
                rows = self._db.execute(
                    "SELECT vector FROM sightings WHERE status = 'unknown' AND reported = 0 "
                    "AND vector IS NOT NULL ORDER BY id DESC LIMIT ?",
                    (MAX_PENDING,),
                ).fetchall()
                if any(cosine(embedding, json.loads(row["vector"])) >= PENDING_SIMILARITY for row in rows):
                    return None
            cursor = self._db.execute(
                "INSERT INTO sightings (at, identity, status, score, thumbnail, reported, vector) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    _now_iso(), identity, status, float(score), thumbnail,
                    0 if status == "unknown" else 1,
                    json.dumps(embedding) if embedding is not None else None,
                ),
            )
            self._db.execute(
                "DELETE FROM sightings WHERE status = 'unknown' AND reported = 0 AND id NOT IN "
                "(SELECT id FROM sightings WHERE status = 'unknown' AND reported = 0 "
                "ORDER BY id DESC LIMIT ?)",
                (MAX_PENDING,),
            )
            self._db.commit()
            return int(cursor.lastrowid or 0)

    def unreported_visitors(self) -> list[Dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, at, identity, score, thumbnail FROM sightings "
                "WHERE status = 'unknown' AND reported = 0 ORDER BY id"
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_sightings(self, limit: int = 30) -> list[Dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, at, identity, status, score, thumbnail FROM sightings "
                "ORDER BY id DESC LIMIT ?",
                (max(1, min(limit, 200)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_reported(self, ids: list[int]) -> int:
        if not ids:
            return 0
        with self._lock:
            self._db.executemany(
                "UPDATE sightings SET reported = 1 WHERE id = ?",
                [(int(identifier),) for identifier in ids],
            )
            self._db.commit()
        return len(ids)

    def approve(self, sighting_id: int, name: str, owner: bool = False) -> Dict[str, Any]:
        with self._lock:
            row = self._db.execute(
                "SELECT vector FROM sightings WHERE id = ?", (sighting_id,)
            ).fetchone()
        if row is None or not row["vector"]:
            raise ValueError(f"no stored face for sighting {sighting_id}")
        result = self.enroll(name, [json.loads(row["vector"])], owner=owner)
        with self._lock:
            self._db.execute(
                "UPDATE sightings SET identity = ?, status = ?, reported = 1 WHERE id = ?",
                (name.strip()[:80], "owner" if owner else "known", sighting_id),
            )
            self._db.commit()
        return result

    def reject(self, sighting_id: int) -> bool:
        with self._lock:
            cursor = self._db.execute("DELETE FROM sightings WHERE id = ?", (sighting_id,))
            self._db.commit()
            return cursor.rowcount > 0


def _download(url: str, destination: Path) -> Path:
    if destination.is_file() and destination.stat().st_size > 100_000:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        with urlopen(url, timeout=60) as response:
            while chunk := response.read(1024 * 1024):
                temporary.write(chunk)
    if temporary_path.stat().st_size <= 100_000:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded model is unexpectedly small: {destination.name}")
    temporary_path.replace(destination)
    return destination


def ensure_task_models() -> Dict[str, str]:
    root = vision_models_home()
    return {
        "gesture": str(_download(GESTURE_MODEL_URL, root / "gesture_recognizer.task")),
        "pose": str(_download(POSE_MODEL_URL, root / "pose_landmarker_lite.task")),
    }


class LocalVisionAnalyzer:
    """InsightFace plus optional official MediaPipe Tasks models."""

    def __init__(self, *, auto_download: bool = True) -> None:
        self.auto_download = auto_download
        self._face: Any = None
        self._gesture: Any = None
        self._pose: Any = None
        self.capabilities = {"faces": False, "gestures": False, "posture": False}

    def load(self) -> None:
        from insightface.app import FaceAnalysis

        face = FaceAnalysis(
            name="buffalo_l",
            root=str(vision_models_home()),
            providers=["CPUExecutionProvider"],
        )
        face.prepare(ctx_id=-1, det_size=DETECT_SIZE)
        self._face = face
        self.capabilities["faces"] = True

        try:
            if self.auto_download:
                models = ensure_task_models()
            else:
                models = {
                    "gesture": str(vision_models_home() / "gesture_recognizer.task"),
                    "pose": str(vision_models_home() / "pose_landmarker_lite.task"),
                }
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision

            self._gesture = mp_vision.GestureRecognizer.create_from_options(
                mp_vision.GestureRecognizerOptions(
                    base_options=mp_tasks.BaseOptions(model_asset_path=models["gesture"]),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_hands=2,
                )
            )
            self.capabilities["gestures"] = True
            self._pose = mp_vision.PoseLandmarker.create_from_options(
                mp_vision.PoseLandmarkerOptions(
                    base_options=mp_tasks.BaseOptions(model_asset_path=models["pose"]),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_poses=2,
                )
            )
            self.capabilities["posture"] = True
        except Exception:
            logger.warning("MediaPipe gesture/posture models unavailable", exc_info=True)

    def analyze(self, frame: Any) -> Dict[str, Any]:
        if self._face is None:
            self.load()
        faces = [
            {
                "embedding": [float(value) for value in face.normed_embedding],
                "bbox": [float(value) for value in face.bbox],
            }
            for face in self._face.get(frame)
        ]
        result: Dict[str, Any] = {
            "faces": faces,
            "person_count": len(faces),
            "gesture": None,
            "gesture_confidence": 0.0,
            "sleep_state": "unknown",
            "capabilities": dict(self.capabilities),
        }
        if self._gesture is None and self._pose is None:
            return result
        try:
            import cv2
            import mediapipe as mp

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            if self._gesture is not None:
                recognised = self._gesture.recognize(image)
                candidates = [items[0] for items in recognised.gestures if items]
                if candidates:
                    best = max(candidates, key=lambda item: float(item.score))
                    result["gesture"] = str(best.category_name)
                    result["gesture_confidence"] = round(float(best.score), 4)
            if self._pose is not None:
                pose = self._pose.detect(image)
                result["person_count"] = max(result["person_count"], len(pose.pose_landmarks))
                result["sleep_state"] = self._posture(pose.pose_landmarks)
        except Exception:
            logger.debug("MediaPipe frame analysis failed", exc_info=True)
        return result

    @staticmethod
    def _posture(poses: list[Any]) -> str:
        if not poses:
            return "unknown"
        # Shoulder midpoint to hip midpoint is normally vertical when upright.
        # A mostly horizontal torso is reported as resting, never definitively
        # asleep; room mode and sustained stillness decide how Marvi phrases it.
        pose = poses[0]
        shoulder_x = (pose[11].x + pose[12].x) / 2
        shoulder_y = (pose[11].y + pose[12].y) / 2
        hip_x = (pose[23].x + pose[24].x) / 2
        hip_y = (pose[23].y + pose[24].y) / 2
        horizontal = abs(hip_x - shoulder_x) > abs(hip_y - shoulder_y) * 1.2
        return "resting" if horizontal else "awake"


class VisionWorker:
    """Continuously owns one camera and publishes structured observations."""

    def __init__(
        self,
        config: Dict[str, Any],
        publish_state: Callable[[VisionState], None],
        emit_event: Callable[[str, Dict[str, Any]], None],
        *,
        library: Optional[FaceLibrary] = None,
        analyzer: Any = None,
        capture_factory: Optional[Callable[[int], Any]] = None,
    ) -> None:
        self.config = config
        self.publish_state = publish_state
        self.emit_event = emit_event
        self.library = library or FaceLibrary()
        self.analyzer = analyzer or LocalVisionAnalyzer(
            auto_download=bool(config.get("auto_download_models", True))
        )
        self.capture_factory = capture_factory
        self.state = VisionState(
            enabled=bool(config.get("enabled", False)),
            camera_index=int(config.get("camera_index", 0)),
        )
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Any = None
        self._latest_embeddings: list[list[float]] = []
        self._last_gesture: Optional[str] = None
        self._last_gesture_at = 0.0

    def start(self) -> None:
        if not self.state.enabled or (self._thread and self._thread.is_alive()):
            self.publish_state(self.snapshot_state())
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="smart_room_vision", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)
        if self._thread and self._thread.is_alive():
            logger.warning("Vision worker did not stop before shutdown timeout")
            return
        self._thread = None
        self.library.close()

    def snapshot_state(self) -> VisionState:
        with self._lock:
            return VisionState(**asdict(self.state))

    def status(self) -> Dict[str, Any]:
        state = asdict(self.snapshot_state())
        state["people"] = self.library.people()
        state["pending_visitors"] = len(self.library.unreported_visitors())
        return state

    def observe(self) -> Dict[str, Any]:
        return {"success": True, "vision": asdict(self.snapshot_state())}

    def describe(self) -> Dict[str, Any]:
        state = self.snapshot_state()
        if not state.camera_open:
            return {"success": False, "error": state.error or "camera is not available", "vision": asdict(state)}
        people = "no people" if state.person_count == 0 else f"{state.person_count} person" + ("" if state.person_count == 1 else "s")
        identity = "owner visible" if state.owner_visible else "owner not visible"
        parts = [people, identity, state.activity]
        if state.gesture:
            parts.append(f"gesture {state.gesture}")
        if state.sleep_state != "unknown":
            parts.append(state.sleep_state)
        return {"success": True, "description": ", ".join(parts), "vision": asdict(state)}

    def people(self) -> Dict[str, Any]:
        return {"success": True, "people": self.library.people(), "owner": self.library.owner_name()}

    def visitors(self) -> Dict[str, Any]:
        return {"success": True, "visitors": self.library.unreported_visitors()}

    def mark_visitors_reported(self, ids: list[int]) -> int:
        return self.library.mark_reported(ids)

    def approve(self, sighting_id: int, name: str, owner: bool = False) -> Dict[str, Any]:
        return {"success": True, **self.library.approve(sighting_id, name, owner=owner)}

    def reject(self, sighting_id: int) -> Dict[str, Any]:
        return {"success": self.library.reject(sighting_id), "sighting_id": sighting_id}

    def enroll_owner(self, name: str, seconds: float = 4.0) -> Dict[str, Any]:
        deadline = time.monotonic() + max(1.0, min(float(seconds), 20.0))
        samples: list[list[float]] = []
        while time.monotonic() < deadline and len(samples) < 8:
            with self._lock:
                visible = [list(item) for item in self._latest_embeddings]
            if len(visible) == 1:
                samples.append(visible[0])
            self._stop.wait(0.25)
        if not samples:
            return {"success": False, "error": "no single clear face is visible"}
        return {"success": True, **self.library.enroll(name, samples, owner=True)}

    def _open_capture(self) -> Any:
        if self.capture_factory:
            return self.capture_factory(self.state.camera_index)
        import cv2

        return cv2.VideoCapture(self.state.camera_index, cv2.CAP_DSHOW)

    def _run(self) -> None:
        retry_seconds = max(1.0, float(self.config.get("reconnect_seconds", 3)))
        while not self._stop.is_set():
            capture = None
            try:
                capture = self._open_capture()
                if not capture.isOpened():
                    raise RuntimeError(f"camera {self.state.camera_index} unavailable")
                self._set_state(running=True, camera_open=True, stale=False, error=None)
                self._capture_loop(capture)
            except Exception as exc:
                logger.warning("Vision capture unavailable: %s", exc)
                self._set_state(running=True, camera_open=False, stale=True, error=str(exc)[:200])
                self._stop.wait(retry_seconds)
            finally:
                if capture is not None:
                    try:
                        capture.release()
                    except Exception:
                        pass
        self._set_state(running=False, camera_open=False, stale=True)

    def _capture_loop(self, capture: Any) -> None:
        import cv2
        import numpy as np

        interval = 1.0 / max(0.2, min(float(self.config.get("inference_fps", 1.0)), 8.0))
        motion_threshold = max(0.0, float(self.config.get("motion_threshold", 6.0)))
        last_inference = 0.0
        previous = None
        while not self._stop.is_set():
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("camera stopped returning frames")
            now = time.monotonic()
            with self._lock:
                self._latest_frame = frame
                self.state.last_frame_at = _now_iso()
                self.state.stale = False
            if now - last_inference < interval:
                continue
            grey = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY)
            motion = 100.0 if previous is None else float(
                np.mean(np.abs(grey.astype("int16") - previous.astype("int16")))
            )
            previous = grey
            last_inference = now
            analysis = self.analyzer.analyze(frame)
            self._apply_analysis(frame, analysis, moving=motion >= motion_threshold)

    def _thumbnail(self, frame: Any, box: list[float]) -> Optional[str]:
        try:
            import cv2

            x1, y1, x2, y2 = (max(0, int(value)) for value in box[:4])
            pad = int(0.25 * max(1, x2 - x1))
            crop = frame[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
            if crop.size == 0:
                return None
            path = self.library.dir / "faces" / f"{time.time_ns()}.jpg"
            return str(path) if cv2.imwrite(str(path), crop) else None
        except Exception:
            return None

    def _apply_analysis(self, frame: Any, analysis: Dict[str, Any], *, moving: bool) -> None:
        identities: list[str] = []
        owner_visible = False
        owner_confidence = 0.0
        embeddings: list[list[float]] = []
        for face in analysis.get("faces") or []:
            embedding = [float(value) for value in face.get("embedding") or []]
            if not embedding:
                continue
            embeddings.append(embedding)
            verdict = self.library.match(embedding)
            identities.append(str(verdict["identity"]))
            if verdict["status"] == "owner":
                owner_visible = True
                owner_confidence = max(owner_confidence, float(verdict["score"]))
            # Known identities are live state, not a surveillance history.
            # Persist only unknown visitors, and create a thumbnail only after
            # that decision, avoiding a JPEG and database row every second for
            # the owner sitting at their desk.
            sighting_id = None
            if verdict["status"] == "unknown":
                sighting_id = self.library.record_sighting(
                    str(verdict["identity"]),
                    str(verdict["status"]),
                    float(verdict["score"]),
                    self._thumbnail(frame, list(face.get("bbox") or [])),
                    embedding,
                )
            if sighting_id is not None:
                self.emit_event(
                    "vision_visitor_seen",
                    {"sighting_id": sighting_id, "summary": "Unknown visitor seen by Smart Room"},
                )

        gesture = analysis.get("gesture")
        gesture_confidence = float(analysis.get("gesture_confidence") or 0.0)
        gesture_threshold = float(self.config.get("gesture_confidence", 0.65))
        if gesture_confidence < gesture_threshold:
            gesture = None
            gesture_confidence = 0.0
        now = time.monotonic()
        if gesture and (gesture != self._last_gesture or now - self._last_gesture_at >= 3.0):
            self._last_gesture = str(gesture)
            self._last_gesture_at = now
            commands = self.config.get("gesture_commands") or {}
            command = commands.get(gesture)
            self.emit_event(
                "vision_gesture",
                {
                    "gesture": gesture,
                    "confidence": gesture_confidence,
                    "command": command,
                    "summary": f"Gesture: {str(gesture).replace('_', ' ')}",
                },
            )

        sleep_state = str(analysis.get("sleep_state") or "unknown")
        person_count = max(int(analysis.get("person_count") or 0), len(embeddings))
        with self._lock:
            previous_sleep = self.state.sleep_state
            self._latest_embeddings = embeddings
            self.state.last_inference_at = _now_iso()
            self.state.person_count = person_count
            self.state.owner_visible = owner_visible
            self.state.owner_confidence = round(owner_confidence, 4)
            self.state.identities = identities[:8]
            self.state.pending_visitors = len(self.library.unreported_visitors())
            self.state.activity = "moving" if moving else "still"
            self.state.gesture = str(gesture) if gesture else None
            self.state.gesture_confidence = round(gesture_confidence, 4)
            self.state.sleep_state = sleep_state
            self.state.capabilities = dict(analysis.get("capabilities") or self.analyzer.capabilities)
            snapshot = VisionState(**asdict(self.state))
        self.publish_state(snapshot)
        if sleep_state != previous_sleep and sleep_state in {"awake", "resting"}:
            self.emit_event(
                "vision_sleep_state",
                {"sleep_state": sleep_state, "summary": f"Vision posture: {sleep_state}"},
            )

    def _set_state(self, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(self.state, key, value)
            snapshot = VisionState(**asdict(self.state))
        self.publish_state(snapshot)
