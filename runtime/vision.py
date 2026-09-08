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
import contextlib
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


#: How long to wait between frames while standing aside. The camera stays
#: open and the loop keeps the pipeline drained; it just stops taking every
#: frame the device offers.
EASY_FRAME_GAP = 0.5

#: And how long between analyses. One a second is right for a room being
#: watched; one every twenty seconds is enough to keep knowing somebody is
#: there, which is all that is needed while they are busy.
EASY_INFERENCE_GAP = 20.0


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

    def __init__(
        self,
        directory: Optional[Path] = None,
        *,
        owner_threshold: float = OWNER_THRESHOLD,
        known_threshold: float = KNOWN_THRESHOLD,
        pending_similarity: float = PENDING_SIMILARITY,
        max_pending: int = MAX_PENDING,
    ) -> None:
        self.dir = directory or vision_home()
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "faces").mkdir(exist_ok=True)
        self.owner_threshold = float(owner_threshold)
        self.known_threshold = float(known_threshold)
        self.pending_similarity = float(pending_similarity)
        self.max_pending = max(1, min(int(max_pending), 200))
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.dir / "faces.sqlite3", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._db.commit()
        self._enforce_pending_limit()
        self._cleanup_orphan_thumbnails()

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
            row = self._db.execute("SELECT id, owner FROM people WHERE name = ?", (clean,)).fetchone()
            if row:
                person_id = int(row["id"])
                # Adding an ordinary sample to the owner's existing identity
                # must never demote them. Ownership changes only through an
                # explicit owner action.
                if owner:
                    self._db.execute("UPDATE people SET owner = 1 WHERE id = ?", (person_id,))
                is_owner = owner or bool(row["owner"])
            else:
                cursor = self._db.execute(
                    "INSERT INTO people (name, owner, at) VALUES (?, ?, ?)",
                    (clean, 1 if owner else 0, _now_iso()),
                )
                person_id = int(cursor.lastrowid or 0)
                is_owner = owner
            self._db.executemany(
                "INSERT INTO embeddings (person_id, vector) VALUES (?, ?)",
                [(person_id, json.dumps(list(map(float, item)))) for item in embeddings],
            )
            self._db.commit()
        return {"name": clean, "owner": is_owner, "samples": len(embeddings)}

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

    def set_owner(self, name: str) -> bool:
        clean = name.strip()[:80]
        if not clean:
            return False
        with self._lock:
            row = self._db.execute(
                "SELECT id FROM people WHERE name = ? COLLATE NOCASE", (clean,)
            ).fetchone()
            if row is None:
                return False
            self._db.execute("UPDATE people SET owner = 0")
            self._db.execute("UPDATE people SET owner = 1 WHERE id = ?", (int(row["id"]),))
            self._db.commit()
        return True

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
        nearest = {"name": best_name, "score": round(best_score, 4)} if best_name else {}
        if best_owner and best_score >= self.owner_threshold:
            return {"identity": best_name, "status": "owner", "score": round(best_score, 4), "nearest": nearest}
        if best_name and best_score >= self.known_threshold:
            return {"identity": best_name, "status": "known", "score": round(best_score, 4), "nearest": nearest}
        return {"identity": "unknown", "status": "unknown", "score": round(best_score, 4), "nearest": nearest}

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
                    (self.max_pending,),
                ).fetchall()
                if any(cosine(embedding, json.loads(row["vector"])) >= self.pending_similarity for row in rows):
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
            removed = self._db.execute(
                "SELECT thumbnail FROM sightings WHERE status = 'unknown' AND reported = 0 "
                "AND id NOT IN (SELECT id FROM sightings WHERE status = 'unknown' AND reported = 0 "
                "ORDER BY id DESC LIMIT ?)",
                (self.max_pending,),
            ).fetchall()
            self._db.execute(
                "DELETE FROM sightings WHERE status = 'unknown' AND reported = 0 AND id NOT IN "
                "(SELECT id FROM sightings WHERE status = 'unknown' AND reported = 0 "
                "ORDER BY id DESC LIMIT ?)",
                (self.max_pending,),
            )
            self._db.commit()
            sighting_id = int(cursor.lastrowid or 0)
        self._remove_thumbnails([row["thumbnail"] for row in removed])
        return sighting_id

    def unreported_visitors(self) -> list[Dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, at, identity, score, thumbnail, vector FROM sightings "
                "WHERE status = 'unknown' AND reported = 0 ORDER BY id"
            ).fetchall()
        visitors = []
        for row in rows:
            item = dict(row)
            vector = item.pop("vector", None)
            nearest: Dict[str, Any] = {}
            if vector:
                try:
                    nearest = self.match(json.loads(vector)).get("nearest") or {}
                except (TypeError, ValueError, json.JSONDecodeError):
                    nearest = {}
            item["nearest"] = nearest
            visitors.append(item)
        return visitors

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
                (name.strip()[:80], "owner" if result["owner"] else "known", sighting_id),
            )
            self._db.commit()
        return result

    def reject(self, sighting_id: int) -> bool:
        with self._lock:
            row = self._db.execute("SELECT thumbnail FROM sightings WHERE id = ?", (sighting_id,)).fetchone()
            cursor = self._db.execute("DELETE FROM sightings WHERE id = ?", (sighting_id,))
            self._db.commit()
            removed = cursor.rowcount > 0
        if removed and row is not None:
            self._remove_thumbnails([row["thumbnail"]])
        return removed

    def reject_all(self) -> int:
        with self._lock:
            rows = self._db.execute(
                "SELECT thumbnail FROM sightings WHERE status = 'unknown' AND reported = 0"
            ).fetchall()
            cursor = self._db.execute(
                "DELETE FROM sightings WHERE status = 'unknown' AND reported = 0"
            )
            self._db.commit()
            removed = max(0, int(cursor.rowcount))
        self._remove_thumbnails([row["thumbnail"] for row in rows])
        return removed

    def _remove_thumbnails(self, paths: list[Any]) -> None:
        faces_dir = (self.dir / "faces").resolve()
        for value in paths:
            if not value:
                continue
            path = Path(str(value))
            try:
                resolved = path.resolve()
                if resolved.parent == faces_dir:
                    resolved.unlink(missing_ok=True)
            except OSError:
                logger.debug("Could not remove face thumbnail %s", path, exc_info=True)

    def _enforce_pending_limit(self) -> None:
        with self._lock:
            removed = self._db.execute(
                "SELECT thumbnail FROM sightings WHERE status = 'unknown' AND reported = 0 "
                "AND id NOT IN (SELECT id FROM sightings WHERE status = 'unknown' AND reported = 0 "
                "ORDER BY id DESC LIMIT ?)",
                (self.max_pending,),
            ).fetchall()
            self._db.execute(
                "DELETE FROM sightings WHERE status = 'unknown' AND reported = 0 AND id NOT IN "
                "(SELECT id FROM sightings WHERE status = 'unknown' AND reported = 0 "
                "ORDER BY id DESC LIMIT ?)",
                (self.max_pending,),
            )
            self._db.commit()
        self._remove_thumbnails([row["thumbnail"] for row in removed])

    def _cleanup_orphan_thumbnails(self) -> None:
        with self._lock:
            referenced = {
                str(Path(str(row["thumbnail"])).resolve())
                for row in self._db.execute(
                    "SELECT thumbnail FROM sightings WHERE thumbnail IS NOT NULL"
                ).fetchall()
                if row["thumbnail"]
            }
        for path in (self.dir / "faces").glob("*.jpg"):
            if str(path.resolve()) not in referenced:
                try:
                    path.unlink()
                except OSError:
                    logger.debug("Could not remove orphan face thumbnail %s", path, exc_info=True)


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
        self.face_model = "buffalo_l"
        self.face_provider = "CPUExecutionProvider"
        self._face: Any = None
        self._gesture: Any = None
        self._pose: Any = None
        self.capabilities = {"faces": False, "gestures": False, "posture": False}

    def load(self) -> None:
        from insightface.app import FaceAnalysis

        face = FaceAnalysis(
            name=self.face_model,
            root=str(vision_models_home()),
            providers=[self.face_provider],
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
                "detection_score": float(face.det_score),
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
        """Upright, horizontal, or not visible. One frame's opinion only.

        The caller must not act on a single frame of this. Pose landmarks are
        noisy and the threshold below is a ratio, so somebody sitting at the
        edge of it flips between answers every inference -- and every flip was
        a state change, and every state change was an event.
        `vision_sleep_state` reached 626 of the mind's events and 94% of
        everything the room sent, which is that flapping and nothing else.
        See `_settled_posture`.
        """
        if not poses:
            # Not visible is not asleep. It is not any posture at all -- the
            # room is empty, or they are out of frame, or the light is off.
            return "unknown"
        # Shoulder midpoint to hip midpoint is normally vertical when upright.
        # A mostly horizontal torso is reported as resting, never as asleep:
        # lying down is a posture and sleeping is a conclusion, and the camera
        # can only see the first.
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
        face_config = config.get("faces") if isinstance(config.get("faces"), dict) else {}
        match_threshold = float(face_config.get("match_threshold", KNOWN_THRESHOLD))
        self.library = library or FaceLibrary(
            owner_threshold=float(face_config.get("owner_match_threshold", match_threshold)),
            known_threshold=match_threshold,
            pending_similarity=float(face_config.get("pending_similarity", PENDING_SIMILARITY)),
            max_pending=int(face_config.get("max_pending", MAX_PENDING)),
        )
        self.analyzer = analyzer or LocalVisionAnalyzer(
            auto_download=bool(config.get("auto_download_models", True))
        )
        self.capture_factory = capture_factory
        self.state = VisionState(
            enabled=bool(config.get("enabled", False)),
            camera_index=int(config.get("camera_index", 0)),
            face_model=str(getattr(self.analyzer, "face_model", "buffalo_l")),
            face_provider=str(
                getattr(self.analyzer, "face_provider", "CPUExecutionProvider")
            ),
            face_model_loaded=bool(
                (getattr(self.analyzer, "capabilities", {}) or {}).get("faces", False)
            ),
        )
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Any = None
        #: The open camera, so a photograph can ask it for a bigger frame.
        self._capture: Any = None
        self._latest_embeddings: list[list[float]] = []
        #: How many inferences in a row agreed, and what they agreed on.
        #: See `_settled_posture`.
        self._posture_seen: str = "unknown"
        self._posture_runs: int = 0
        self._last_gesture: Optional[str] = None
        self._last_gesture_at = 0.0
        #: Set while something else should have the machine. See `pace`.
        self._easy = threading.Event()

    def pace(self, easy: bool) -> None:
        """Slow right down while something else needs the machine, or resume.

        The camera loop is the most expensive thing in this process and the
        least interruptible: `capture.read()` runs with no delay at all, so it
        pulls and copies every frame the device produces, and MediaPipe runs
        face detection and pose landmarks on top of that once a second. That is
        the correct cost when the room is what matters. It is the wrong cost
        during a match, where the same core is drawing a stadium.

        Slowed rather than stopped, deliberately. The camera stays open --
        reopening a DSHOW device costs seconds and sometimes fails outright --
        and presence keeps updating, just at a pace that suits a game. She
        should still know you are in the room while you are playing; she does
        not need to know it thirty times a second.
        """
        if easy == self._easy.is_set():
            return
        if easy:
            self._easy.set()
        else:
            self._easy.clear()
        with self._lock:
            self.state.low_power = easy
        logger.info("Vision %s", "standing down; something else needs the machine"
                    if easy else "back to its normal pace")
        self.publish_state(self.snapshot_state())

    #: How many inferences in a row must agree before the posture changes.
    #:
    #: Three, at one inference a second -- so a real change is reported within
    #: about three seconds, and a landmark jittering across the threshold is
    #: reported not at all. Without it every jitter was a state change and
    #: every state change was an event: `vision_sleep_state` was 94% of
    #: everything the room ever sent.
    STEADY_POSTURES = 3

    def _settled_posture(self, seen: str) -> str:
        """The posture, once the camera has stopped changing its mind.

        Returns the currently held posture until a new one has been seen
        `STEADY_POSTURES` times running, so a single odd frame changes nothing.
        """
        if seen == self._posture_seen:
            self._posture_runs = 0
            return self._posture_seen
        self._posture_runs += 1
        if self._posture_runs < self.STEADY_POSTURES:
            return self._posture_seen
        self._posture_seen, self._posture_runs = seen, 0
        return seen

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

    #: How many photographs a visitor burst takes, and how far apart.
    #:
    #: One is not enough and the reason is mundane: a single frame at the
    #: moment somebody walks in is very often the back of their head. Three,
    #: ten seconds apart, covers turning round, sitting down, and looking up --
    #: and thirty seconds is short enough that they are still in the room.
    BURST_PHOTOS = 3
    BURST_GAP_SECONDS = 10.0

    def photograph(self, count: int = 0, gap: float = 0.0) -> Dict[str, Any]:
        """Take a burst of whole-frame photographs, for a visitor report.

        Whole frames rather than face crops: a crop answers "who" and this has
        to answer "what happened", which needs the room in it. Each is stamped
        with the wall-clock time it was taken, because the answer to "when was
        somebody in my room" is the entire point of the report and a file
        modification time is not an answer -- it changes when the file is
        copied.
        """
        import cv2

        count = int(count or self.BURST_PHOTOS)
        gap = float(gap or self.BURST_GAP_SECONDS)
        folder = self.library.dir / "visits"
        folder.mkdir(parents=True, exist_ok=True)
        taken: list[Dict[str, Any]] = []

        # Turned up for the burst, and put back after.
        #
        # The stream is small on purpose -- face recognition has to keep up
        # with it -- and a photograph has the opposite requirement: it is
        # looked at once, by a person, who needs to be able to say who that
        # was. Reading `_latest_frame` gave the recognition-sized frame, which
        # is exactly the wrong one.
        capture = self._capture
        restore: tuple[int, int] | None = None
        if capture is not None:
            got = self._resize_capture(capture, self._photo_size())
            restore = self._stream_size()
            logger.info("Photographing at %dx%d", got[0], got[1])

        for index in range(max(1, count)):
            if index:
                self._stop.wait(gap)
                if self._stop.is_set():
                    break
            frame = None
            if capture is not None:
                with contextlib.suppress(Exception):
                    ok, grabbed = capture.read()
                    frame = grabbed if ok else None
            if frame is None:
                # No camera handle, or the read failed. The recognition frame
                # is worse than a full one and much better than nothing.
                with self._lock:
                    frame = None if self._latest_frame is None else self._latest_frame.copy()
            if frame is None:
                continue
            at = datetime.now(timezone.utc)
            path = folder / f"{at.strftime('%Y%m%dT%H%M%SZ')}-{index + 1}.jpg"
            try:
                written = cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            except Exception:
                written = False
            if written:
                height, width = frame.shape[:2]
                taken.append({
                    # ISO, from the clock, at the moment the frame was taken.
                    "at": at.isoformat(),
                    "path": str(path),
                    "index": index + 1,
                    "width": int(width),
                    "height": int(height),
                })
        if capture is not None and restore is not None:
            # Back to the size recognition expects, whatever happened above.
            self._resize_capture(capture, restore)
        return {"success": bool(taken), "photos": taken, "count": len(taken)}

    def visitors(self) -> Dict[str, Any]:
        return {"success": True, "visitors": self.library.unreported_visitors()}

    def mark_visitors_reported(self, ids: list[int]) -> int:
        return self.library.mark_reported(ids)

    def approve(self, sighting_id: int, name: str, owner: bool = False) -> Dict[str, Any]:
        return {"success": True, **self.library.approve(sighting_id, name, owner=owner)}

    def reject(self, sighting_id: int) -> Dict[str, Any]:
        return {"success": self.library.reject(sighting_id), "sighting_id": sighting_id}

    def reject_all(self) -> Dict[str, Any]:
        return {"success": True, "rejected": self.library.reject_all()}

    def set_owner(self, name: str) -> Dict[str, Any]:
        changed = self.library.set_owner(name)
        return {"success": changed, "name": name, **({} if changed else {"error": f"unknown person: {name}"})}

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

        capture = cv2.VideoCapture(self.state.camera_index, cv2.CAP_DSHOW)
        # The configured size, actually applied.
        #
        # `width` and `height` have been in the config since it was written and
        # nothing ever set them, so the stream ran at whatever the driver
        # defaulted to -- usually 640x480 on DSHOW. Everything downstream was
        # tuned against that resolution without anybody choosing it.
        self._resize_capture(capture, self._stream_size())
        return capture

    def _stream_size(self) -> tuple[int, int]:
        """What the running stream should be: small enough to stay fast."""
        return (
            max(160, int(self.config.get("width", 1280) or 1280)),
            max(120, int(self.config.get("height", 720) or 720)),
        )

    def _photo_size(self) -> tuple[int, int]:
        """And what a photograph should be: as much as the camera will give.

        Face recognition runs on a deliberately small frame because it has to
        keep up. A photograph is looked at by a person afterwards, once, and
        the whole point of it is being able to say who that was -- so it gets
        the sensor's full resolution even though the stream does not.
        """
        return (
            max(640, int(self.config.get("photo_width", 1920) or 1920)),
            max(480, int(self.config.get("photo_height", 1080) or 1080)),
        )

    @staticmethod
    def _resize_capture(capture: Any, size: tuple[int, int]) -> tuple[int, int]:
        """Ask the camera for a size. Returns what it actually gave."""
        import cv2

        with contextlib.suppress(Exception):
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        try:
            return (
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        except Exception:
            return size

    def _run(self) -> None:
        retry_seconds = max(1.0, float(self.config.get("reconnect_seconds", 3)))
        while not self._stop.is_set():
            capture = None
            try:
                capture = self._open_capture()
                if not capture.isOpened():
                    raise RuntimeError(f"camera {self.state.camera_index} unavailable")
                # Held so a photograph can ask the same camera for a bigger
                # frame; there is only one device and DSHOW will not open it
                # twice.
                self._capture = capture
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

        normal = 1.0 / max(0.2, min(float(self.config.get("inference_fps", 1.0)), 8.0))
        motion_threshold = max(0.0, float(self.config.get("motion_threshold", 6.0)))
        last_inference = 0.0
        previous = None
        while not self._stop.is_set():
            easy = self._easy.is_set()
            if easy:
                # Between frames rather than spinning on the device. Waiting on
                # `_stop` rather than sleeping so a shutdown is still prompt.
                self._stop.wait(EASY_FRAME_GAP)
                if self._stop.is_set():
                    return
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("camera stopped returning frames")
            now = time.monotonic()
            interval = EASY_INFERENCE_GAP if easy else normal
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

    @staticmethod
    def _review_crop_bounds(
        frame_width: int, frame_height: int, box: list[float]
    ) -> tuple[int, int, int, int]:
        """Return a bounded 4:3 context crop around a detected face."""
        x1, y1, x2, y2 = (max(0, int(value)) for value in box[:4])
        face_width = max(1, min(frame_width, x2) - min(frame_width, x1))
        face_height = max(1, min(frame_height, y2) - min(frame_height, y1))
        max_width = max(1, min(frame_width, 960))
        max_height = max(1, min(frame_height, 720))
        target_height = max(face_height * 2.5, face_width * 2.25, 240.0)
        scale = min(1.0, max_width / (target_height * 4.0 / 3.0), max_height / target_height)
        crop_height = max(face_height, min(max_height, int(round(target_height * scale))))
        crop_width = max(face_width, min(max_width, int(round(crop_height * 4.0 / 3.0))))

        center_x = (x1 + x2) / 2.0
        # Bias down slightly so the review image includes shoulders and useful context.
        center_y = (y1 + y2) / 2.0 + face_height * 0.2
        left = max(0, min(frame_width - crop_width, int(round(center_x - crop_width / 2.0))))
        top = max(0, min(frame_height - crop_height, int(round(center_y - crop_height / 2.0))))
        return left, top, left + crop_width, top + crop_height

    def _thumbnail(self, frame: Any, box: list[float]) -> Optional[str]:
        try:
            import cv2

            frame_height, frame_width = frame.shape[:2]
            x1, y1, x2, y2 = self._review_crop_bounds(frame_width, frame_height, box)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            path = self.library.dir / "faces" / f"{time.time_ns()}.jpg"
            return str(path) if cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90]) else None
        except Exception:
            return None

    def _reviewable_face(self, frame: Any, face: Dict[str, Any]) -> bool:
        """Keep partial, tiny, low-confidence and blurred faces out of review."""
        settings = self.config.get("faces") if isinstance(self.config.get("faces"), dict) else {}
        try:
            score = float(face.get("detection_score", 1.0))
            box = [float(value) for value in face.get("bbox") or []]
        except (TypeError, ValueError):
            return False
        if score < float(settings.get("min_detection_confidence", 0.65)):
            return False
        if len(box) < 4:
            return frame is None
        x1, y1, x2, y2 = box[:4]
        width, height = x2 - x1, y2 - y1
        minimum = float(settings.get("min_face_size", 80))
        if width < minimum or height < minimum or width <= 0 or height <= 0:
            return False
        ratio = width / height
        if ratio < 0.55 or ratio > 1.45:
            return False
        if frame is None or not hasattr(frame, "shape"):
            return True
        frame_height, frame_width = frame.shape[:2]
        edge = float(settings.get("edge_margin_ratio", 0.015))
        if x1 <= frame_width * edge or y1 <= frame_height * edge or x2 >= frame_width * (1 - edge) or y2 >= frame_height * (1 - edge):
            return False
        blur_floor = float(settings.get("min_blur_variance", 35.0))
        if blur_floor <= 0:
            return True
        try:
            import cv2
            crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
            if crop.size == 0:
                return False
            grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(grey, cv2.CV_64F).var()) >= blur_floor
        except Exception:
            return False

    def _apply_analysis(self, frame: Any, analysis: Dict[str, Any], *, moving: bool) -> None:
        identities: list[str] = []
        owner_visible = False
        owner_confidence = 0.0
        embeddings: list[list[float]] = []
        reviewable_embeddings: list[list[float]] = []
        for face in analysis.get("faces") or []:
            embedding = [float(value) for value in face.get("embedding") or []]
            if not embedding:
                continue
            embeddings.append(embedding)
            reviewable = self._reviewable_face(frame, face)
            if reviewable:
                reviewable_embeddings.append(embedding)
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
            if verdict["status"] == "unknown" and reviewable:
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

        sleep_state = self._settled_posture(str(analysis.get("sleep_state") or "unknown"))
        person_count = max(int(analysis.get("person_count") or 0), len(embeddings))
        with self._lock:
            previous_sleep = self.state.sleep_state
            self._latest_embeddings = reviewable_embeddings
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
            self.state.face_model_loaded = bool(self.state.capabilities.get("faces", False))
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
