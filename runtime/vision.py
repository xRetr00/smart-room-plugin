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

# Measured on the owner's own library, 11 September: each of his 120 samples
# matched against the other 119 scores 0.404 at the median and 0.368 at the
# tenth percentile, while the best any other enrolled person reached against
# him was 0.271. At 0.42 more than half of his own face was "unknown", which is
# where nearly every "unknown visitor" came from.
OWNER_THRESHOLD = 0.36
KNOWN_THRESHOLD = 0.36
#: (match threshold, stranger ceiling) for each recogniser. Different models
#: are different spaces; a threshold means nothing outside its own.
#:
#: AdaFace, from the same labelled test: at 0.30 it kept 94% of the owner's
#: faces and accepted none of eight friend faces as him (their best was 0.22);
#: his friends match themselves at 0.28-0.48. Below 0.20 is a stranger.
MODEL_THRESHOLDS = {
    "arcface_r50": (0.36, 0.30),
    "adaface_ir101": (0.30, 0.20),
}
#: A known person not seen for this long is arriving, and is welcomed.
WELCOME_AFTER_HOURS = 3.0
#: Below this, a face is confidently nobody known, and only then a visitor.
#:
#: Between here and the match thresholds is "could not tell": the owner looking
#: down, turned half away, badly lit. Recording those as strangers is what put
#: his own face in the visitor queue, and then -- once approved -- side-on
#: glimpses of him into his own library, where they made matching worse.
STRANGER_CEILING = 0.30
#: How far off the camera a face may point and still be judged at all. The
#: nose's position between the eyes, 0 to 1; 0.5 is looking straight at it.
FRONTAL_SPAN = (0.2, 0.8)
#: Owner samples further than this from the rest are not him, or not usably.
#:
#: The owner's library had three below it on 11 September, all side profiles
#: approved from the visitor queue. Removing them kept his own recognition at
#: 91% and moved the closest stranger further away.
OUTLIER_FLOOR = 0.2
#: And no more than this many. Not lower: trimming to 40 by diversity dropped
#: his recognition from 92% to 57% -- his face varies, and each sample covers
#: a narrow slice of it. Matching 200 vectors costs microseconds.
LIBRARY_CAP = 200
#: A liveness score this low, averaged over a few frames, is a photo or a
#: screen. Deliberately strict: a real face in bad light scores low too, and
#: the cost of being wrong is calling the owner a stranger.
SPOOF_BELOW = 0.15
#: Recognised by clothes for this long after the face was last recognised.
OUTFIT_HOURS = 12.0
OUTFIT_MATCH = 0.35
PENDING_SIMILARITY = 0.45
MAX_PENDING = 40

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

#: An empty, still room is looked at this often instead, after this long.
#: Around the clock that is most of the day, and a face in a doorway still
#: moves -- which is what brings the full rate back.
IDLE_INFERENCE_GAP = 5.0
IDLE_AFTER_SECONDS = 60.0

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
        self._migrate()
        self._db.commit()
        #: Which recogniser's embeddings are matched. Set by the worker once
        #: the engine has loaded; rows from other models stay, unused, so
        #: switching back costs nothing.
        self.model = "arcface_r50"
        self.stranger_ceiling = STRANGER_CEILING
        self._last_learned: Dict[str, float] = {}
        self._seen_marked: Dict[str, float] = {}
        self._enforce_pending_limit()
        self._cleanup_orphan_thumbnails()
        with contextlib.suppress(Exception):
            self.curate_owner()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _migrate(self) -> None:
        """Columns added after libraries already existed. Old rows are ArcFace."""
        wanted = {
            "embeddings": [
                ("model", "TEXT NOT NULL DEFAULT 'arcface_r50'"),
                ("source", "TEXT NOT NULL DEFAULT 'enrolled'"),
                ("at", "TEXT"),
            ],
            "sightings": [("model", "TEXT NOT NULL DEFAULT 'arcface_r50'")],
            "people": [("last_seen", "TEXT")],
        }
        for table, columns in wanted.items():
            have = {row[1] for row in self._db.execute(f"PRAGMA table_info({table})")}
            for name, kind in columns:
                if name not in have:
                    self._db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")

    def use_model(self, model: str, owner_threshold: float, stranger_ceiling: float) -> None:
        self.model = model
        self.owner_threshold = self.known_threshold = float(owner_threshold)
        self.stranger_ceiling = float(stranger_ceiling)

    def samples_for(self, model: Optional[str] = None) -> int:
        with self._lock:
            return int(
                self._db.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE model = ?", (model or self.model,)
                ).fetchone()[0]
            )

    def rebuild(self, embed_image: Callable[[Any], Optional[list[float]]]) -> Dict[str, int]:
        """Fill the current model's library from the face crops already on disk.

        Every reviewed sighting kept its crop, so a new recogniser does not
        mean enrolling everybody again: each crop is embedded afresh and filed
        under the person it was approved as. What cannot be rebuilt -- samples
        enrolled live, with no crop -- is left to be learned again, which the
        worker does by itself from clear frames.
        """
        import cv2

        with self._lock:
            rows = self._db.execute(
                "SELECT identity, thumbnail FROM sightings WHERE status IN ('owner', 'known') "
                "AND thumbnail IS NOT NULL"
            ).fetchall()
            owner = self.owner_name()
        self._backup()
        added: Dict[str, int] = {}
        for row in rows:
            path = str(row["thumbnail"])
            image = cv2.imread(path) if Path(path).is_file() else None
            vector = embed_image(image) if image is not None else None
            if vector is None:
                continue
            name = str(row["identity"])
            self.enroll(name, [vector], owner=False, source="rebuilt")
            added[name] = added.get(name, 0) + 1
        if owner:
            self.set_owner(owner)
        logger.info("Face library rebuilt for %s: %s", self.model, added or "no usable crops")
        return added

    def enroll(
        self, name: str, embeddings: list[list[float]], owner: bool = False, source: str = "enrolled"
    ) -> Dict[str, Any]:
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
            stamp = _now_iso()
            self._db.executemany(
                "INSERT INTO embeddings (person_id, vector, model, source, at) VALUES (?, ?, ?, ?, ?)",
                [
                    (person_id, json.dumps(list(map(float, item))), self.model, source, stamp)
                    for item in embeddings
                ],
            )
            self._db.commit()
        return {"name": clean, "owner": is_owner, "samples": len(embeddings)}

    def people(self) -> list[Dict[str, Any]]:
        """Everybody known, with what says how well they are known.

        `samples` is this model's count. `learned` is how many were captured
        automatically, `last_seen` when the camera last named them, and
        `consistency` how alike their samples are (0-1): a library of one
        person's face should agree with itself, and a low number is the first
        sign that something that is not them has been filed under their name.
        """
        import numpy as np

        with self._lock:
            people = self._db.execute(
                "SELECT id, name, owner, at, last_seen FROM people ORDER BY owner DESC, name"
            ).fetchall()
            vectors: Dict[int, list] = {}
            learned: Dict[int, int] = {}
            newest: Dict[int, str] = {}
            for row in self._db.execute(
                "SELECT person_id, vector, source, at FROM embeddings WHERE model = ?", (self.model,)
            ):
                vectors.setdefault(row["person_id"], []).append(json.loads(row["vector"]))
                if row["source"] == "learned":
                    learned[row["person_id"]] = learned.get(row["person_id"], 0) + 1
                if row["at"] and row["at"] > newest.get(row["person_id"], ""):
                    newest[row["person_id"]] = row["at"]
        result = []
        for row in people:
            samples = vectors.get(row["id"], [])
            consistency = None
            if len(samples) >= 2:
                matrix = np.array(samples, dtype=np.float32)
                matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
                centre = matrix.mean(0)
                centre /= max(float(np.linalg.norm(centre)), 1e-9)
                consistency = round(float(np.median(matrix @ centre)), 3)
            result.append({
                "name": row["name"],
                "owner": bool(row["owner"]),
                "samples": len(samples),
                "learned": learned.get(row["id"], 0),
                "consistency": consistency,
                "last_seen": row["last_seen"],
                "newest_sample": newest.get(row["id"]),
                "at": row["at"],
                "model": self.model,
            })
        return result

    def mark_seen(self, name: str) -> Optional[str]:
        """Record that the camera named this person. Returns when it last did.

        Written at most once a minute per person: it is asked on every frame.
        """
        now = time.monotonic()
        with self._lock:
            row = self._db.execute("SELECT last_seen FROM people WHERE name = ?", (name,)).fetchone()
            if row is None:
                return None
            previous = row["last_seen"]
            if now - self._seen_marked.get(name, 0.0) >= 60:
                self._seen_marked[name] = now
                self._db.execute("UPDATE people SET last_seen = ? WHERE name = ?", (_now_iso(), name))
                self._db.commit()
        return previous

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
                "JOIN people p ON p.id = e.person_id WHERE e.model = ?",
                (self.model,),
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
        # Near someone known but not near enough: not a stranger either.
        status = "uncertain" if best_name and best_score >= self.stranger_ceiling else "unknown"
        return {"identity": "unknown", "status": status, "score": round(best_score, 4), "nearest": nearest}

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
                    "AND vector IS NOT NULL AND model = ? ORDER BY id DESC LIMIT ?",
                    (self.model, self.max_pending),
                ).fetchall()
                if any(cosine(embedding, json.loads(row["vector"])) >= self.pending_similarity for row in rows):
                    return None
            cursor = self._db.execute(
                "INSERT INTO sightings (at, identity, status, score, thumbnail, reported, vector, model) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _now_iso(), identity, status, float(score), thumbnail,
                    0 if status == "unknown" else 1,
                    json.dumps(embedding) if embedding is not None else None,
                    self.model,
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

    def curate_owner(self) -> int:
        """Drop the owner samples that are not him, and cap the rest. Returns removed.

        Backed up first, once a day: this deletes rows somebody approved by
        hand, and an approval is not something to lose to a heuristic.
        """
        import numpy as np

        with self._lock:
            rows = self._db.execute(
                "SELECT e.id, e.vector FROM embeddings e JOIN people p ON p.id = e.person_id "
                "WHERE p.owner = 1 AND e.model = ? ORDER BY e.id",
                (self.model,),
            ).fetchall()
            if len(rows) < 10:
                return 0
            vectors = np.array([json.loads(row["vector"]) for row in rows], dtype=np.float32)
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
            centre = vectors.mean(0)
            centre /= max(float(np.linalg.norm(centre)), 1e-9)
            keep = [i for i in range(len(rows)) if float(vectors[i] @ centre) >= OUTLIER_FLOOR]
            if len(keep) > LIBRARY_CAP:
                # Farthest-point: keep the samples that cover the most ground.
                similar = vectors[keep] @ vectors[keep].T
                chosen = [int(np.argmax(vectors[keep] @ centre))]
                nearest = similar[chosen[0]].copy()
                while len(chosen) < LIBRARY_CAP:
                    pick = int(np.argmin(nearest))
                    chosen.append(pick)
                    nearest = np.maximum(nearest, similar[pick])
                keep = [keep[i] for i in chosen]
            kept = set(keep)
            dropped = [int(rows[i]["id"]) for i in range(len(rows)) if i not in kept]
            if not dropped:
                return 0
            self._backup()
            self._db.executemany("DELETE FROM embeddings WHERE id = ?", [(i,) for i in dropped])
            self._db.commit()
        logger.info("Face library: removed %d owner sample(s) that did not fit", len(dropped))
        return len(dropped)

    def _backup(self) -> None:
        target = self.dir / f"faces.sqlite3.{datetime.now(timezone.utc):%Y%m%d}.bak"
        if not target.exists():
            with contextlib.suppress(Exception):
                backup = sqlite3.connect(target)
                self._db.backup(backup)
                backup.close()

    #: At most one new sample per person this often, however good the frames are.
    LEARN_EVERY_SECONDS = 900.0

    def learn_owner(self, embedding: list[float], score: float) -> bool:
        name = self.owner_name()
        return bool(name) and self.learn(name, embedding, score)

    def learn(self, name: str, embedding: list[float], score: float) -> bool:
        """Keep a new view of the owner, when it is clearly him and actually new.

        Clearly him: it already matched on its own, from a frame the caller
        judged good. New: its best match is below 0.75 -- a near-duplicate
        adds nothing but weight. This is how the library keeps up with
        lighting, a beard, glasses, without anybody approving a queue.
        """
        now = time.monotonic()
        if score < self.known_threshold or score > 0.75:
            return False
        last = self._last_learned.get(name, 0.0)
        if last and now - last < self.LEARN_EVERY_SECONDS:
            return False
        self._last_learned[name] = now
        self.enroll(name, [embedding], source="learned")
        self.curate_owner()
        logger.info("Face library: learned a new view of %s (score %.3f)", name, score)
        return True

    def unreported_visitors(self) -> list[Dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, at, identity, score, thumbnail, vector, model FROM sightings "
                "WHERE status = 'unknown' AND reported = 0 ORDER BY id"
            ).fetchall()
        visitors = []
        for row in rows:
            item = dict(row)
            vector = item.pop("vector", None)
            nearest: Dict[str, Any] = {}
            # A vector from another model has no nearest in this one.
            if vector and item.pop("model", self.model) == self.model:
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
                "SELECT vector, model FROM sightings WHERE id = ?", (sighting_id,)
            ).fetchone()
        if row is None or not row["vector"]:
            raise ValueError(f"no stored face for sighting {sighting_id}")
        if row["model"] != self.model:
            raise ValueError("that face was stored by a different recognition model; it can only be rejected")
        result = self.enroll(name, [json.loads(row["vector"])], owner=owner, source="approved")
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


def outfit(frame: Any, landmarks: list[Any]) -> Optional[Dict[str, Any]]:
    """What somebody is wearing: a colour histogram of the torso.

    The answer to "who is that" when there is no face to ask -- the owner
    turned to the wall, bent over the desk, walking away. Clothes change daily,
    so it only ever means "the person whose face was recognised earlier today",
    and it is only asked when no face is on offer.
    """
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    points = [landmarks[i] for i in (11, 12, 23, 24) if i < len(landmarks)]
    if len(points) < 2:
        return None
    xs = [min(max(point.x, 0.0), 1.0) * width for point in points]
    ys = [min(max(point.y, 0.0), 1.0) * height for point in points]
    left, right, top, bottom = int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))
    if right - left < 24:
        return None
    # Shoulders to hips, or to the bottom of the frame when sitting at the desk.
    bottom = max(bottom, min(height, top + (right - left)))
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, 1.0, 0.0, cv2.NORM_L1)
    return {"box": [left, top, right, bottom], "outfit": histogram.flatten().round(5).tolist()}


def _inside(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    """Ray casting: is a 0-1 point inside a 0-1 polygon."""
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-9) + x1:
            inside = not inside
    return inside


def outfit_distance(first: list[float], second: list[float]) -> float:
    """Bhattacharyya distance: 0 is identical, 1 is nothing in common."""
    import cv2
    import numpy as np

    return float(
        cv2.compareHist(np.float32(first), np.float32(second), cv2.HISTCMP_BHATTACHARYYA)
    )


class LocalVisionAnalyzer:
    """The lean face engine plus optional official MediaPipe Tasks models."""

    def __init__(
        self,
        *,
        auto_download: bool = True,
        detection_size: int = 320,
        dark_luma: float = 28.0,
        threads: int = 2,
        recogniser: str = "adaface_ir101",
    ) -> None:
        self.auto_download = auto_download
        self.recogniser = recogniser
        #: Replaced by what actually loaded; see `FaceEngine.describe`.
        self.face_model = "ArcFace R50 · SCRFD-10G"
        self.face_provider = "CPUExecutionProvider"
        #: 320 unless a moving room shows nobody, then one look at 640 for a
        #: face further from the lens. 25 ms a frame against 97.
        self.detection_size = int(detection_size)
        #: Below this mean brightness the frame is too dark to judge as it is.
        self.dark_luma = float(dark_luma)
        self.threads = threads
        self._face: Any = None
        self._gesture: Any = None
        self._pose: Any = None
        self.capabilities = {"faces": False, "gestures": False, "posture": False}

    def load(self) -> None:
        from .face_engine import FaceEngine

        engine = FaceEngine(
            vision_models_home(),
            threads=self.threads,
            auto_download=self.auto_download,
            fetch=_download,
            recogniser=self.recogniser,
        )
        engine.load()
        self._face = engine
        self.face_model = engine.describe()
        self.capabilities["faces"] = True
        self.capabilities["quality"] = engine.quality is not None
        self.capabilities["liveness"] = engine.liveness is not None

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

    def analyze(self, frame: Any, *, wide: bool = False) -> Dict[str, Any]:
        from .face_engine import brighten, luma

        if self._face is None:
            self.load()
        brightness = luma(frame)
        dark = brightness < self.dark_luma
        # Judged on a lifted copy when dark: a face the detector cannot find is
        # a face nobody can decide anything about, including whether to light
        # the room to see it.
        seen = brighten(frame) if dark else frame
        faces = self._face.faces(seen, self.detection_size)
        if not faces and wide and self.detection_size < 640:
            faces = self._face.faces(seen, 640)
        result: Dict[str, Any] = {
            "faces": faces,
            "person_count": len(faces),
            "gesture": None,
            "gesture_confidence": 0.0,
            "sleep_state": "unknown",
            "brightness": round(brightness, 1),
            "dark": dark,
            "bodies": [],
            "capabilities": dict(self.capabilities),
        }
        if self._gesture is None and self._pose is None:
            return result
        try:
            import cv2
            import mediapipe as mp

            rgb = cv2.cvtColor(seen, cv2.COLOR_BGR2RGB)
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
                # Colour lies in a lifted dark frame, so no outfit from one.
                if not dark:
                    result["bodies"] = [
                        body for body in (outfit(frame, landmarks) for landmarks in pose.pose_landmarks) if body
                    ]
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


class Track:
    """One face followed from frame to frame, and what it has been judged to be."""

    def __init__(self, box: list[float], now: float) -> None:
        self.box = box
        self.seen_at = now
        self.samples: list[tuple[list[float], float]] = []
        self.live: list[float] = []
        self.label: Optional[str] = None
        self.name = "unknown"
        self.score = 0.0
        self.recorded = False
        #: (quality, liveness) per front-on frame, enforced or not.
        self.observed: list[tuple[float, float]] = []
        self.calibrated = False
        #: Whether this track's person has been noted as seen.
        self.announced = False

    def add(self, embedding: list[float], weight: float, live: Optional[float]) -> None:
        self.samples = (self.samples + [(embedding, max(0.05, weight))])[-8:]
        if live is not None:
            self.live = (self.live + [float(live)])[-8:]

    def spoofed(self) -> bool:
        return len(self.live) >= 3 and sum(self.live) / len(self.live) < SPOOF_BELOW

    def judge(self, library: "FaceLibrary", confirm: int) -> Optional[str]:
        """The track's identity, from every good frame so far rather than this one.

        Once the owner, the owner for as long as the face stays tracked: the
        frame where he turns to the wall is the same person as the one where
        he looked at the lens, and deciding it afresh is how his back became
        a visitor.
        """
        if self.label == "owner" or not self.samples:
            return self.label
        import numpy as np

        weights = np.array([weight for _, weight in self.samples])
        mean = (np.array([vector for vector, _ in self.samples]) * weights[:, None]).sum(0)
        mean /= max(float(np.linalg.norm(mean)), 1e-9)
        verdict = library.match(mean.tolist())
        self.score = float(verdict["score"])
        if self.spoofed():
            # The owner's face on a photo or a screen is exactly a visitor.
            self.label, self.name = "unknown", "unknown"
        elif verdict["status"] == "owner" and self.live and len(self.live) < 3:
            # Not yet. Owner is sticky for the life of the track, so it must
            # not be granted before liveness has had frames enough to object
            # -- otherwise a photo held up is the owner from its first frame.
            return None
        elif verdict["status"] in ("owner", "known"):
            self.label, self.name = str(verdict["status"]), str(verdict["identity"])
        elif verdict["status"] == "unknown" and len(self.samples) >= confirm:
            self.label, self.name = "unknown", "unknown"
        return self.label


class FaceTracks:
    """Faces followed across frames by where they are."""

    def __init__(self, ttl: float = 4.0) -> None:
        self.ttl = ttl
        self.tracks: list[Track] = []

    @staticmethod
    def _overlap(a: list[float], b: list[float]) -> float:
        left, top = max(a[0], b[0]), max(a[1], b[1])
        right, bottom = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, right - left) * max(0.0, bottom - top)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0

    def assign(self, faces: list[Dict[str, Any]], now: float) -> list[Track]:
        """The track for each face, in order. New faces start new tracks."""
        self.tracks = [track for track in self.tracks if now - track.seen_at <= self.ttl]
        free = list(self.tracks)
        assigned: list[Track] = []
        for face in faces:
            box = [float(value) for value in (face.get("bbox") or [0, 0, 0, 0])[:4]]
            best = max(free, key=lambda track: self._overlap(track.box, box), default=None)
            if best is None or self._overlap(best.box, box) < 0.25:
                best = Track(box, now)
                self.tracks.append(best)
            else:
                free.remove(best)
            best.box, best.seen_at = box, now
            assigned.append(best)
        return assigned


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
        self.face_config = face_config
        self.analyzer = analyzer or LocalVisionAnalyzer(
            auto_download=bool(config.get("auto_download_models", True)),
            detection_size=int(face_config.get("face_detection_size", 320)),
            dark_luma=float(config.get("dark_brightness", 28)),
            recogniser=str(face_config.get("recogniser", "adaface_ir101")),
        )
        #: Called with a known person's name when they arrive -- seen for the
        #: first time in `WELCOME_AFTER_HOURS`. The runtime welcomes them.
        self.on_person: Optional[Callable[[str], None]] = None
        #: Tracked for four seconds of absence -- several analyses at the
        #: normal pace, so a face that blurs for a frame keeps its identity.
        self.tracks = FaceTracks(ttl=float(face_config.get("track_seconds", 4.0)))
        self.confirm_frames = int(face_config.get("confirm_frames", 3))
        self.learn_quality = float(face_config.get("learn_quality", 0.5))
        self.close_ratio = float(face_config.get("close_face_ratio", 0.2))
        #: On, from 40 live frames of the owner on 12 September: every one
        #: scored 1.00 against a spoof line of 0.15. (The stored review crops
        #: scored far lower -- recompressed JPEGs look like prints -- which is
        #: why it stayed off until live frames could be measured.) Never on a
        #: lifted dark frame; see `_apply_analysis`.
        self.enforce_liveness = bool(face_config.get("enforce_liveness", True))
        #: What the owner is wearing, and when his face last vouched for it.
        self._outfit: Optional[tuple[list[float], float]] = None
        self.zones = self._read_zones(config.get("zones"))
        self._quiet_since = 0.0
        self.capture_factory = capture_factory
        self.state = VisionState(
            enabled=bool(config.get("enabled", False)),
            camera_index=int(config.get("camera_index", 0)),
            face_model=str(getattr(self.analyzer, "face_model", "ArcFace R50 · SCRFD-10G")),
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

    def _prepare_library(self) -> None:
        """Point the library at the model the engine actually loaded, rebuilding it if new.

        Runs once, on the camera thread, before the first frame: loading the
        models takes seconds and the rebuild a minute, and neither should hold
        up the runtime's start.
        """
        engine = getattr(self.analyzer, "_face", None)
        if engine is None and hasattr(self.analyzer, "load"):
            self.analyzer.load()
            engine = getattr(self.analyzer, "_face", None)
        model = getattr(engine, "model", None)
        if not model:
            return
        threshold, ceiling = MODEL_THRESHOLDS.get(model, (OWNER_THRESHOLD, STRANGER_CEILING))
        if model == "arcface_r50" and "match_threshold" in self.face_config:
            threshold = float(self.face_config["match_threshold"])
        threshold = float(self.face_config.get(f"{model}_match_threshold", threshold))
        self.library.use_model(model, threshold, ceiling)
        if self.library.samples_for() == 0:
            self.library.rebuild(engine.embed_image)

    def _capture_loop(self, capture: Any) -> None:
        import cv2
        import numpy as np

        if not getattr(self, "_library_ready", False):
            try:
                self._prepare_library()
            except Exception:
                logger.warning("Could not prepare the face library for the loaded model", exc_info=True)
            self._library_ready = True

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
            if not easy and self._quiet_since and now - self._quiet_since > IDLE_AFTER_SECONDS:
                # Nobody here and nothing moving for a while: look every few
                # seconds instead of every second. Movement brings it straight
                # back, because motion is measured on every analysis.
                interval = max(interval, IDLE_INFERENCE_GAP)
            with self._lock:
                self._latest_frame = frame
                self.state.last_frame_at = _now_iso()
                self.state.stale = False
            if now - last_inference < interval:
                continue
            grey = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY)
            difference = None if previous is None else np.abs(grey.astype("int16") - previous.astype("int16"))
            motion = 100.0 if difference is None else float(np.mean(difference))
            # A light switching on changes every pixel at once, and read as
            # movement it put "motion" in every zone -- the door included, at
            # the very moment an automation lit the room for an arrival.
            relit = difference is not None and (
                abs(float(grey.mean()) - float(previous.mean())) > 12 or float((difference > 25).mean()) > 0.5
            )
            # Per zone as well as overall: movement at the door is somebody
            # arriving, and movement only at the desk or the bed is not.
            zone_motion = {} if difference is None or easy or relit else self._zone_motion(difference > 25)
            previous = grey
            last_inference = now
            moving = motion >= motion_threshold
            try:
                analysis = self.analyzer.analyze(frame, wide=moving)
            except TypeError:
                # An analyzer without the wide retry (tests, older builds).
                analysis = self.analyzer.analyze(frame)
            self._apply_analysis(frame, analysis, moving=moving, zone_motion=zone_motion)
            with self._lock:
                empty = self.state.person_count == 0
            self._quiet_since = (self._quiet_since or now) if (empty and not moving) else 0.0

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

    @staticmethod
    def _facing_camera(face: Dict[str, Any]) -> bool:
        """Whether this face points at the lens closely enough to be judged.

        A profile, the top of a head, somebody looking down at a phone: the
        embedding of any of those says little about who it is, and every one
        of them was being matched, failing, and filed as a visitor. The face
        in the owner's 14:47 "unknown visitor" was the owner, from behind.

        The nose's place between the eyes is the turn; its place between the
        eyes and the mouth is the tilt. No landmarks means an analyzer that
        does not provide them, which is judged as before rather than refused.
        """
        points = face.get("landmarks") or []
        if len(points) < 5:
            return True
        (lx, ly), (rx, ry), (nx, ny), (ml_x, ml_y), (mr_x, mr_y) = points[:5]
        low, high = FRONTAL_SPAN
        across = rx - lx
        eyes_y, mouth_y = (ly + ry) / 2, (ml_y + mr_y) / 2
        down = mouth_y - eyes_y
        if across <= 0 or down <= 0:
            return False
        return low <= (nx - lx) / across <= high and low <= (ny - eyes_y) / down <= high

    def _reviewable_face(self, frame: Any, face: Dict[str, Any]) -> bool:
        """Keep partial, tiny, turned, low-confidence and blurred faces out of review."""
        settings = self.config.get("faces") if isinstance(self.config.get("faces"), dict) else {}
        if not self._facing_camera(face):
            return False
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

    def _apply_analysis(
        self,
        frame: Any,
        analysis: Dict[str, Any],
        *,
        moving: bool,
        zone_motion: Optional[Dict[str, str]] = None,
    ) -> None:
        now = time.monotonic()
        faces = [face for face in analysis.get("faces") or [] if face.get("embedding")]
        identities: list[str] = []
        owner_visible = False
        owner_confidence = 0.0
        owner_seen_by = ""
        close_unidentified = False
        embeddings: list[list[float]] = []
        reviewable_embeddings: list[list[float]] = []
        frame_height = float(frame.shape[0]) if frame is not None and hasattr(frame, "shape") else 0.0
        someone_else = False
        for face, track in zip(faces, self.tracks.assign(faces, now)):
            embedding = [float(value) for value in face["embedding"]]
            embeddings.append(embedding)
            facing = self._facing_camera(face)
            reviewable = self._reviewable_face(frame, face)
            if reviewable:
                reviewable_embeddings.append(embedding)
            quality = face.get("quality")
            if facing:
                # A face pointed at the lens can say anything about who it is,
                # including "nobody known".
                live = face.get("live")
                if live is not None:
                    track.observed.append((1.0 if quality is None else float(quality), float(live)))
                    # A dark frame is brightened before it is judged, and a
                    # lifted image is exactly what a print looks like.
                    if not self.enforce_liveness or analysis.get("dark"):
                        live = None
                track.add(embedding, 1.0 if quality is None else float(quality), live)
            else:
                # A turned face may only say "this is someone known". With
                # AdaFace it does that well: 33 of 40 live frames of the owner
                # on his phone were side-on, and every one matched him at a
                # median 0.59. It can never make a stranger or be learned from.
                side = self.library.match(embedding)
                if side["status"] in ("owner", "known") and float(side["score"]) >= self.library.known_threshold + 0.05:
                    track.add(embedding, 0.5, None)
            label = track.judge(self.library, self.confirm_frames)
            identities.append(track.name if label in ("owner", "known") else "unknown")
            if label in ("owner", "known"):
                self._arrived(track)
                self._learn(face, embedding, quality, facing and reviewable, track)
            if label == "owner":
                owner_visible, owner_seen_by = True, "face"
                owner_confidence = max(owner_confidence, track.score)
                if not track.calibrated and len(track.observed) >= 5:
                    # Once per track, from frames known to be the owner: the
                    # real distribution on this camera, which is what the
                    # quality and liveness thresholds have to be set from.
                    track.calibrated = True
                    qualities = [q for q, _ in track.observed]
                    lives = [v for _, v in track.observed]
                    logger.info(
                        "Owner face calibration: quality %.2f (min %.2f) liveness %.2f (min %.2f) "
                        "over %d frames, brightness %.0f",
                        sum(qualities) / len(qualities), min(qualities),
                        sum(lives) / len(lives), min(lives), len(lives),
                        float(analysis.get("brightness") or 0.0),
                    )
            elif label in ("known", "unknown"):
                someone_else = True
            box = face.get("bbox") or [0, 0, 0, 0]
            if label not in ("owner", "known") and frame_height and (box[3] - box[1]) >= frame_height * self.close_ratio:
                close_unidentified = True
            # Known identities are live state, not a surveillance history.
            # Persist only a confirmed stranger, once per track, from a frame
            # good enough to look at.
            if label == "unknown" and reviewable and not track.recorded:
                track.recorded = True
                sighting_id = self.library.record_sighting(
                    "unknown", "unknown", track.score, self._thumbnail(frame, list(box)), embedding
                )
                if sighting_id is not None:
                    self.emit_event(
                        "vision_visitor_seen",
                        {
                            "sighting_id": sighting_id,
                            "spoofed": track.spoofed(),
                            "summary": (
                                "A face held up to the camera -- a photo or a screen"
                                if track.spoofed()
                                else "Unknown visitor seen by Smart Room"
                            ),
                        },
                    )

        owner_visible, owner_seen_by = self._by_outfit(
            analysis.get("bodies") or [], owner_visible, owner_seen_by, someone_else, now
        )
        place = self._place(frame, faces, analysis.get("bodies") or [])

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
            self.state.owner_seen_by = owner_seen_by
            self.state.place = place
            self.state.close_face_unidentified = close_unidentified
            self.state.brightness = float(analysis.get("brightness") or 0.0)
            self.state.dark = bool(analysis.get("dark", False))
            if zone_motion:
                self.state.zone_motion = {**self.state.zone_motion, **zone_motion}
            self.state.identities = identities[:8]
            self.state.pending_visitors = len(self.library.unreported_visitors())
            self.state.activity = "moving" if moving else "still"
            self.state.gesture = str(gesture) if gesture else None
            self.state.gesture_confidence = round(gesture_confidence, 4)
            self.state.sleep_state = sleep_state
            self.state.capabilities = dict(analysis.get("capabilities") or self.analyzer.capabilities)
            self.state.face_model_loaded = bool(self.state.capabilities.get("faces", False))
            # Read again here: the worker's state is built before the models
            # load, so the name it started with is only ever a placeholder.
            self.state.face_model = str(getattr(self.analyzer, "face_model", self.state.face_model))
            snapshot = VisionState(**asdict(self.state))
        self.publish_state(snapshot)
        if sleep_state != previous_sleep and sleep_state in {"awake", "resting"}:
            self.emit_event(
                "vision_sleep_state",
                {"sleep_state": sleep_state, "summary": f"Vision posture: {sleep_state}"},
            )

    def _learn(
        self, face: Dict[str, Any], embedding: list[float], quality: Optional[float], good: bool, track: Track
    ) -> None:
        """Offer this frame to the library, when it is good enough to learn from.

        Only with the quality model loaded, only from a front-on face, only
        from a live one, and only when the frame matched the owner by itself
        -- never on the strength of the track's earlier verdict.
        """
        if not good or quality is None or float(quality) < self.learn_quality or track.spoofed():
            return
        verdict = self.library.match(embedding)
        # The frame on its own must name the same person the track did.
        if verdict["status"] in ("owner", "known") and verdict["identity"] == track.name:
            self.library.learn(track.name, embedding, float(verdict["score"]))

    def _arrived(self, track: Track) -> None:
        """Note that the camera named somebody, and say so if they are arriving."""
        # Every frame (the library writes at most once a minute), so "last
        # seen" stays true for somebody who sits at the desk all afternoon.
        previous = self.library.mark_seen(track.name)
        if track.announced:
            return
        track.announced = True
        if track.label != "known" or self.on_person is None:
            return
        try:
            gone = (
                datetime.now(timezone.utc) - datetime.fromisoformat(str(previous).replace("Z", "+00:00"))
            ).total_seconds() if previous else float("inf")
        except ValueError:
            gone = float("inf")
        if gone >= WELCOME_AFTER_HOURS * 3600:
            try:
                self.on_person(track.name)
            except Exception:
                logger.warning("Could not welcome %s", track.name, exc_info=True)

    def _by_outfit(
        self,
        bodies: list[Dict[str, Any]],
        owner_visible: bool,
        seen_by: str,
        someone_else: bool,
        now: float,
    ) -> tuple[bool, str]:
        """Remember what the owner is wearing, and recognise him by it.

        Learned only while his face is recognised and he is the one person in
        view, so the outfit is certainly his. Used only when no face is on
        offer at all -- any identified face outranks clothes -- and only for
        one person, because two people in similar colours is exactly the case
        a histogram cannot separate.
        """
        if owner_visible:
            if len(bodies) == 1:
                self._outfit = (list(bodies[0]["outfit"]), now)
            return owner_visible, seen_by
        if someone_else or len(bodies) != 1 or self._outfit is None:
            return False, ""
        outfit, learned_at = self._outfit
        if now - learned_at > OUTFIT_HOURS * 3600:
            self._outfit = None
            return False, ""
        if outfit_distance(outfit, bodies[0]["outfit"]) <= OUTFIT_MATCH:
            return True, "outfit"
        return False, ""

    def _place(self, frame: Any, faces: list[Dict[str, Any]], bodies: list[Dict[str, Any]]) -> str:
        """Which zone the person is in -- the desk, the bed, the door -- or empty.

        From the face when there is one and the body otherwise, and only for a
        single person: with two, "where is he" has no one answer.
        """
        if not self.zones or frame is None or not hasattr(frame, "shape"):
            return ""
        boxes = [face.get("bbox") for face in faces] or [body.get("box") for body in bodies]
        if len(boxes) != 1 or not boxes[0]:
            return ""
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = boxes[0][:4]
        point = ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height)
        for name, polygon in self.zones.items():
            if _inside(point, polygon):
                return name
        return ""

    @staticmethod
    def _read_zones(zones: Any) -> Dict[str, list[list[float]]]:
        """Named polygons in 0-1 frame coordinates, from config strings or lists."""
        found: Dict[str, list[list[float]]] = {}
        for name, polygon in (zones or {}).items() if isinstance(zones, dict) else []:
            try:
                points = json.loads(polygon) if isinstance(polygon, str) else polygon
                points = [[float(x), float(y)] for x, y in points]
            except (TypeError, ValueError):
                logger.warning("Ignoring vision zone %r: not a list of points", name)
                continue
            if len(points) >= 3:
                found[str(name)] = points
        return found

    #: Share of a zone's pixels that must change to count as movement there.
    ZONE_MOTION_SHARE = 0.03

    def _zone_motion(self, changed: Any) -> Dict[str, str]:
        """Which zones moved, given a boolean change mask at any resolution."""
        import cv2
        import numpy as np

        if not self.zones or changed is None:
            return {}
        height, width = changed.shape[:2]
        moved: Dict[str, str] = {}
        for name, polygon in self.zones.items():
            mask = np.zeros((height, width), dtype=np.uint8)
            points = np.array([[x * (width - 1), y * (height - 1)] for x, y in polygon], dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)
            area = int(mask.sum())
            if area and float(changed[mask.astype(bool)].sum()) / area >= self.ZONE_MOTION_SHARE:
                moved[name] = _now_iso()
        return moved

    def _set_state(self, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(self.state, key, value)
            snapshot = VisionState(**asdict(self.state))
        self.publish_state(snapshot)
