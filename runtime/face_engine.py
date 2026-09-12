"""Face detection, alignment and recognition on ONNX Runtime, and nothing else.

This replaces `insightface.app.FaceAnalysis`, and the reason is memory rather
than accuracy: the models are the same two files. Measured on the owner's
machine, 11 September:

* `import insightface` alone cost 549 MB, because it imports PyTorch (with its
  CUDA libraries), SciPy, scikit-image, matplotlib and albumentations. The
  vision worker uses none of them.
* `FaceAnalysis` loads five models and the worker used two. The other three --
  3D landmarks, 106-point landmarks, gender and age -- cost 175 MB and ran on
  every face in every frame: 351 ms a frame with a face in it, against 143 ms
  for the two that matter.
* ONNX Runtime's default arena keeps every allocation it ever made. Without it
  the two sessions hold 210 MB instead of 337 MB, at the same speed.

What is left is SCRFD for faces and five landmarks, the ArcFace alignment, and
ArcFace R50 for the embedding -- byte-for-byte the models, and the maths, that
produced every embedding already in the owner's library, so nothing enrolled
has to be redone. Two small optional models ride along: eDifFIQA(T) for how
usable a face is, and MiniFASNet for whether it is a live one.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

#: Where InsightFace publishes the pack the library's embeddings came from.
PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
PACK = "buffalo_l"
DETECTOR = "det_10g.onnx"
RECOGNISER = "w600k_r50.onnx"
#: eDifFIQA(T), 6.6 MB, CC BY 4.0, from the OpenCV model zoo.
QUALITY_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_image_quality_assessment_ediffiqa/ediffiqa_tiny_jun2024.onnx"
)
#: MiniFASNetV2, about 1.7 MB, Apache 2.0.
LIVENESS_URL = "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx"

#: The recognisers this engine can run, by the name stored beside every
#: embedding. Embeddings from different models are different spaces and are
#: never compared with each other.
#:
#: AdaFace IR-101 is the default, from a test on the owner's own labelled faces
#: (51 of him, 8 of two friends, all from this camera, 12 September): the gap
#: between his weakest 10% of matches and the best friend-versus-him score was
#: 0.126, against 0.058 for ArcFace -- more than twice the safety margin, for
#: about 90 MB and 80 ms a face more. Kept for the case it cannot load.
RECOGNISERS: Dict[str, Dict[str, Any]] = {
    "adaface_ir101": {
        "label": "AdaFace IR-101",
        "file": "adaface_ir_101.onnx",
        "url": "https://github.com/yakhyo/adaface-onnx/releases/download/weights/adaface_ir_101.onnx",
        "sha256": "f2eb07d03de0af560a82e1214df799fec5e09375d43521e2868f9dc387e5a43e",
        # AdaFace was trained on BGR; ArcFace on RGB.
        "rgb": False,
    },
    "arcface_r50": {"label": "ArcFace R50", "file": None, "url": None, "sha256": None, "rgb": True},
}
DEFAULT_RECOGNISER = "adaface_ir101"

#: The five points ArcFace was trained on, in its 112x112 crop.
ARCFACE_POINTS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def session(path: Path, threads: int = 2) -> Any:
    """A CPU session that gives memory back and leaves cores for everything else."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.intra_op_num_threads = max(1, int(threads))
    options.inter_op_num_threads = 1
    # SCRFD is exported with fixed anchor counts and run at 320, so ORT warns
    # about output shapes on every frame. The shapes are right; the log is not.
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def similarity_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Umeyama's least-squares similarity, as scikit-image computes it.

    Reproduced rather than approximated with `cv2.estimateAffinePartial2D`:
    the owner's library was aligned with exactly this, and a different fit
    moves every embedding a little.
    """
    count = source.shape[0]
    source_mean, target_mean = source.mean(0), target.mean(0)
    source_centred, target_centred = source - source_mean, target - target_mean
    covariance = target_centred.T @ source_centred / count
    signs = np.ones(2)
    if np.linalg.det(covariance) < 0:
        signs[1] = -1
    u, singular, vt = np.linalg.svd(covariance)
    rotation = u @ np.diag(signs) @ vt
    scale = (singular @ signs) / source_centred.var(axis=0).sum()
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = rotation * scale
    matrix[:, 2] = target_mean - scale * (rotation @ source_mean)
    return matrix


def align(frame: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    import cv2

    matrix = similarity_transform(np.asarray(landmarks, dtype=np.float64), ARCFACE_POINTS * (size / 112.0))
    return cv2.warpAffine(frame, matrix, (size, size), borderValue=0.0)


class Detector:
    """SCRFD-10G: boxes, a confidence, and five landmarks per face."""

    STRIDES = (8, 16, 32)
    ANCHORS = 2

    def __init__(self, path: Path, threads: int = 2, threshold: float = 0.5) -> None:
        self.session = session(path, threads)
        self.input = self.session.get_inputs()[0].name
        self.outputs = [item.name for item in self.session.get_outputs()]
        self.threshold = threshold
        self._centres: Dict[tuple[int, int, int], np.ndarray] = {}

    def _anchor_centres(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        if key not in self._centres:
            grid = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            centres = (grid * stride).reshape(-1, 2)
            self._centres[key] = np.repeat(centres, self.ANCHORS, axis=0)
        return self._centres[key]

    def detect(self, frame: np.ndarray, size: int = 320) -> list[Dict[str, Any]]:
        import cv2

        height, width = frame.shape[:2]
        # InsightFace's own letterboxing, rounding included, so boxes and
        # landmarks land where they did when the library was built.
        if height / width > 1:
            new_height, new_width = size, int(size / (height / width))
        else:
            new_width, new_height = size, int(size * (height / width))
        scale = new_height / height
        resized = cv2.resize(frame, (new_width, new_height))
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, (size, size), (127.5, 127.5, 127.5), swapRB=True)
        outputs = self.session.run(self.outputs, {self.input: blob})
        if outputs[0].ndim == 3:
            outputs = [item[0] for item in outputs]

        scores, boxes, points = [], [], []
        levels = len(self.STRIDES)
        for index, stride in enumerate(self.STRIDES):
            score = outputs[index].reshape(-1)
            keep = np.where(score >= self.threshold)[0]
            if not len(keep):
                continue
            centres = self._anchor_centres(size // stride, size // stride, stride)[keep]
            distance = outputs[index + levels][keep] * stride
            landmark = outputs[index + levels * 2][keep] * stride
            boxes.append(np.hstack([centres - distance[:, :2], centres + distance[:, 2:]]))
            points.append((np.tile(centres, 5) + landmark).reshape(-1, 5, 2))
            scores.append(score[keep])
        if not scores:
            return []
        score = np.concatenate(scores)
        box = np.vstack(boxes) / scale
        point = np.vstack(points) / scale
        chosen = self._suppress(box, score)
        return [
            {"bbox": box[i].tolist(), "detection_score": float(score[i]), "landmarks": point[i].tolist()}
            for i in chosen
        ]

    @staticmethod
    def _suppress(boxes: np.ndarray, scores: np.ndarray, overlap: float = 0.4) -> list[int]:
        order = scores.argsort()[::-1]
        area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        kept: list[int] = []
        while order.size:
            first = int(order[0])
            kept.append(first)
            rest = order[1:]
            left = np.maximum(boxes[first, 0], boxes[rest, 0])
            top = np.maximum(boxes[first, 1], boxes[rest, 1])
            right = np.minimum(boxes[first, 2], boxes[rest, 2])
            bottom = np.minimum(boxes[first, 3], boxes[rest, 3])
            inter = np.maximum(0.0, right - left + 1) * np.maximum(0.0, bottom - top + 1)
            order = rest[inter / (area[first] + area[rest] - inter) <= overlap]
        return kept


class Recogniser:
    """A unit vector per aligned face. ArcFace R50 or AdaFace IR-101."""

    def __init__(self, path: Path, threads: int = 2, rgb: bool = True) -> None:
        self.session = session(path, threads)
        self.input = self.session.get_inputs()[0].name
        self.rgb = rgb

    def embed(self, face: np.ndarray) -> np.ndarray:
        import cv2

        blob = cv2.dnn.blobFromImage(face, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=self.rgb)
        vector = self.session.run(None, {self.input: blob})[0][0]
        return vector / max(float(np.linalg.norm(vector)), 1e-9)


class Quality:
    """eDifFIQA(T): how usable an aligned face is. Higher is better, about 0 to 1."""

    def __init__(self, path: Path) -> None:
        self.session = session(path, 1)
        self.input = self.session.get_inputs()[0].name

    def score(self, face: np.ndarray) -> float:
        import cv2

        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = ((rgb / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
        return float(np.ravel(self.session.run(None, {self.input: blob})[0])[0])


class Liveness:
    """MiniFASNetV2: the chance this is a real face rather than a photo or screen."""

    SCALE = 2.7
    SIZE = 80

    def __init__(self, path: Path) -> None:
        self.session = session(path, 1)
        self.input = self.session.get_inputs()[0].name

    def real(self, frame: np.ndarray, bbox: list[float]) -> float:
        import cv2

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox[:4]
        box_w, box_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        scale = min((height - 1) / box_h, (width - 1) / box_w, self.SCALE)
        centre_x, centre_y = x1 + box_w / 2, y1 + box_h / 2
        left = max(0, int(centre_x - box_w * scale / 2))
        top = max(0, int(centre_y - box_h * scale / 2))
        right = min(width - 1, int(centre_x + box_w * scale / 2))
        bottom = min(height - 1, int(centre_y + box_h * scale / 2))
        crop = cv2.resize(frame[top : bottom + 1, left : right + 1], (self.SIZE, self.SIZE))
        logits = self.session.run(None, {self.input: crop.astype(np.float32).transpose(2, 0, 1)[None]})[0][0]
        odds = np.exp(logits - logits.max())
        return float(odds[1] / odds.sum())


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def brighten(frame: np.ndarray) -> np.ndarray:
    """A dark frame, lifted enough for the detector to find a face in it.

    Gamma rather than a plain gain: it lifts the shadows a face sits in
    without blowing out the one lamp in the corner.
    """
    import cv2

    table = (np.linspace(0, 1, 256) ** 0.45 * 255).astype(np.uint8)
    return cv2.LUT(frame, table)


def luma(frame: np.ndarray) -> float:
    import cv2

    return float(cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).mean())


class FaceEngine:
    """Everything the worker asks about faces, loaded lean."""

    def __init__(
        self,
        root: Path,
        *,
        threads: int = 2,
        auto_download: bool = True,
        fetch: Optional[Callable[[str, Path], Path]] = None,
        recogniser: str = DEFAULT_RECOGNISER,
    ) -> None:
        self.root = Path(root)
        self.threads = threads
        #: The model the embeddings come from; see `RECOGNISERS`.
        self.model = recogniser if recogniser in RECOGNISERS else DEFAULT_RECOGNISER
        self.auto_download = auto_download
        self.fetch = fetch
        self.detector: Optional[Detector] = None
        self.recogniser: Optional[Recogniser] = None
        self.quality: Optional[Quality] = None
        self.liveness: Optional[Liveness] = None

    def _pack(self) -> Path:
        """The two files from the InsightFace pack, fetching it only when absent."""
        folder = self.root / "models" / PACK
        needed = [folder / DETECTOR, folder / RECOGNISER]
        if all(path.is_file() for path in needed):
            return folder
        if not (self.auto_download and self.fetch):
            raise FileNotFoundError(f"face models missing from {folder}")
        archive = self.fetch(PACK_URL, self.root / "models" / f"{PACK}.zip")
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            for name in bundle.namelist():
                if Path(name).name in (DETECTOR, RECOGNISER):
                    (folder / Path(name).name).write_bytes(bundle.read(name))
        # 288 MB of models this process will never open.
        archive.unlink(missing_ok=True)
        return folder

    def _optional(self, url: str, name: str, kind: Any) -> Any:
        path = self.root / name
        try:
            if not path.is_file() and self.auto_download and self.fetch:
                self.fetch(url, path)
            return kind(path) if path.is_file() else None
        except Exception:
            logger.warning("Optional face model %s unavailable; carrying on without it", name, exc_info=True)
            return None

    def _recogniser_path(self, folder: Path) -> Path:
        spec = RECOGNISERS[self.model]
        if not spec["file"]:
            return folder / RECOGNISER
        path = self.root / spec["file"]
        if not path.is_file():
            if not (self.auto_download and self.fetch):
                raise FileNotFoundError(f"{spec['label']} is not downloaded")
            self.fetch(spec["url"], path)
        if spec["sha256"] and _sha256(path) != spec["sha256"]:
            path.unlink(missing_ok=True)
            raise ValueError(f"{spec['label']} failed its checksum and was removed")
        return path

    def load(self) -> None:
        folder = self._pack()
        self.detector = Detector(folder / DETECTOR, self.threads)
        try:
            path = self._recogniser_path(folder)
        except Exception:
            # Recognition must not go dark because the better model could not
            # be fetched. ArcFace is already on disk, and the library keeps its
            # embeddings, so falling back is a working room, not an empty one.
            logger.warning("%s unavailable; using ArcFace R50", RECOGNISERS[self.model]["label"], exc_info=True)
            self.model, path = "arcface_r50", folder / RECOGNISER
        self.recogniser = Recogniser(path, self.threads, rgb=bool(RECOGNISERS[self.model]["rgb"]))
        self.quality = self._optional(QUALITY_URL, "ediffiqa_tiny_jun2024.onnx", Quality)
        self.liveness = self._optional(LIVENESS_URL, "MiniFASNetV2.onnx", Liveness)

    def describe(self) -> str:
        """What is actually loaded, for the screen. Not the pack's name:
        "buffalo_l" is five models, and this runs two of them."""
        parts = [str(RECOGNISERS[self.model]["label"]), "SCRFD-10G"]
        if self.quality:
            parts.append("eDifFIQA-T")
        if self.liveness:
            parts.append("MiniFASNetV2")
        return " · ".join(parts)

    def embed_image(self, image: np.ndarray) -> Optional[list[float]]:
        """The embedding of the largest face in a picture, or None.

        For rebuilding a library from the crops on disk when the model changes.
        """
        import cv2

        if self.detector is None or self.recogniser is None:
            self.load()
        if max(image.shape[:2]) < 640:
            scale = 640 / max(image.shape[:2])
            image = cv2.resize(image, None, fx=scale, fy=scale)
        found = self.detector.detect(image, 640)  # type: ignore[union-attr]
        if not found:
            return None
        face = max(found, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
        return self.recogniser.embed(align(image, np.array(face["landmarks"]))).tolist()  # type: ignore[union-attr]

    def faces(self, frame: np.ndarray, size: int = 320) -> list[Dict[str, Any]]:
        """Every face in the frame, with what the worker needs to judge it."""
        if self.detector is None or self.recogniser is None:
            self.load()
        found = self.detector.detect(frame, size)  # type: ignore[union-attr]
        for face in found:
            crop = align(frame, np.array(face["landmarks"]))
            face["embedding"] = self.recogniser.embed(crop).tolist()  # type: ignore[union-attr]
            face["quality"] = self.quality.score(crop) if self.quality else None
            face["live"] = self.liveness.real(frame, face["bbox"]) if self.liveness else None
        return found
