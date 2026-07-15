"""Plugin-local clap detection for Smart Room actions.

The microphone stays inside the Smart Room runtime. A cheap transient gate
selects short audio windows, then quantized YAMNet confirms that the event is
actually clapping. No speech-to-text or Marvi core audio hooks are involved.
"""

from __future__ import annotations

import hashlib
import logging
import os
import queue
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_MODEL_URL = (
    "https://raw.githubusercontent.com/tensorflow/tflite-support/master/"
    "tensorflow_lite_support/metadata/python/tests/testdata/audio_classifier/"
    "yamnet_wavin_quantized_mel_relu6.tflite"
)
_MODEL_SHA256 = "b8cc7ebd9edf3c16fcd79230e4c105316e8872c51ce447348176f75d6d35d570"
_MODEL_SAMPLES = 15_600
_MODEL_RATE = 16_000
_CLAPPING_CLASS = 58


class ClapSequence:
    """Convert confirmed claps into deliberate two/three-clap actions."""

    def __init__(
        self,
        on_action: Callable[[str], None],
        *,
        max_gap: float = 0.9,
        decision_delay: float = 0.65,
        cooldown: float = 3.0,
    ) -> None:
        self._on_action = on_action
        self._max_gap = max(0.2, max_gap)
        self._decision_delay = min(self._max_gap, max(0.2, decision_delay))
        self._cooldown = max(0.0, cooldown)
        self._claps: list[float] = []
        self._cooldown_until = 0.0

    def add(self, at: Optional[float] = None) -> bool:
        """Add one model-confirmed clap; return whether it was accepted."""
        now = time.monotonic() if at is None else at
        if now < self._cooldown_until:
            return False
        if self._claps and now - self._claps[-1] > self._max_gap:
            self._claps.clear()
        self._claps.append(now)
        if len(self._claps) >= 3:
            self._fire("sleep", now)
        return True

    def tick(self, now: Optional[float] = None) -> None:
        """Resolve a pending double clap or discard a lone clap."""
        current = time.monotonic() if now is None else now
        if len(self._claps) == 2 and current >= self._claps[-1] + self._decision_delay:
            self._fire("toggle_light", current)
        elif len(self._claps) == 1 and current >= self._claps[0] + self._max_gap:
            self._claps.clear()

    def _fire(self, action: str, now: float) -> None:
        self._claps.clear()
        self._cooldown_until = now + self._cooldown
        self._on_action(action)


class SoundEventListener:
    """Capture microphone audio and confirm clap sequences with YAMNet."""

    def __init__(self, config: Dict[str, Any], on_action: Callable[[str], None]):
        self._config = config or {}
        self._on_action = on_action
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stream: Any = None
        self._lock = threading.Lock()
        self._audio: Any = None
        self._write_at = 0
        self._filled = 0
        self._candidates: queue.Queue[tuple[float, float]] = queue.Queue(maxsize=8)
        self._last_candidate = 0.0
        self._noise_floor = 0.005
        self._status: Dict[str, Any] = {
            "enabled": bool(self._config.get("enabled", False)),
            "running": False,
            "model_ready": False,
            "microphone": None,
            "last_error": None,
            "last_score": None,
            "noise_floor": self._noise_floor,
            "candidates": 0,
            "confirmed_claps": 0,
            "last_action": None,
        }
        self._sequence = ClapSequence(
            self._dispatch_action,
            max_gap=float(self._config.get("max_gap_ms", 900)) / 1000,
            decision_delay=float(self._config.get("decision_ms", 650)) / 1000,
            cooldown=float(self._config.get("cooldown_ms", 3000)) / 1000,
        )

    def start(self) -> None:
        if not self._status["enabled"] or (self._thread and self._thread.is_alive()):
            return
        self._thread = threading.Thread(
            target=self._run, name="smart_room_sound_events", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        stream = self._stream
        if stream is not None:
            try:
                stream.abort()
                stream.close()
            except Exception:
                logger.debug("Sound input close failed", exc_info=True)
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=3)
        self._status["running"] = False

    def status(self) -> Dict[str, Any]:
        with self._lock:
            result = dict(self._status)
            result["noise_floor"] = round(self._noise_floor, 5)
            return result

    def _run(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd
            from ai_edge_litert.interpreter import Interpreter

            model_path = _ensure_model()
            interpreter = Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            input_index = interpreter.get_input_details()[0]["index"]
            output_index = interpreter.get_output_details()[0]["index"]

            device = self._config.get("input_device")
            device_info = sd.query_devices(device, "input")
            native_rate = int(round(float(device_info["default_samplerate"])))
            blocksize = max(256, int(native_rate * 0.1))
            self._audio = np.zeros(_MODEL_SAMPLES, dtype=np.float32)
            self._status.update(
                model_ready=True,
                microphone=str(device_info.get("name", device or "default")),
            )

            def callback(indata, frames, callback_time, status) -> None:
                if status:
                    logger.debug("Sound input status: %s", status)
                samples = np.asarray(indata[:, 0], dtype=np.float32)
                if native_rate != _MODEL_RATE:
                    target_count = max(1, round(len(samples) * _MODEL_RATE / native_rate))
                    samples = np.interp(
                        np.linspace(0, len(samples) - 1, target_count),
                        np.arange(len(samples)),
                        samples,
                    ).astype(np.float32)
                self._consume_block(samples, np)

            self._stream = sd.InputStream(
                device=device,
                channels=1,
                samplerate=native_rate,
                blocksize=blocksize,
                dtype="float32",
                callback=callback,
            )
            self._stream.start()
            self._status["running"] = True
            logger.info("Sound events listening on %s", self._status["microphone"])

            delay = float(self._config.get("model_delay_ms", 150)) / 1000
            pending: list[tuple[float, float]] = []
            while not self._stop.wait(0.05):
                try:
                    while True:
                        pending.append(self._candidates.get_nowait())
                except queue.Empty:
                    pass
                now = time.monotonic()
                due, pending = [item for item in pending if item[0] <= now], [
                    item for item in pending if item[0] > now
                ]
                for _, detected_at in due:
                    waveform = self._snapshot(np)
                    if waveform is None:
                        continue
                    interpreter.set_tensor(input_index, waveform)
                    interpreter.invoke()
                    scores = interpreter.get_tensor(output_index)
                    score = float(scores.reshape(-1)[_CLAPPING_CLASS])
                    self._status["last_score"] = round(score, 4)
                    if score >= float(self._config.get("confidence", 0.45)):
                        self._status["confirmed_claps"] += 1
                        self._sequence.add(detected_at + delay)
                self._sequence.tick(now)
        except Exception as exc:
            self._status["last_error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("Sound events disabled: %s", self._status["last_error"])
        finally:
            self._status["running"] = False
            stream = self._stream
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            self._stream = None

    def _consume_block(self, samples: Any, np: Any) -> None:
        count = min(len(samples), _MODEL_SAMPLES)
        samples = samples[-count:]
        with self._lock:
            first = min(count, _MODEL_SAMPLES - self._write_at)
            self._audio[self._write_at : self._write_at + first] = samples[:first]
            remainder = count - first
            if remainder:
                self._audio[:remainder] = samples[first:]
            self._write_at = (self._write_at + count) % _MODEL_SAMPLES
            self._filled = min(_MODEL_SAMPLES, self._filled + count)

        peak = float(np.max(np.abs(samples))) if count else 0.0
        rms = float(np.sqrt(np.mean(np.square(samples)))) if count else 0.0
        self._noise_floor = 0.995 * self._noise_floor + 0.005 * min(rms, 0.05)
        threshold = max(
            float(self._config.get("min_peak", 0.12)),
            self._noise_floor * float(self._config.get("noise_multiplier", 8.0)),
        )
        crest = peak / max(rms, 1e-6)
        now = time.monotonic()
        refractory = float(self._config.get("candidate_refractory_ms", 250)) / 1000
        if peak >= threshold and crest >= float(self._config.get("min_crest", 3.0)) and now - self._last_candidate >= refractory:
            self._last_candidate = now
            self._status["candidates"] += 1
            due = now + float(self._config.get("model_delay_ms", 150)) / 1000
            try:
                self._candidates.put_nowait((due, now))
            except queue.Full:
                pass

    def _snapshot(self, np: Any) -> Any:
        with self._lock:
            if self._filled < _MODEL_SAMPLES:
                return None
            return np.concatenate(
                (self._audio[self._write_at :], self._audio[: self._write_at])
            ).astype(np.float32, copy=False)

    def _dispatch_action(self, action: str) -> None:
        self._status["last_action"] = action
        try:
            self._on_action(action)
        except Exception:
            logger.exception("Sound action failed: %s", action)


def _ensure_model() -> Path:
    model_dir = Path(get_hermes_home()) / "smart_room" / "models"
    model_path = model_dir / "yamnet_clap_quantized.tflite"
    if model_path.is_file() and _sha256(model_path) == _MODEL_SHA256:
        return model_path

    model_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="yamnet-", suffix=".tflite", dir=model_dir)
    os.close(fd)
    temp_path = Path(temporary)
    try:
        with urllib.request.urlopen(_MODEL_URL, timeout=30) as response, temp_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        actual = _sha256(temp_path)
        if actual != _MODEL_SHA256:
            raise RuntimeError(f"YAMNet checksum mismatch: {actual}")
        os.replace(temp_path, model_path)
        return model_path
    finally:
        temp_path.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
