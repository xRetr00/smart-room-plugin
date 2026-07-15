from __future__ import annotations

import pytest

from plugins.smart_room.runtime import sound_events
from plugins.smart_room.runtime.app import Runtime
from plugins.smart_room.runtime.sound_events import ClapSequence


def test_single_clap_is_ignored_and_double_clap_toggles() -> None:
    actions = []
    sequence = ClapSequence(actions.append, max_gap=0.9, decision_delay=0.6, cooldown=3)

    sequence.add(1.0)
    sequence.tick(1.91)
    assert actions == []

    sequence.add(3.0)
    sequence.add(3.4)
    sequence.tick(4.01)
    assert actions == ["toggle_light"]


def test_triple_clap_enters_sleep_and_cooldown_suppresses_retrigger() -> None:
    actions = []
    sequence = ClapSequence(actions.append, max_gap=0.9, decision_delay=0.7, cooldown=3)

    assert sequence.add(1.0)
    assert sequence.add(1.3)
    assert sequence.add(1.6)
    assert actions == ["sleep"]
    assert not sequence.add(2.0)
    sequence.tick(5.0)
    assert actions == ["sleep"]


def test_bad_model_download_is_rejected(monkeypatch, tmp_path) -> None:
    source = tmp_path / "not-yamnet.tflite"
    source.write_bytes(b"not the pinned model")
    monkeypatch.setattr(sound_events, "get_hermes_home", lambda: tmp_path / "profile")
    monkeypatch.setattr(sound_events, "_MODEL_URL", source.as_uri())

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        sound_events._ensure_model()

    assert not (tmp_path / "profile/smart_room/models/yamnet_clap_quantized.tflite").exists()


def test_sound_actions_use_room_state_and_tuya_status() -> None:
    class Tuya:
        def __init__(self) -> None:
            self.commands = []

        def get_light_status(self):
            return {"success": True, "on": True}

        def set_light(self, **kwargs):
            self.commands.append(kwargs)
            return {"success": True}

    runtime = Runtime({"sound_events": {"require_occupancy": True}})
    runtime._tuya = Tuya()
    runtime._state.mmwave.occupied = True

    runtime._on_sound_action("toggle_light")
    assert runtime._tuya.commands[-1]["on"] is False

    runtime._on_sound_action("sleep")
    assert runtime._state.modes.sleep is True
    assert runtime._tuya.commands[-1]["on"] is False
