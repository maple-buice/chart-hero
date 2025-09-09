import numpy as np
import soundfile as sf
import pytest

from chart_hero.inference.input_transform import audio_to_tensors
from chart_hero.inference.segment_utils import detect_leading_silence_from_segments
from chart_hero.model_training.transformer_config import get_config


def test_detect_leading_silence(tmp_path):
    sr = 22050
    silence = np.zeros(sr * 5)
    t = np.linspace(0, 1, sr, False)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio = np.concatenate([silence, tone])
    file_path = tmp_path / "silence_click.wav"
    sf.write(file_path, audio, sr)

    config = get_config("local")
    segments = audio_to_tensors(str(file_path), config)

    frames = detect_leading_silence_from_segments(segments)
    seconds = frames * config.hop_length / config.sample_rate
    assert seconds == pytest.approx(5.0, abs=0.1)
