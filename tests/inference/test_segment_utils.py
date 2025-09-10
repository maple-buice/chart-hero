import numpy as np
import pytest

from chart_hero.inference.segment_utils import detect_leading_silence_from_segments
from chart_hero.model_training.transformer_config import get_config


def test_detect_leading_silence():
    config = get_config("local")
    fps = config.sample_rate / config.hop_length
    offset_frames = int(5 * fps)
    n_mels = config.n_mels

    seg_silence = {
        "spec": np.zeros((n_mels, offset_frames)),
        "start_frame": 0,
        "end_frame": offset_frames,
        "total_frames": offset_frames * 2,
    }
    seg_sound = {
        "spec": np.ones((n_mels, offset_frames)),
        "start_frame": offset_frames,
        "end_frame": offset_frames * 2,
        "total_frames": offset_frames * 2,
    }

    frames = detect_leading_silence_from_segments(
        [seg_silence, seg_sound], threshold_db=-20.0
    )
    seconds = frames * config.hop_length / config.sample_rate
    assert seconds == pytest.approx(5.0, abs=0.1)
