import numpy as np
import pytest

from chart_hero.inference.input_transform import audio_to_tensors
from chart_hero.model_training.transformer_config import get_config


@pytest.fixture
def dummy_audio_file(tmp_path):
    """Create a dummy audio file for testing."""
    sr = 22050
    duration = 15  # seconds
    dummy_audio = np.random.randn(sr * duration)
    file_path = tmp_path / "dummy_audio.wav"
    import soundfile as sf

    sf.write(file_path, dummy_audio, sr)
    return file_path


def test_audio_to_tensors(dummy_audio_file):
    """
    Test that audio_to_tensors returns a list of segment dicts with spectrograms and offsets.
    """
    config = get_config("local")
    segments = audio_to_tensors(str(dummy_audio_file), config)

    assert isinstance(segments, list)
    assert len(segments) > 0

    # Check the first segment fields
    seg = segments[0]
    assert isinstance(seg, dict)
    assert "spec" in seg and "start_frame" in seg and "end_frame" in seg
    spec = seg["spec"]

    # Expect shape (n_mels, frames)
    assert spec.shape[0] == config.n_mels
    expected_time_frames = int(
        config.max_audio_length * config.sample_rate / config.hop_length
    )
    assert spec.shape[1] == expected_time_frames
