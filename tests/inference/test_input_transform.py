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
    Test that audio_to_tensors returns a list of tensors of the correct shape.
    """
    config = get_config("local")
    tensors = audio_to_tensors(str(dummy_audio_file), config)

    assert isinstance(tensors, list)
    assert len(tensors) > 0

    # Check the shape of the first tensor
    tensor = tensors[0]
    assert tensor.dim() == 3  # (1, freq, time)

    expected_time_frames = int(
        config.max_audio_length * config.sample_rate / config.hop_length
    )

    assert tensor.shape[1] == config.n_mels
    assert tensor.shape[2] == expected_time_frames
