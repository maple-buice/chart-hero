from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio

from chart_hero.inference.input_transform import drum_extraction, drum_to_frame


class MockDemucs(nn.Module):
    def __init__(self, sample_rate=44100, audio_channels=2):
        super().__init__()
        self.samplerate = sample_rate
        self.audio_channels = audio_channels
        self.sources = ["drums", "bass", "other", "vocals"]

    def forward(self, x):
        return torch.randn(1, 4, x.shape[-1])


def test_drum_extraction(tmp_path):
    """Test the drum_extraction function."""

    # Create a dummy audio file
    dummy_audio_path = tmp_path / "test.wav"
    sample_rate = 44100
    dummy_audio = torch.randn(1, sample_rate * 5)  # 5 seconds of audio
    torchaudio.save(dummy_audio_path, dummy_audio, sample_rate)

    # Mock the demucs separator
    with patch("demucs.apply.apply_model") as mock_apply_model:
        # Mock the return value of apply_model to be a dummy drum track
        mock_apply_model.return_value = torch.randn(1, 4, sample_rate * 5)

        with patch("demucs.pretrained.get_model") as mock_get_model:
            mock_model = MockDemucs(sample_rate=sample_rate)
            mock_get_model.return_value = mock_model

            drum_track, sr = drum_extraction(str(dummy_audio_path))

            assert drum_track is not None
            assert isinstance(drum_track, np.ndarray)
            assert sr == sample_rate


def test_drum_to_frame():
    """Test the drum_to_frame function."""

    # Create a dummy drum track
    sample_rate = 22050
    dummy_drum_track = np.random.randn(sample_rate * 10)  # 10 seconds of audio

    # Run the drum_to_frame function
    df, bpm = drum_to_frame(dummy_drum_track, sample_rate)

    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "audio_clip" in df.columns
    assert "sample_start" in df.columns
    assert "sample_end" in df.columns
    assert "sampling_rate" in df.columns
    assert "peak_sample" in df.columns

    assert bpm is not None
    assert isinstance(bpm, float)
