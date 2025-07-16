"""
Test script to verify the data loading fixes work correctly.
"""

# Set up logging
import logging
import os
import sys
from unittest.mock import patch

import mido
import pandas as pd
import torch
import torchaudio

from chart_hero.model_training.transformer_config import get_config
from chart_hero.model_training.transformer_data import create_data_loaders
from chart_hero.prepare_egmd_data import main as prepare_egmd_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_data_loading():
    """Test that data loading works without dimension errors."""

    # Get local config
    config = get_config("local")

    # Create dummy data for testing
    dummy_data_dir = "tests/assets/dummy_data"
    os.makedirs(dummy_data_dir, exist_ok=True)
    dummy_audio_dir = os.path.join(dummy_data_dir, "audio")
    os.makedirs(dummy_audio_dir, exist_ok=True)

    # Create dummy metadata file
    dummy_metadata = {
        "audio_filename": [f"audio/dummy_{i}.wav" for i in range(10)],
        "midi_filename": [f"dummy_{i}.mid" for i in range(10)],
        "split": ["train"] * 8 + ["val"] * 1 + ["test"] * 1,
    }
    pd.DataFrame(dummy_metadata).to_csv(
        os.path.join(dummy_data_dir, "metadata.csv"), index=False
    )

    # Create dummy audio and midi files
    for i in range(10):
        dummy_audio_path = os.path.join(dummy_audio_dir, f"dummy_{i}.wav")
        dummy_audio = torch.randn(1, int(config.sample_rate * 5))
        torchaudio.save(dummy_audio_path, dummy_audio, config.sample_rate)

        dummy_midi_path = os.path.join(dummy_data_dir, f"dummy_{i}.mid")
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.Message("note_on", note=60, velocity=64, time=32))
        track.append(mido.Message("note_off", note=60, velocity=127, time=32))
        mid.save(dummy_midi_path)

    # Run data preparation
    with patch.object(
        sys,
        "argv",
        [
            "prepare_egmd_data",
            "--input-dir",
            dummy_data_dir,
            "--output-dir",
            dummy_data_dir,
            "--splits",
            "0.8",
            "0.1",
            "0.1",
        ],
    ):
        prepare_egmd_data()

    # Test data loaders creation
    train_loader, val_loader, test_loader = create_data_loaders(
        config=config,
        data_dir=dummy_data_dir,
    )

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

    # Test loading a few samples
    for i, batch in enumerate(train_loader):
        assert "spectrogram" in batch
        assert "labels" in batch
        if i >= 2:
            break
