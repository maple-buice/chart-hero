import logging
from pathlib import Path

import mido
import numpy as np
import pandas as pd
import pytest
import torch
import torchaudio

from chart_hero.model_training.data_preparation import main as prepare_egmd_data
from chart_hero.model_training.transformer_config import get_config
from chart_hero.model_training.transformer_data import create_data_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def dummy_processed_dataset(tmp_path: Path):
    """Creates a dummy processed dataset in a temporary directory."""
    config = get_config("local")
    processed_dir = tmp_path / "processed"

    for split in ["train", "val", "test"]:
        split_dir = processed_dir / split
        split_dir.mkdir(parents=True)
        for i in range(10):
            # Ensure all spectrograms and labels have the exact same dimensions
            spec_shape = (config.n_mels, config.max_seq_len)
            label_shape = (config.max_seq_len, config.num_drum_classes)

            dummy_spectrogram = np.random.rand(*spec_shape).astype(np.float32)
            dummy_labels = np.random.randint(0, 2, size=label_shape).astype(np.float32)

            np.save(split_dir / f"{split}_{i}_mel.npy", dummy_spectrogram)
            np.save(split_dir / f"{split}_{i}_label.npy", dummy_labels)

    return processed_dir, config


def test_data_loading(dummy_processed_dataset):
    """Test that data loading works without dimension errors."""
    processed_dir, config = dummy_processed_dataset

    # Test data loaders creation
    train_loader, val_loader, test_loader = create_data_loaders(
        config=config,
        data_dir=str(processed_dir),
    )

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

    # Test loading a few samples
    for i, (spectrogram, labels) in enumerate(train_loader):
        assert spectrogram is not None
        assert labels is not None
        # Check for correct dimensions
        assert spectrogram.shape == (
            config.train_batch_size,
            1,
            config.n_mels,
            config.max_seq_len,
        )
        assert labels.shape == (
            config.train_batch_size,
            config.max_seq_len,
            config.num_drum_classes,
        )
        if i >= 2:
            break
