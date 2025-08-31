from pathlib import Path
from unittest.mock import patch

import numpy as np

from chart_hero.model_training.transformer_config import get_config
from chart_hero.model_training.transformer_data import create_data_loaders


@patch("numpy.load")
def test_data_preparation(mock_np_load):
    """Test that the data preparation pipeline works correctly."""
    # Create dummy data
    config = get_config("local")
    dummy_spectrograms = np.random.randn(
        10, 1, config.n_mels, config.max_seq_len
    ).astype(np.float32)
    dummy_labels = np.random.randint(0, 2, (10, config.num_drum_classes)).astype(
        np.int8
    )

    # Mock np.load to return the dummy data
    def mock_load(path):
        if "mel" in path:
            return dummy_spectrograms
        elif "label" in path:
            return dummy_labels
        return None

    mock_np_load.side_effect = mock_load

    # Create a dummy directory structure
    with (
        patch("pathlib.Path.glob") as mock_glob,
        patch("pathlib.Path.exists") as mock_exists,
    ):
        mock_glob.return_value = [
            Path("train_mel.npy"),
            Path("val_mel.npy"),
            Path("test_mel.npy"),
        ]
        mock_exists.return_value = True

        # Test data loaders creation
        train_loader, val_loader, test_loader = create_data_loaders(
            config=config,
            data_dir="/dummy_data",
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
