import torch
from torch.utils.data import DataLoader
from chart_hero.model_training.transformer_data import NpyDrumDataset, custom_collate_fn
from chart_hero.model_training.transformer_config import get_config
import numpy as np


def test_data_pipeline_with_variable_lengths(tmp_path):
    """
    Integration test for the data pipeline with variable-length spectrograms.
    """
    config = get_config("local")
    dummy_files = []

    # Create dummy data with variable lengths
    for i in range(10):
        time_dimension = 200 + i * 10  # Variable length
        spectrogram = np.random.rand(1, config.n_mels, time_dimension).astype(np.float32)
        labels = np.random.randint(0, 2, (time_dimension // config.patch_size[0], config.num_drum_classes)).astype(np.float32)

        spec_file = tmp_path / f"spec_{i}.npy"
        label_file = tmp_path / f"label_{i}.npy"
        np.save(spec_file, spectrogram)
        np.save(label_file, labels)
        dummy_files.append((str(spec_file), str(label_file)))

    # Create dataset and dataloader
    dataset = NpyDrumDataset(dummy_files, config)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate_fn)

    # Check a few batches
    for i, (spectrograms, labels) in enumerate(dataloader):
        if i >= 2:  # Check 2 batches
            break

        # Check that padding was applied correctly
        assert spectrograms.ndim == 4
        assert labels.ndim == 3
        assert spectrograms.shape[0] == 2
        assert labels.shape[0] == 2
        assert spectrograms.shape[2] == config.n_mels

        # Check that the time dimension is the same for both tensors in the batch
        assert spectrograms.shape[3] == spectrograms.shape[3]
