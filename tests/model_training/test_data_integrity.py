import pandas as pd
import torch

from chart_hero.model_training.data_preparation import EGMDRawDataset
from chart_hero.model_training.transformer_config import get_config


def test_data_pipeline_integrity():
    """
    Test that the data preparation pipeline produces a consistent output.
    """
    # Load the golden input
    config = get_config("local")
    dataset = EGMDRawDataset(
        data_map=pd.read_csv("tests/assets/golden_input/metadata.csv"),
        dataset_dir="tests/assets/golden_input",
        config=config,
    )

    # Process the golden input
    spectrogram, label_matrix = dataset[0]

    # Basic integrity checks
    assert spectrogram.shape[1] == config.n_mels
    assert label_matrix.shape[1] == config.num_drum_classes
    assert not torch.isnan(spectrogram).any()
    assert not torch.isnan(label_matrix).any()
