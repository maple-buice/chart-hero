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

    # Load the golden output
    golden_spectrogram = torch.load("tests/assets/golden_data/golden_spectrogram.pt")
    golden_label_matrix = torch.load("tests/assets/golden_data/golden_label_matrix.pt")

    # Compare the output with the golden dataset
    assert torch.allclose(spectrogram, golden_spectrogram, rtol=1e-4, atol=1e-6)
    assert torch.equal(label_matrix, golden_label_matrix)
