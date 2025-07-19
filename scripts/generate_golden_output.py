import os

import pandas as pd
import torch
from chart_hero.model_training.data_preparation import EGMDRawDataset
from chart_hero.model_training.transformer_config import get_config

config = get_config("local")
dataset = EGMDRawDataset(
    data_map=pd.read_csv("tests/assets/golden_input/metadata.csv"),
    dataset_dir="tests/assets/golden_input",
    config=config,
)

spectrogram, label_matrix = dataset[0]

os.makedirs("tests/assets/golden_data", exist_ok=True)
torch.save(spectrogram, "tests/assets/golden_data/golden_spectrogram.pt")
torch.save(label_matrix, "tests/assets/golden_data/golden_label_matrix.pt")
