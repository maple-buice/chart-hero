#!/usr/bin/env python3
"""
Main script to prepare E-GMD dataset for transformer training.

This script now acts as a simple wrapper around the new, streamlined
data preparation pipeline in `model_training.prepare_transformer_data`.
"""

import argparse
import logging
import os
import sys

import pandas as pd
import torch

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch.utils.data import DataLoader

from chart_hero.model_training.data_preparation import EGMDRawDataset
from chart_hero.model_training.transformer_config import get_config

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Prepare E-GMD dataset for transformer training using the new pipeline."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="datasets/e-gmd-v1.0.0",
        help="Path to E-GMD dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/processed_transformer",
        help="Output directory for processed .npy data",
    )
    parser.add_argument(
        "--splits",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
        help="Train, validation, and test split ratios (e.g., 0.8 0.1 0.1)",
    )

    args = parser.parse_args(args)

    torch.manual_seed(42)

    config = get_config("local")

    # Create the dataset
    dataset = EGMDRawDataset(
        data_map=pd.read_csv(os.path.join(args.input_dir, "metadata.csv")),
        dataset_dir=args.input_dir,
        config=config,
    )

    # Split the dataset
    train_size = int(args.splits[0] * len(dataset))
    val_size = int(args.splits[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Save the data
    os.makedirs(args.output_dir, exist_ok=True)

    def save_data(loader, name):
        spectrograms = []
        labels = []
        for spec, label in loader:
            spectrograms.append(spec)
            labels.append(label)
        torch.save(
            torch.cat(spectrograms), os.path.join(args.output_dir, f"{name}_mel.npy")
        )
        torch.save(
            torch.cat(labels), os.path.join(args.output_dir, f"{name}_label.npy")
        )

    save_data(train_loader, "train")
    save_data(val_loader, "val")
    save_data(test_loader, "test")


if __name__ == "__main__":
    main()
