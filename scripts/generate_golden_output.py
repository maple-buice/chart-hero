from pathlib import Path

import pandas as pd
import torch

from chart_hero.model_training.data_preparation import EGMDRawDataset
from chart_hero.model_training.transformer_config import get_config


def main() -> None:
    config = get_config("local")
    input_dir = Path("tests/assets/golden_input")
    output_dir = Path("tests/assets/golden_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_map_path = input_dir / "metadata.csv"
    if not data_map_path.exists():
        raise FileNotFoundError(
            f"Missing {data_map_path}. Run create_golden_input.py first."
        )

    dataset = EGMDRawDataset(
        data_map=pd.read_csv(data_map_path),
        dataset_dir=str(input_dir),
        config=config,
    )

    spectrogram, label_matrix = dataset[0]

    # Basic shape sanity checks
    assert spectrogram.dim() == 3 and spectrogram.size(0) == 1
    assert label_matrix.dim() == 2 and label_matrix.size(1) == config.num_drum_classes

    torch.save(spectrogram, str(output_dir / "golden_spectrogram.pt"))
    torch.save(label_matrix, str(output_dir / "golden_label_matrix.pt"))
    print(f"Wrote golden outputs to {output_dir}")


if __name__ == "__main__":
    main()
