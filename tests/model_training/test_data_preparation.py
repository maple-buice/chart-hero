from pathlib import Path

import numpy as np

from chart_hero.model_training.transformer_config import get_config
from chart_hero.model_training.transformer_data import create_data_loaders


def test_data_preparation(tmp_path: Path):
    """Integration-style test using real files on disk."""
    config = get_config("local")

    root = tmp_path / "processed"
    for split in ["train", "val", "test"]:
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            spec = np.random.randn(config.n_mels, config.max_seq_len).astype(np.float32)
            labels = np.random.randint(
                0, 2, (config.max_seq_len, config.num_drum_classes)
            ).astype(np.float32)
            np.save(split_dir / f"{split}_{i}_mel.npy", spec)
            np.save(split_dir / f"{split}_{i}_label.npy", labels)

    train_loader, val_loader, test_loader = create_data_loaders(
        config=config, data_dir=str(root)
    )

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

    # Inspect a couple of batches
    for i, (specs, labels) in enumerate(train_loader):
        assert specs.dim() == 4 and specs.size(1) == 1
        assert specs.size(2) == config.n_mels and specs.size(3) == config.max_seq_len
        assert labels.dim() == 3 and labels.size(1) == config.max_seq_len
        assert labels.size(2) == config.num_drum_classes
        if i >= 1:
            break
