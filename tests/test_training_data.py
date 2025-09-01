import torch

from chart_hero.model_training.transformer_config import get_config
from chart_hero.model_training.transformer_data import (
    compute_class_pos_weights,
    create_data_loaders,
)


def test_compute_class_pos_weights_dummy_dataset():
    config = get_config("local")
    data_dir = "tests/assets/dummy_data"
    pw = compute_class_pos_weights(
        data_dir, num_classes=config.num_drum_classes, split="train"
    )
    assert isinstance(pw, torch.Tensor)
    assert pw.shape[0] == config.num_drum_classes
    # ensure finite and positive
    assert torch.isfinite(pw).all()
    assert (pw > 0).all()


def test_create_data_loaders_shapes():
    config = get_config("local")
    data_dir = "tests/assets/dummy_data"
    train_loader, val_loader, test_loader = create_data_loaders(
        config, data_dir, batch_size=2
    )

    batch = next(iter(train_loader))
    specs, labels = batch
    # spectrograms: [B, 1, F, T]
    assert specs.dim() == 4 and specs.size(1) == 1 and specs.size(2) == config.n_mels
    # labels: [B, T, C]
    assert labels.dim() == 3 and labels.size(2) == config.num_drum_classes
