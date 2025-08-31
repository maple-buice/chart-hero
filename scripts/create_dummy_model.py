import torch

from chart_hero.model_training.train_transformer import DrumTranscriptionModule
from chart_hero.model_training.transformer_config import get_config


def create_dummy_checkpoint(path="tests/assets/dummy_model.ckpt"):
    """Creates a dummy model checkpoint for testing."""
    from chart_hero.model_training.transformer_config import TARGET_CLASSES

    config = get_config("local")
    config.num_drum_classes = len(TARGET_CLASSES)
    model = DrumTranscriptionModule(config)
    torch.save(
        {"state_dict": model.state_dict(), "pytorch-lightning_version": "2.5.1.post0"},
        path,
    )


if __name__ == "__main__":
    create_dummy_checkpoint()
