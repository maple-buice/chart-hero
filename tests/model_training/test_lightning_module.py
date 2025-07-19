"""
Tests for the DrumTranscriptionModule in lightning_module.py.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from chart_hero.model_training.lightning_module import DrumTranscriptionModule
from chart_hero.model_training.transformer_config import get_config


@pytest.fixture
def config():
    """Returns a default config for testing."""
    return get_config("local")


def test_initialization(config):
    """Test that the DrumTranscriptionModule initializes correctly."""
    module = DrumTranscriptionModule(config)
    assert module.config == config
    assert module.model is not None
    assert module.criterion is not None


def test_training_step(config):
    """Test a single training step."""
    config.max_seq_len = 512
    device = torch.device(config.device)
    max_time_patches = config.max_seq_len // config.patch_size[0]
    module = DrumTranscriptionModule(config, max_time_patches=max_time_patches).to(
        device
    )
    module.trainer = MagicMock()

    batch = (
        torch.randn(1, 1, config.n_mels, config.max_seq_len).to(device),
        torch.randint(0, 2, (1, config.max_seq_len, config.num_drum_classes))
        .float()
        .to(device),
    )

    with patch.object(module, "log") as mock_log:
        loss = module.training_step(batch, 0)
        assert loss is not None
        mock_log.assert_called()


def test_validation_step(config):
    """Test a single validation step."""
    config.max_seq_len = 512
    device = torch.device(config.device)
    max_time_patches = config.max_seq_len // config.patch_size[0]
    module = DrumTranscriptionModule(config, max_time_patches=max_time_patches).to(
        device
    )
    module.trainer = MagicMock()

    batch = (
        torch.randn(1, 1, config.n_mels, config.max_seq_len).to(device),
        torch.randint(0, 2, (1, config.max_seq_len, config.num_drum_classes))
        .float()
        .to(device),
    )

    with patch.object(module, "log") as mock_log:
        loss = module.validation_step(batch, 0)
        assert loss is not None
        mock_log.assert_called()


def test_configure_optimizers(config):
    """Test the optimizer and scheduler configuration."""
    module = DrumTranscriptionModule(config)
    module.trainer = MagicMock()
    optimizers, schedulers = module.configure_optimizers()
    assert len(optimizers) == 1
    assert len(schedulers) == 1
