from unittest.mock import MagicMock, patch

import pytest
import torch

from chart_hero.model_training.train_transformer import DrumTranscriptionModule
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
    assert module.train_f1 is not None
    assert module.val_f1 is not None
    assert module.test_f1 is not None
    assert module.train_acc is not None
    assert module.val_acc is not None
    assert module.test_acc is not None


def test_training_step(config):
    """Test a single training step."""
    device = torch.device(config.device)
    module = DrumTranscriptionModule(config).to(device)
    module.trainer = MagicMock()

    batch = {
        "spectrogram": torch.randn(1, 1, config.n_mels, config.max_seq_len).to(device),
        "labels": torch.randint(0, 2, (1, config.num_drum_classes)).float().to(device),
    }

    with patch.object(module, "log") as mock_log:
        loss = module.training_step(batch, 0)
        assert loss is not None
        assert loss > 0
        mock_log.assert_called()


def test_validation_step(config):
    """Test a single validation step."""
    device = torch.device(config.device)
    module = DrumTranscriptionModule(config).to(device)
    module.trainer = MagicMock()

    batch = {
        "spectrogram": torch.randn(1, 1, config.n_mels, config.max_seq_len).to(device),
        "labels": torch.randint(0, 2, (1, config.num_drum_classes)).float().to(device),
    }

    with patch.object(module, "log") as mock_log:
        loss = module.validation_step(batch, 0)
        assert loss is not None
        assert loss > 0
        mock_log.assert_called()
        assert len(module.validation_step_outputs) == 1


def test_test_step(config):
    """Test a single test step."""
    device = torch.device(config.device)
    module = DrumTranscriptionModule(config).to(device)
    module.trainer = MagicMock()

    batch = {
        "spectrogram": torch.randn(1, 1, config.n_mels, config.max_seq_len).to(device),
        "labels": torch.randint(0, 2, (1, config.num_drum_classes)).float().to(device),
    }

    with patch.object(module, "log") as mock_log:
        loss = module.test_step(batch, 0)
        assert loss is not None
        assert loss > 0
        mock_log.assert_called()
        assert len(module.test_step_outputs) == 1


def test_configure_optimizers(config):
    """Test the optimizer and scheduler configuration."""
    module = DrumTranscriptionModule(config)
    module.trainer = MagicMock()
    optimizers = module.configure_optimizers()
    assert "optimizer" in optimizers
    assert "lr_scheduler" in optimizers


def test_setup_callbacks(config):
    """Test the setup_callbacks function."""
    from chart_hero.model_training.train_transformer import setup_callbacks

    callbacks = setup_callbacks(config)
    assert len(callbacks) == 3


@patch("chart_hero.model_training.train_transformer.WandbLogger")
def test_setup_logger(mock_wandb_logger, config):
    """Test the setup_logger function."""
    from chart_hero.model_training.train_transformer import setup_logger

    # Test with W&B enabled
    logger = setup_logger(config, use_wandb=True)
    assert logger is not None
    mock_wandb_logger.assert_called_once()

    # Test with W&B disabled
    mock_wandb_logger.reset_mock()
    logger = setup_logger(config, use_wandb=False)
    assert logger is None
    mock_wandb_logger.assert_not_called()


@patch("pytorch_lightning.Trainer")
@patch("chart_hero.model_training.train_transformer.create_data_loaders")
def test_train_model(mock_create_data_loaders, mock_trainer, config):
    """Test the train_model function."""
    from chart_hero.model_training.train_transformer import train_model

    # Mock the data loaders
    mock_create_data_loaders.return_value = (MagicMock(), MagicMock(), MagicMock())

    # Run the training function
    model, trainer = train_model(
        config,
        project_name="test_project",
        experiment_tag="test_tag",
        use_wandb_logging=False,
        monitor_gpu_usage=False,
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        test_loader=MagicMock(),
    )

    # Check that the trainer was called correctly
    mock_trainer.assert_called_once()
    trainer.fit.assert_called_once()
