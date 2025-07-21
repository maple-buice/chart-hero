"""
Tests for the training setup functions in training_setup.py.
"""

from unittest.mock import MagicMock, patch

import pytest
from chart_hero.model_training.training_setup import (
    configure_paths,
    setup_callbacks,
    setup_logger,
)
from chart_hero.model_training.transformer_config import get_config


@pytest.fixture
def config():
    """Returns a default config for testing."""
    return get_config("local")


def test_setup_callbacks(config):
    """Test the setup_callbacks function."""
    callbacks = setup_callbacks(config)
    assert len(callbacks) == 3


@patch("chart_hero.model_training.training_setup.WandbLogger")
def test_setup_logger(mock_wandb_logger, config):
    """Test the setup_logger function."""
    # Test with W&B enabled
    logger = setup_logger(config, "test_project", True, "test_tag")
    assert logger is not None
    mock_wandb_logger.assert_called_once()

    # Test with W&B disabled
    mock_wandb_logger.reset_mock()
    logger = setup_logger(config, "test_project", False, "test_tag")
    assert logger is None
    mock_wandb_logger.assert_not_called()


def test_configure_paths_handles_str_log_dir(tmp_path):
    """
    Tests that `configure_paths` correctly handles `log_dir` when it's a string
    and creates the directory.
    """
    # 1. Setup
    mock_config = {
        "log_dir": str(tmp_path / "test_logs"),
        "model_dir": str(tmp_path / "test_models"),
    }

    mock_args = {
        "data_dir": None,
        "audio_dir": None,
        "experiment_tag": "test_experiment",
    }

    # 2. Execute
    configure_paths(
        MagicMock(**mock_config), MagicMock(**mock_args)
    )  # Use MagicMock to simulate object access

    # 3. Assert
    log_dir_path = tmp_path / "test_logs"
    model_dir_path = tmp_path / "test_models" / "test_experiment"

    assert isinstance(mock_config["log_dir"], str)  # Original is unchanged
    assert log_dir_path.is_dir()
    assert model_dir_path.is_dir()
