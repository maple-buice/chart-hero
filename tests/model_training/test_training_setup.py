"""
Tests for the training setup functions in training_setup.py.
"""

from unittest.mock import patch

import pytest
from chart_hero.model_training.training_setup import (
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
