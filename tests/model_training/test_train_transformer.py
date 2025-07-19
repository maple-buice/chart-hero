"""
Tests for the main training orchestration script train_transformer.py.
"""

from unittest.mock import MagicMock, patch

import pytest

from chart_hero.model_training.train_transformer import main as train_main
from chart_hero.model_training.transformer_config import get_config


@pytest.fixture
def config():
    """Returns a default config for testing."""
    return get_config("local")


@patch("chart_hero.model_training.train_transformer.create_data_loaders")
@patch("chart_hero.model_training.train_transformer.pl.Trainer")
@patch("chart_hero.model_training.train_transformer.configure_run")
def test_main_train_flow(mock_configure_run, mock_trainer, mock_create_data_loaders):
    """Test the main training flow orchestrator."""
    # Setup mocks
    mock_configure_run.return_value = (get_config("local"), False)
    mock_create_data_loaders.return_value = (MagicMock(), MagicMock(), MagicMock())
    mock_trainer_instance = MagicMock()
    mock_trainer.return_value = mock_trainer_instance

    # Mock sys.argv
    with patch("sys.argv", ["train_transformer.py", "--quick-test"]):
        train_main()

    # Assertions
    mock_configure_run.assert_called_once()
    mock_create_data_loaders.assert_called_once()
    mock_trainer.assert_called_once()
    mock_trainer_instance.fit.assert_called_once()
