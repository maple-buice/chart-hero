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


@patch("chart_hero.model_training.train_transformer.create_data_loaders")
@patch("chart_hero.model_training.train_transformer.pl.Trainer")
@patch("chart_hero.model_training.train_transformer.compute_class_pos_weights")
@patch("chart_hero.model_training.train_transformer.configure_run")
def test_main_calls_pos_weight_when_not_quick_test(
    mock_configure_run,
    mock_compute_class_pos_weights,
    mock_trainer,
    mock_create_data_loaders,
    tmp_path,
):
    cfg = get_config("local")
    cfg.data_dir = str((tmp_path / "dummy_data").resolve())
    cfg.model_dir = str((tmp_path / "models").resolve())
    # Create a minimal train split dir to pass the existence check
    train_dir = tmp_path / "dummy_data" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    # Create one dummy label file (empty but valid shape when attempted to be loaded by the mocked function)
    (train_dir / "train_0_label.npy").write_bytes(
        b"\x93NUMPY\x01\x00\xf2\x10\x00\x00{'descr': '<f8', 'fortran_order': False, 'shape': (0, 0), }\n "
    )

    mock_configure_run.return_value = (cfg, False)
    mock_create_data_loaders.return_value = (MagicMock(), MagicMock(), MagicMock())
    mock_trainer.return_value = MagicMock()
    import torch

    mock_compute_class_pos_weights.return_value = torch.ones(
        get_config("local").num_drum_classes
    )

    with patch("sys.argv", ["train_transformer.py"]):
        train_main()

    assert mock_compute_class_pos_weights.called, (
        "pos_weight should be computed when not quick-test"
    )


@patch("chart_hero.model_training.train_transformer.create_data_loaders")
@patch("chart_hero.model_training.train_transformer.pl.Trainer")
@patch("chart_hero.model_training.train_transformer.compute_class_pos_weights")
@patch("chart_hero.model_training.train_transformer.configure_run")
def test_main_skips_pos_weight_on_quick_test(
    mock_configure_run,
    mock_compute_class_pos_weights,
    mock_trainer,
    mock_create_data_loaders,
    tmp_path,
):
    cfg = get_config("local")
    cfg.data_dir = str((tmp_path / "dummy_data").resolve())
    cfg.model_dir = str((tmp_path / "models").resolve())
    train_dir = tmp_path / "dummy_data" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "train_0_label.npy").write_bytes(
        b"\x93NUMPY\x01\x00\xf2\x10\x00\x00{'descr': '<f8', 'fortran_order': False, 'shape': (0, 0), }\n "
    )

    mock_configure_run.return_value = (cfg, False)
    mock_create_data_loaders.return_value = (MagicMock(), MagicMock(), MagicMock())
    mock_trainer.return_value = MagicMock()

    with patch("sys.argv", ["train_transformer.py", "--quick-test"]):
        train_main()

    mock_compute_class_pos_weights.assert_not_called()
