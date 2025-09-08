from unittest.mock import patch

import pytest

from chart_hero.model_training.transformer_config import (
    CloudConfig,
    LocalConfig,
    MpsMixedConfig,
    auto_detect_config,
    get_config,
    validate_config,
)


def test_get_config():
    """Test the get_config function."""
    # Test getting the local config
    config = get_config("local")
    assert isinstance(config, LocalConfig)

    # Test getting the cloud config
    config = get_config("cloud")
    assert isinstance(config, CloudConfig)

    # Test that it raises an error for an unknown config type
    with pytest.raises(ValueError):
        get_config("unknown")


def test_get_config_mps_mixed():
    """Ensure the mps_mixed profile is available."""
    config = get_config("mps_mixed")
    assert isinstance(config, MpsMixedConfig)


@patch("torch.cuda.is_available", return_value=True)
def test_auto_detect_config_cuda(mock_is_available):
    """Test the auto_detect_config function when CUDA is available."""
    config = auto_detect_config()
    assert isinstance(config, CloudConfig)


@patch("torch.cuda.is_available", return_value=False)
@patch("torch.backends.mps.is_available", return_value=True)
def test_auto_detect_config_mps(mock_mps_is_available, mock_cuda_is_available):
    """Test the auto_detect_config function when MPS is available."""
    config = auto_detect_config()
    assert isinstance(config, LocalConfig)


@patch("torch.cuda.is_available", return_value=False)
@patch("torch.backends.mps.is_available", return_value=False)
def test_auto_detect_config_cpu(mock_mps_is_available, mock_cuda_is_available):
    """Test the auto_detect_config function when only CPU is available."""
    config = auto_detect_config()
    assert config.device == "cpu"


def test_validate_config():
    """Test the validate_config function."""
    # Test that it doesn't raise an error for a valid config
    config = get_config("local")
    validate_config(config)

    # Test that it raises an error for an invalid config
    config.hidden_size = 0
    with pytest.raises(AssertionError):
        validate_config(config)


# Test cases for the LocalConfig device selection logic
@patch("torch.backends.mps.is_available", return_value=False)
@patch("torch.cuda.is_available", return_value=False)
def test_local_config_device_cpu_only(mock_cuda, mock_mps):
    """Verify LocalConfig defaults to 'cpu' when no accelerator is available."""
    config = LocalConfig()
    assert config.device == "cpu"


@patch("torch.backends.mps.is_available", return_value=True)
@patch("torch.cuda.is_available", return_value=False)
def test_local_config_device_mps_selected(mock_cuda, mock_mps):
    """Verify LocalConfig selects 'mps' when it is available."""
    config = LocalConfig()
    assert config.device == "mps"


@patch("torch.backends.mps.is_available", return_value=False)
@patch("torch.cuda.is_available", return_value=True)
def test_local_config_device_cuda_selected(mock_cuda, mock_mps):
    """Verify LocalConfig selects 'cuda' when it is available and MPS is not."""
    config = LocalConfig()
    assert config.device == "cuda"


@patch("torch.backends.mps.is_available", return_value=True)
@patch("torch.cuda.is_available", return_value=True)
def test_local_config_device_mps_preferred_over_cuda(mock_cuda, mock_mps):
    """Verify LocalConfig prefers 'mps' when both MPS and CUDA are available."""
    config = LocalConfig()
    assert config.device == "mps"
