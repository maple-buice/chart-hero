"""
Test script to verify transformer setup and basic functionality.
"""

import logging
import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from chart_hero.model_training.train_transformer import DrumTranscriptionModule
from chart_hero.model_training.transformer_config import (
    auto_detect_config,
    get_config,
    validate_config,
)
from chart_hero.model_training.transformer_data import (
    SpectrogramProcessor,
    create_data_loaders,
)
from chart_hero.model_training.transformer_model import (
    PatchEmbedding,
    PositionalEncoding2D,
    create_model,
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("config_name", ["local", "cloud", "auto"])
def test_config_loading_and_validation(config_name):
    """
    Test that configurations load and validate without errors.
    """
    if config_name == "auto":
        config = auto_detect_config()
    else:
        config = get_config(config_name)

    assert config is not None, f"Failed to load config: {config_name}"
    validate_config(config)


def test_model_creation():
    """
    Test that the model can be created with a valid configuration.
    """
    config = get_config("local")
    model = create_model(config)
    assert model is not None, "Model creation failed"


def test_data_loading():
    """
    Test that the data loaders can be created.
    """
    config = get_config("local")
    # Create dummy data for testing
    dummy_data_dir = "tests/assets/dummy_data"
    os.makedirs(dummy_data_dir, exist_ok=True)
    # Create dummy metadata file
    dummy_metadata = {
        "audio_filename": ["dummy.wav"],
        "midi_filename": ["dummy.mid"],
        "split": ["train"],
    }
    pd.DataFrame(dummy_metadata).to_csv(
        os.path.join(dummy_data_dir, "metadata.csv"), index=False
    )

    train_loader, val_loader, test_loader = create_data_loaders(
        config, data_dir=dummy_data_dir
    )
    assert train_loader is not None, "Train loader creation failed"
    assert val_loader is not None, "Validation loader creation failed"
    assert test_loader is not None, "Test loader creation failed"


@patch("pytorch_lightning.Trainer")
def test_training_module(mock_trainer):
    """
    Test that the training module can be initialized and a training step can be run.
    """
    config = get_config("local")
    device = torch.device(config.device)
    model = DrumTranscriptionModule(config).to(device)
    model.trainer = mock_trainer
    assert model is not None, "Training module creation failed"

    # Create a dummy batch
    dummy_spectrogram = torch.randn(1, 1, config.n_mels, config.max_seq_len).to(device)
    dummy_labels = torch.randint(0, 2, (1, config.num_drum_classes)).to(device).float()
    dummy_batch = {"spectrogram": dummy_spectrogram, "labels": dummy_labels}

    # Test training step
    loss = model.training_step(dummy_batch, 0)
    assert loss is not None, "Training step failed"


def test_config():
    """Test configuration classes."""
    logger.info("Testing configuration classes...")
    overall_test_success = True
    invoked_config_name = "auto"

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    # 1. Test the invoked configuration (this one MUST pass)
    logger.info(f"--- Validating invoked config: '{invoked_config_name}' ---")
    try:
        # If invoked is 'auto', resolve it first for the primary test
        if invoked_config_name == "auto":
            cfg_invoked = auto_detect_config()
            logger.info(
                f"Auto-detected config for invoked test: {type(cfg_invoked).__name__}"
            )
        else:
            cfg_invoked = get_config(invoked_config_name)
        validate_config(cfg_invoked)  # This must pass
        logger.info(
            f"✓ Invoked config '{invoked_config_name}' (resolved to {type(cfg_invoked).__name__ if invoked_config_name == 'auto' else invoked_config_name}): VALIDATED - device {cfg_invoked.device}, batch_size={cfg_invoked.train_batch_size}"
        )
    except Exception as e:
        logger.error(f"✗ Invoked config '{invoked_config_name}' FAILED validation: {e}")
        assert False  # Critical failure

    # 2. Test other standard configurations for completeness, with conditional MPS check
    standard_configs_to_check = {"local", "cloud", "auto"}
    # Determine the actual resolved name of the invoked config if it was 'auto'
    if invoked_config_name == "auto":
        # This is a bit tricky as auto_detect_config() returns an instance, not a name.
        # For simplicity in exclusion, we might just always test 'local' and 'cloud' if 'auto' was invoked,
        # or refine this if a specific name is needed.
        # For now, let's assume 'auto' detection might resolve to 'local' or 'cloud' (or others).
        # We will test all explicit standard configs not *literally* matching invoked_config_name.
        other_configs_to_verify = standard_configs_to_check - {invoked_config_name}
    else:
        other_configs_to_verify = standard_configs_to_check - {invoked_config_name}

    for cfg_name_to_check in other_configs_to_verify:
        logger.info(f"--- Checking standard config: '{cfg_name_to_check}' ---")
        try:
            if cfg_name_to_check == "auto":
                cfg = auto_detect_config()
                logger.info(
                    f"Auto-detected config for verification: {type(cfg).__name__}"
                )
            else:
                cfg = get_config(cfg_name_to_check)

            if cfg.device == "mps" and not mps_available:
                logger.warning(
                    f"✓ Config '{cfg_name_to_check}' (device: {cfg.device}) specifies MPS, but MPS not available. Loaded but skipping full device validation for this non-invoked config."
                )
            else:
                validate_config(cfg)
                logger.info(
                    f"✓ Config '{cfg_name_to_check}': Loaded & Validated - device {cfg.device}, batch_size={cfg.train_batch_size}"
                )
        except Exception as e:
            logger.error(
                f"✗ Config '{cfg_name_to_check}' (non-invoked) test generated an error: {e}"
            )
            overall_test_success = False

    assert overall_test_success


def test_model():
    """Test model instantiation and forward pass."""
    logger.info("Testing model architecture...")
    config_name = "auto"

    if config_name == "auto":
        config = auto_detect_config()
        logger.info(f"Auto-detected config for model test: {type(config).__name__}")
    else:
        config = get_config(config_name)  # Use passed config_name
    model = create_model(config)

    # Test forward pass
    batch_size = 2
    # Adjusted time_frames to be divisible by patch_size_t for ViT compatibility
    # Example: if patch_size_t is 16, time_frames could be 256 (16*16)
    # Ensure config.patch_size_t is defined and used if model expects it
    patch_size_t = (
        config.patch_size[0]
        if isinstance(config.patch_size, tuple)
        else config.patch_size_t
    )
    patch_size_f = (
        config.patch_size[1]
        if isinstance(config.patch_size, tuple)
        else config.patch_size_f
    )

    time_frames = patch_size_t * 16
    freq_bins = patch_size_f * 8

    dummy_input = torch.randn(batch_size, 1, time_frames, freq_bins)

    with torch.no_grad():
        output = model(dummy_input, return_embeddings=True)

    logger.info(f"✓ Input shape: {dummy_input.shape}")
    logger.info(f"✓ Output logits shape: {output['logits'].shape}")
    logger.info(f"✓ CLS embedding shape: {output['cls_embedding'].shape}")

    # Check output dimensions
    assert output["logits"].shape == (batch_size, config.num_drum_classes)
    assert output["cls_embedding"].shape == (batch_size, config.hidden_size)

    assert True


def test_spectrogram_processor():
    """Test the SpectrogramProcessor class."""
    config = get_config("local")
    processor = SpectrogramProcessor(config)
    dummy_audio = torch.randn(1, int(config.max_audio_length * config.sample_rate))
    spectrogram = processor.audio_to_spectrogram(dummy_audio)
    assert spectrogram.shape == (1, 216, 128)


def test_data_processing():
    """Test data processing pipeline."""
    config = get_config("local")
    processor = SpectrogramProcessor(config)

    # Test spectrogram processing
    dummy_audio = torch.randn(1, int(config.max_audio_length * config.sample_rate))
    spectrogram = processor.audio_to_spectrogram(dummy_audio)
    assert len(spectrogram.shape) == 3
    assert spectrogram.shape[0] == 1  # Channels

    # Test patch preparation
    padded_spec, patch_shape = processor.prepare_patches(spectrogram)
    assert len(padded_spec.shape) == 3
    assert patch_shape is not None


def test_patch_embedding():
    """Test the PatchEmbedding class."""
    config = get_config("local")
    patch_embed = PatchEmbedding(
        patch_size=config.patch_size,
        in_channels=1,
        embed_dim=config.hidden_size,
    )
    dummy_spectrogram = torch.randn(1, 1, 256, 128)
    patch_embeddings = patch_embed(dummy_spectrogram)
    assert patch_embeddings.shape == (1, 128, config.hidden_size)


def test_positional_encoding():
    """Test the PositionalEncoding2D class."""
    config = get_config("local")
    pos_encoding = PositionalEncoding2D(embed_dim=config.hidden_size)
    dummy_embeddings = torch.randn(1, 128, config.hidden_size)
    patch_shape = (16, 8)
    encoded_embeddings = pos_encoding(dummy_embeddings, patch_shape)
    assert encoded_embeddings.shape == (1, 129, config.hidden_size)  # +1 for CLS token
