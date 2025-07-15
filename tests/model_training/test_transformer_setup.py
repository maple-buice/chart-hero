"""
Test script to verify transformer setup and basic functionality.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import sys
import os
import argparse # Added
from unittest.mock import MagicMock # Added
from chart_hero.model_training.transformer_config import get_config, auto_detect_config, validate_config
from chart_hero.model_training.transformer_model import create_model
from chart_hero.model_training.transformer_data import NpyDrumDataset, SpectrogramProcessor
from chart_hero.model_training.train_transformer import DrumTranscriptionModule

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            logger.info(f"Auto-detected config for invoked test: {type(cfg_invoked).__name__}")
        else:
            cfg_invoked = get_config(invoked_config_name)
        validate_config(cfg_invoked) # This must pass
        logger.info(f"✓ Invoked config '{invoked_config_name}' (resolved to {type(cfg_invoked).__name__ if invoked_config_name == 'auto' else invoked_config_name}): VALIDATED - device {cfg_invoked.device}, batch_size={cfg_invoked.train_batch_size}")
    except Exception as e:
        logger.error(f"✗ Invoked config '{invoked_config_name}' FAILED validation: {e}")
        assert False # Critical failure

    # 2. Test other standard configurations for completeness, with conditional MPS check
    standard_configs_to_check = {"local", "cloud", "auto"} 
    # Determine the actual resolved name of the invoked config if it was 'auto'
    resolved_invoked_name = None
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
                logger.info(f"Auto-detected config for verification: {type(cfg).__name__}")
            else:
                cfg = get_config(cfg_name_to_check)
            
            if cfg.device == "mps" and not mps_available:
                logger.warning(f"✓ Config '{cfg_name_to_check}' (device: {cfg.device}) specifies MPS, but MPS not available. Loaded but skipping full device validation for this non-invoked config.")
            else:
                validate_config(cfg) 
                logger.info(f"✓ Config '{cfg_name_to_check}': Loaded & Validated - device {cfg.device}, batch_size={cfg.train_batch_size}")
        except Exception as e:
            logger.error(f"✗ Config '{cfg_name_to_check}' (non-invoked) test generated an error: {e}")
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
        config = get_config(config_name) # Use passed config_name
    model = create_model(config)
        
    # Test forward pass
    batch_size = 2
    # Adjusted time_frames to be divisible by patch_size_t for ViT compatibility
    # Example: if patch_size_t is 16, time_frames could be 256 (16*16)
    # Ensure config.patch_size_t is defined and used if model expects it
    patch_size_t = config.patch_size[0] if isinstance(config.patch_size, tuple) else config.patch_size_t
    patch_size_f = config.patch_size[1] if isinstance(config.patch_size, tuple) else config.patch_size_f
        
    time_frames = patch_size_t * 16 
    freq_bins = patch_size_f * 8
        
    dummy_input = torch.randn(batch_size, 1, time_frames, freq_bins)
        
    with torch.no_grad():
        output = model(dummy_input, return_embeddings=True)
        
    logger.info(f"✓ Input shape: {dummy_input.shape}")
    logger.info(f"✓ Output logits shape: {output['logits'].shape}")
    logger.info(f"✓ CLS embedding shape: {output['cls_embedding'].shape}")
        
    # Check output dimensions
    assert output['logits'].shape == (batch_size, config.num_drum_classes)
    assert output['cls_embedding'].shape == (batch_size, config.hidden_size)
        
    assert True


def test_data_processing():
    """Test data processing pipeline."""
    logger.info("Testing data processing...")
    config_name = "auto"
    
    if config_name == "auto":
        config = auto_detect_config()
        logger.info(f"Auto-detected config for data processing test: {type(config).__name__}")
    else:
        config = get_config(config_name) # Use passed config_name
    processor = SpectrogramProcessor(config)
        
    # Test spectrogram processing
    dummy_audio = torch.randn(1, int(config.max_audio_length * config.sample_rate))
    spectrogram = processor.audio_to_spectrogram(dummy_audio)
        
    logger.info(f"✓ Audio shape: {dummy_audio.shape}")
    logger.info(f"✓ Spectrogram shape: {spectrogram.shape}")
        
    # Test patch preparation
    padded_spec, patch_shape = processor.prepare_patches(spectrogram)
    logger.info(f"✓ Padded spectrogram shape: {padded_spec.shape}")
    logger.info(f"✓ Patch shape: {patch_shape}")
        
    # Test dataset creation with dummy data
    temp_data_dir = Path("/tmp/chart_hero_test_data")
    temp_data_dir.mkdir(parents=True, exist_ok=True)
        
    # Create dummy .npy files
    dummy_spectrograms = np.random.randn(10, 1, 256, 128).astype(np.float32)
    dummy_labels = np.random.randint(0, 2, (10, 9)).astype(np.int8)
        
    train_spec_path = temp_data_dir / "train_mel.npy"
    train_label_path = temp_data_dir / "train_label.npy"
        
    np.save(train_spec_path, dummy_spectrograms)
    np.save(train_label_path, dummy_labels)
        
    data_files = [(str(train_spec_path), str(train_label_path))]
        
    dataset = NpyDrumDataset(data_files, config, mode='test', augment=False)
    logger.info(f"✓ Dataset created with {len(dataset)} samples")
        
    # Test sample loading
    if len(dataset) > 0:
        sample = dataset[0]
        logger.info(f"✓ Sample spectrogram shape: {sample['spectrogram'].shape}")
        logger.info(f"✓ Sample labels shape: {sample['labels'].shape}")
    else:
        logger.warning("Dataset is empty, cannot test sample loading.")
        # This might be an error depending on expectations

        

    assert True


def test_training_module():
    """Test PyTorch Lightning training module."""
    logger.info("Testing training module...")
    config_name = "auto"
    
    if config_name == "auto":
        config = auto_detect_config()
        logger.info(f"Auto-detected config for training module test: {type(config).__name__}")
    else:
        config = get_config(config_name) # Use passed config_name
    module = DrumTranscriptionModule(config)

    # --- Mock Trainer for self.log() and other trainer-dependent calls ---
    mock_trainer = MagicMock()
    # Mock logger_connector and its log_metrics method
    mock_trainer.logger_connector = MagicMock()
    mock_trainer.logger_connector.log_metrics = MagicMock(return_value=None)
    # Mock other commonly used trainer attributes
    mock_trainer.current_epoch = 0
    mock_trainer.global_step = 0
    mock_trainer.model = module 
    mock_trainer.device = torch.device(config.device) 
    mock_trainer.world_size = 1 
    mock_trainer.local_rank = 0 
    mock_trainer.training = True # For calls that check trainer.training state
    mock_trainer.datamodule = None # If module interacts with datamodule
    mock_trainer.precision_plugin = MagicMock() # If precision related calls are made
    mock_trainer.precision_plugin.convert_input = lambda x: x # No-op conversion
    mock_trainer.precision_plugin.convert_output = lambda x: x # No-op conversion
    mock_trainer.accelerator_connector = MagicMock(use_ddp=False, use_dp=False) # For distributed checks
    mock_trainer.lightning_module = module # For self.trainer.lightning_module references
    # Add any other specific attributes your module might access from self.trainer
    module.trainer = mock_trainer
    # --- End Mock Trainer ---
        
    # Test forward pass (already part of training/validation step implicitly)
    batch_size = config.train_batch_size
    # Use patch_size from config for time_frames and freq_bins
    patch_size_t = config.patch_size[0] if isinstance(config.patch_size, tuple) else config.patch_size_t
    patch_size_f = config.patch_size[1] if isinstance(config.patch_size, tuple) else config.patch_size_f

    time_frames = patch_size_t * 16 
    freq_bins = patch_size_f * 8  
        
    dummy_batch = {
        'spectrogram': torch.randn(batch_size, 1, time_frames, freq_bins).to(config.device),
        'labels': torch.randint(0, 2, (batch_size, config.num_drum_classes)).float().to(config.device),
        'patch_shape': torch.tensor([[patch_size_t, patch_size_f]] * batch_size).to(config.device),
        'track_id': torch.tensor([i for i in range(batch_size)]).to(config.device),
        'start_time': torch.tensor([0.0] * batch_size).to(config.device),
        'end_time': torch.tensor([1.0] * batch_size).to(config.device)
    }
        
    # Test training step
    # Ensure module and data are on the correct device
    module.to(config.device)
    loss = module.training_step(dummy_batch, 0)
    logger.info(f"✓ Training step loss: {loss.item():.4f}")
        
    # Test validation step
    mock_trainer.training = False # Set to evaluation mode for validation_step
    val_loss = module.validation_step(dummy_batch, 0)
    logger.info(f"✓ Validation step loss: {val_loss.item():.4f}")
        
    # Test optimizer configuration
    optimizer_config = module.configure_optimizers()
    if isinstance(optimizer_config, tuple): # Optimizer and LR scheduler
        opt_name = type(optimizer_config[0][0]).__name__
        sched_name = type(optimizer_config[1][0]).__name__
        logger.info(f"✓ Optimizer configured: {opt_name}, Scheduler: {sched_name}")
    elif isinstance(optimizer_config, dict): # Optimizer and LR scheduler dictionary
        opt_name = type(optimizer_config["optimizer"]).__name__
        sched_name = type(optimizer_config["lr_scheduler"]["scheduler"]).__name__
        logger.info(f"✓ Optimizer configured: {opt_name}, Scheduler: {sched_name}")
    else: # Just optimizer
        opt_name = type(optimizer_config).__name__
        logger.info(f"✓ Optimizer configured: {opt_name}")
        
    assert True