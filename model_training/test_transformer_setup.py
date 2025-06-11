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

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_config(invoked_config_name: str): # Modified signature
    """Test configuration classes."""
    logger.info("Testing configuration classes...")
    overall_test_success = True
    
    try:
        from model_training.transformer_config import get_config, auto_detect_config, validate_config
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        # 1. Test the invoked configuration (this one MUST pass)
        logger.info(f"--- Validating invoked config: '{invoked_config_name}' ---")
        try:
            cfg_invoked = get_config(invoked_config_name) # get_config handles "auto"
            validate_config(cfg_invoked) # This must pass
            logger.info(f"✓ Invoked config '{invoked_config_name}': VALIDATED - device {cfg_invoked.device}, batch_size={cfg_invoked.train_batch_size}")
        except Exception as e:
            logger.error(f"✗ Invoked config '{invoked_config_name}' FAILED validation: {e}")
            return False # Critical failure

        # 2. Test other standard configurations for completeness, with conditional MPS check
        standard_configs_to_check = {"local", "cloud", "auto"} # Set of all standard configs
        other_configs_to_verify = standard_configs_to_check - {invoked_config_name} # Exclude already tested one

        for cfg_name_to_check in other_configs_to_verify:
            logger.info(f"--- Checking standard config: '{cfg_name_to_check}' ---")
            try:
                cfg = get_config(cfg_name_to_check)
                if cfg.device == "mps" and not mps_available:
                    logger.warning(f"✓ Config '{cfg_name_to_check}' (device: {cfg.device}) specifies MPS, but MPS not available. Loaded but skipping full device validation for this non-invoked config.")
                    # Perform a "light" validation if possible, or just log it was loaded
                    # For now, we just note it. If validate_config has device specific checks, this avoids them.
                else:
                    validate_config(cfg) # Validate if not MPS or if MPS is available
                    logger.info(f"✓ Config '{cfg_name_to_check}': Loaded & Validated - device {cfg.device}, batch_size={cfg.train_batch_size}")
            except Exception as e:
                logger.error(f"✗ Config '{cfg_name_to_check}' (non-invoked) test generated an error: {e}")
                overall_test_success = False # Mark as failed but continue testing others
        
        return overall_test_success
    except Exception as e:
        logger.error(f"✗ Config test suite failed globally: {e}")
        return False


def test_model(config_name: str): # Modified signature
    """Test model instantiation and forward pass."""
    logger.info("Testing model architecture...")
    
    try:
        from model_training.transformer_config import get_config
        from model_training.transformer_model import create_model
        
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
        
        return True
    except Exception as e:
        logger.error(f"✗ Model test failed: {e}")
        return False


def test_data_processing(config_name: str): # Modified signature
    """Test data processing pipeline."""
    logger.info("Testing data processing...")
    
    try:
        from model_training.transformer_config import get_config
        from model_training.transformer_data import SpectrogramProcessor, DrumDataset
        
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
        # Create a temporary directory for dummy audio files if needed by DrumDataset
        temp_data_dir = Path("/tmp/chart_hero_test_data")
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        dummy_audio_path = temp_data_dir / "dummy_audio.wav"
        # Create a minimal dummy wav file if DrumDataset tries to load it
        try:
            import soundfile as sf
            sf.write(dummy_audio_path, np.random.randn(config.sample_rate), config.sample_rate)
        except ImportError:
            logger.warning("soundfile not installed, cannot create dummy .wav for dataset test. Dataset might fail if it loads audio.")

        dummy_df = pd.DataFrame({
            'audio_filename': [str(dummy_audio_path)] * 5, # Use path to dummy wav
            'track_id': range(5),
            'start': np.random.uniform(0, 0.5, 5), # Shorter starts for 1s audio
            'end': np.random.uniform(0.6, 1.0, 5),   # Shorter ends for 1s audio
            'label': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8], 5) # Assuming 9 classes
        })
        
        dataset = DrumDataset(dummy_df, config, str(temp_data_dir), mode='test', augment=False)
        logger.info(f"✓ Dataset created with {len(dataset)} samples")
        
        # Test sample loading
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"✓ Sample spectrogram shape: {sample['spectrogram'].shape}")
            logger.info(f"✓ Sample labels shape: {sample['labels'].shape}")
        else:
            logger.warning("Dataset is empty, cannot test sample loading.")
            # This might be an error depending on expectations

        # Clean up dummy file and dir
        if dummy_audio_path.exists():
            dummy_audio_path.unlink()
        if temp_data_dir.exists():
            # Remove directory if empty, otherwise be careful
            try: 
                os.rmdir(temp_data_dir) 
            except OSError: 
                logger.warning(f"Could not remove temp_data_dir {temp_data_dir}, it might not be empty.")

        return True
    except Exception as e:
        logger.error(f"✗ Data processing test failed: {e}")
        return False


def test_training_module(config_name: str): # Modified signature
    """Test PyTorch Lightning training module."""
    logger.info("Testing training module...")
    
    try:
        from model_training.transformer_config import get_config
        from model_training.train_transformer import DrumTranscriptionModule
        
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
        
        return True
    except Exception as e:
        logger.error(f"✗ Training module test failed: {e}", exc_info=True) # Add exc_info for more details
        return False


def main():
    """Run all tests."""
    logger.info("=" * 50)
    logger.info("TRANSFORMER SETUP TESTS")
    logger.info("=" * 50)

    parser = argparse.ArgumentParser(description="Run transformer setup tests.")
    parser.add_argument("--config", type=str, default="auto", choices=["local", "cloud", "auto"],
                        help="Specify which configuration to primarily test (local, cloud, or auto-detect).")
    script_args = parser.parse_args()
    
    effective_config_name_for_tests = script_args.config

    tests = [
        ("Configuration", test_config),
        ("Model Architecture", test_model),
        ("Data Processing", test_data_processing),
        ("Training Module", test_training_module),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'-' * 30}")
        logger.info(f"Running {test_name} test (using '{effective_config_name_for_tests}' config context)...")
        logger.info(f"{'-' * 30}")
        
        result = test_func(effective_config_name_for_tests) # Pass the config name string
        results.append((test_name, result))
        
        if result:
            logger.info(f"✓ {test_name} test PASSED")
        else:
            logger.error(f"✗ {test_name} test FAILED")
    
    # Summary
    logger.info(f"\n{'=' * 50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'=' * 50}")
    
    passed_count = sum(1 for _, result in results if result)
    total_count = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        logger.info("🎉 All tests passed! Transformer setup is ready.")
        # exit(0) # Exit with 0 if running as a standalone script and all pass
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        # exit(1) # Exit with 1 if running as a standalone script and some fail
    return passed_count == total_count # Return overall success status

if __name__ == "__main__":
    overall_success = main()
    exit(0 if overall_success else 1)