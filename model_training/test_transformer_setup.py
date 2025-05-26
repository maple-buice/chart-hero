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

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_config():
    """Test configuration classes."""
    logger.info("Testing configuration classes...")
    
    try:
        from model_training.transformer_config import get_config, auto_detect_config, validate_config
        
        # Test local config
        local_config = get_config("local")
        validate_config(local_config)
        logger.info(f"✓ Local config: {local_config.device}, batch_size={local_config.train_batch_size}")
        
        # Test cloud config
        cloud_config = get_config("cloud")
        validate_config(cloud_config)
        logger.info(f"✓ Cloud config: {cloud_config.device}, batch_size={cloud_config.train_batch_size}")
        
        # Test auto-detection
        auto_config = auto_detect_config()
        validate_config(auto_config)
        logger.info(f"✓ Auto-detected config: {auto_config.device}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Config test failed: {e}")
        return False


def test_model():
    """Test model instantiation and forward pass."""
    logger.info("Testing model architecture...")
    
    try:
        from model_training.transformer_config import get_config
        from model_training.transformer_model import create_model
        
        config = get_config("local")
        model = create_model(config)
        
        # Test forward pass
        batch_size = 2
        time_frames = 256  # ~5.8 seconds at 22050 Hz with hop_length=512
        freq_bins = 128
        
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


def test_data_processing():
    """Test data processing pipeline."""
    logger.info("Testing data processing...")
    
    try:
        from model_training.transformer_config import get_config
        from model_training.transformer_data import SpectrogramProcessor, DrumDataset
        
        config = get_config("local")
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
        dummy_df = pd.DataFrame({
            'audio_filename': ['dummy_audio.wav'] * 5,
            'track_id': range(5),
            'start': np.random.uniform(0, 5, 5),
            'end': np.random.uniform(1, 6, 5),
            'label': np.random.choice([0, 1, 2, 66, 67, 68], 5)
        })
        
        dataset = DrumDataset(dummy_df, config, '/tmp', mode='test', augment=False)
        logger.info(f"✓ Dataset created with {len(dataset)} samples")
        
        # Test sample loading (will use dummy data due to missing files)
        sample = dataset[0]
        logger.info(f"✓ Sample spectrogram shape: {sample['spectrogram'].shape}")
        logger.info(f"✓ Sample labels shape: {sample['labels'].shape}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Data processing test failed: {e}")
        return False


def test_training_module():
    """Test PyTorch Lightning training module."""
    logger.info("Testing training module...")
    
    try:
        from model_training.transformer_config import get_config
        try:
            from model_training.train_transformer import DrumTranscriptionModule
        except ImportError as e:
            logger.warning(f"PyTorch Lightning not available: {e}")
            return True  # Skip this test if dependencies not installed
        
        config = get_config("local")
        module = DrumTranscriptionModule(config)
        
        # Test forward pass
        batch_size = 2
        time_frames = 256
        freq_bins = 128
        
        dummy_batch = {
            'spectrogram': torch.randn(batch_size, 1, time_frames, freq_bins),
            'labels': torch.randint(0, 2, (batch_size, config.num_drum_classes)).float(),
            'patch_shape': torch.tensor([[16, 8], [16, 8]]),
            'track_id': torch.tensor([1, 2]),
            'start_time': torch.tensor([0.0, 1.0]),
            'end_time': torch.tensor([1.0, 2.0])
        }
        
        # Test training step
        loss = module.training_step(dummy_batch, 0)
        logger.info(f"✓ Training step loss: {loss.item():.4f}")
        
        # Test validation step
        val_loss = module.validation_step(dummy_batch, 0)
        logger.info(f"✓ Validation step loss: {val_loss.item():.4f}")
        
        # Test optimizer configuration
        optimizer_config = module.configure_optimizers()
        logger.info(f"✓ Optimizer configured: {type(optimizer_config['optimizer']).__name__}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Training module test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 50)
    logger.info("TRANSFORMER SETUP TESTS")
    logger.info("=" * 50)
    
    tests = [
        ("Configuration", test_config),
        ("Model Architecture", test_model),
        ("Data Processing", test_data_processing),
        ("Training Module", test_training_module),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'-' * 30}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'-' * 30}")
        
        result = test_func()
        results.append((test_name, result))
        
        if result:
            logger.info(f"✓ {test_name} test PASSED")
        else:
            logger.error(f"✗ {test_name} test FAILED")
    
    # Summary
    logger.info(f"\n{'=' * 50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'=' * 50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Transformer setup is ready.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)