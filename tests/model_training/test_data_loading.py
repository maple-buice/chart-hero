"""
Test script to verify the data loading fixes work correctly.
"""

import torch
from chart_hero.model_training.transformer_config import get_config
from chart_hero.model_training.transformer_data import create_data_loaders
from chart_hero.model_training.data_preparation import data_preparation

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_loading():
    """Test that data loading works without dimension errors."""
    
    # Get local config
    config = get_config("local")
    
    print(f"Testing with config:")
    print(f"  Sample rate: {config.sample_rate}")
    print(f"  Max audio length: {config.max_audio_length}")
    print(f"  N mels: {config.n_mels}")
    print(f"  Patch size: {config.patch_size}")
    print(f"  Batch size: {config.train_batch_size}")
    print()
    
    try:
        # Generate data first
        print("Generating test data...")
        data_prep = data_preparation(
            directory_path='datasets/e-gmd-v1.0.0',
            dataset='egmd',
            sample_ratio=0.001
        )
        data_prep.create_audio_set(
            dir_path='datasets/processed',
            num_batches=1
        )
        print("Test data generated.")

        # Test data loaders creation
        print("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            config=config,
            data_dir="datasets/processed",
            batch_size=2  # Small batch for testing
        )
        
        print(f"✅ Data loaders created successfully")
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Val samples: {len(val_loader.dataset)}")
        print(f"  Test samples: {len(test_loader.dataset)}")
        print()
        
        # Test loading a few samples
        print("Testing sample loading...")
        for i, batch in enumerate(train_loader):
            print(f"  Batch {i}:")
            print(f"    Spectrogram shape: {batch['spectrogram'].shape}")
            print(f"    Labels shape: {batch['labels'].shape}")
            print(f"    Patch shape: {batch['patch_shape']}")
            
            # Test that all spectrograms in batch have same shape
            spec_shapes = [s.shape for s in batch['spectrogram']]
            if len(set(spec_shapes)) == 1:
                print(f"    ✅ All spectrograms have consistent shape: {spec_shapes[0]}")
            else:
                print(f"    ❌ Inconsistent spectrogram shapes: {spec_shapes}")
                return False
            
            if i >= 2:  # Test only first 3 batches
                break
        
        print("✅ Data loading test completed successfully!")
        assert True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        assert False


