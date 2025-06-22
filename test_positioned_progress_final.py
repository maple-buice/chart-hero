#!/usr/bin/env python3
"""
Test the updated positioned progress bars with a small dataset.
"""

import sys
import os
import logging
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from model_training.data_preparation import data_preparation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def test_positioned_progress_bars():
    """Test the positioned progress bars with a small dataset."""
    
    dataset_dir = "/Users/maple/Repos/chart-hero/datasets/e-gmd-v1.0.0"
    output_dir = "/Users/maple/Repos/chart-hero/datasets/positioned_tqdm_test"
    
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return False
    
    try:
        logger.info("🧪 Testing positioned progress bars with small dataset...")
        
        # Create data preparation instance (this will show positioned duration bars)
        data_prep = data_preparation(dataset_dir, 'egmd', sample_ratio=0.002)  # Very small sample - about 9 files
        
        logger.info(f"Created data_prep with {len(data_prep.midi_wav_map)} file pairs")
        
        # Test sequential processing (should show positioned main process bars)
        logger.info("Testing sequential processing with positioned bars...")
        
        result = data_prep.create_audio_set(
            pad_before=0.1,
            pad_after=0.1,
            fix_length=2,  # Very short length for quick test
            batching=True,
            dir_path=output_dir,
            num_batches=1,
            memory_limit_gb=8,
            enable_process_parallelization=False  # Sequential first
        )
        
        logger.info(f"✅ Sequential test completed: {result} files processed")
        
        # Test parallel processing (should show positioned worker bars)
        if len(data_prep.midi_wav_map) > 5:  # Only test parallel if we have enough files
            logger.info("Testing parallel processing with positioned bars...")
            
            result_parallel = data_prep.create_audio_set(
                pad_before=0.1,
                pad_after=0.1,
                fix_length=2,  # Very short length for quick test
                batching=True,
                dir_path=output_dir + "_parallel",
                num_batches=1,
                memory_limit_gb=20,  # Higher to trigger parallel
                enable_process_parallelization=True  # Parallel mode
            )
            
            logger.info(f"✅ Parallel test completed: {result_parallel} files processed")
        else:
            logger.info("⚠️  Skipping parallel test - not enough files for meaningful parallel processing")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_positioned_progress_bars()
    if success:
        print("\n🎉 Positioned progress bar test completed successfully!")
        print("You should have seen clean, organized progress bars without jerky switching.")
    else:
        print("\n❌ Test failed - check the logs above for details.")
    
    sys.exit(0 if success else 1)
