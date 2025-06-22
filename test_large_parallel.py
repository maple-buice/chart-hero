#!/usr/bin/env python3
"""
Test script to verify parallel processing with a larger dataset
"""

import logging
import random
import numpy as np
from model_training.data_preparation import data_preparation

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_large_parallel():
    """Test parallel processing with a larger dataset"""
    
    logger.info("🧪 Testing Parallel Processing with Large Dataset Simulation")
    logger.info(f"📂 Dataset: /Users/maple/Repos/chart-hero/datasets/e-gmd-v1.0.0")
    
    # Use a larger sample ratio to get more files
    sample_ratio = 0.005  # This should give us ~100 files, well above the 50 file threshold
    
    logger.info(f"🎲 Creating larger dataset with sample_ratio={sample_ratio}")
    
    # Initialize data_preparation
    data_prep = data_preparation(
        directory_path='/Users/maple/Repos/chart-hero/datasets/e-gmd-v1.0.0',
        dataset='egmd',
        sample_ratio=sample_ratio,
        diff_threshold=1.0
    )
    
    dataset_size = len(data_prep.midi_wav_map)
    logger.info(f"📊 Dataset size: {dataset_size} files")
    
    if dataset_size < 50:
        logger.warning(f"⚠️  Dataset size ({dataset_size}) still below parallel threshold (50)")
        logger.warning("   This will still fall back to sequential mode")
    else:
        logger.info(f"✅ Dataset size ({dataset_size}) above parallel threshold (50)")
        logger.info("   Parallel processing should be used")
    
    # Test parallel processing
    logger.info("🚀 Testing parallel processing...")
    
    try:
        processed_files = data_prep.create_audio_set(
            pad_before=0.02,
            pad_after=0.02,
            fix_length=5.0,
            batching=False,  # Disable batching for speed
            enable_process_parallelization=True,
            memory_limit_gb=16
        )
        
        logger.info(f"✅ Parallel test completed: {processed_files} files processed")
        
    except Exception as e:
        logger.error(f"❌ Parallel test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_large_parallel()
