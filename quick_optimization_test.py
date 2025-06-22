#!/usr/bin/env python3
"""
Quick Optimization Test - Test vectorized and parallel optimizations with small dataset
"""

import sys
import os
import logging
import time
import psutil
from pathlib import Path

# Ensure the model_training package is discoverable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from model_training.data_preparation import data_preparation

# --- Quick Test Configuration ---
EGMD_DATASET_DIR = "/Users/maple/Repos/chart-hero/datasets/e-gmd-v1.0.0"
PROCESSED_OUTPUT_DIR = "/Users/maple/Repos/chart-hero/datasets/processed_quick_test"
SAMPLE_RATIO = 0.002  # Very small sample for quick testing (0.2% of dataset)
NUM_BATCHES = 2  # Just 2 batches
MEMORY_LIMIT_GB = 16  # Higher memory limit
BATCH_SIZE_MULTIPLIER = 1.0

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def quick_performance_test():
    """Run a quick performance test with optimizations."""
    
    if not os.path.exists(EGMD_DATASET_DIR):
        logger.error(f"❌ E-GMD dataset directory not found: {EGMD_DATASET_DIR}")
        return False
        
    if not os.path.exists(PROCESSED_OUTPUT_DIR):
        os.makedirs(PROCESSED_OUTPUT_DIR)
    
    start_time = time.perf_counter()
    start_memory = psutil.Process().memory_info().rss / (1024**3)
    
    try:
        # Initialize data preparation
        logger.info(f"🔧 Initializing with sample_ratio={SAMPLE_RATIO}")
        data_prep = data_preparation(
            directory_path=EGMD_DATASET_DIR,
            dataset='egmd',
            sample_ratio=SAMPLE_RATIO,
            diff_threshold=1.0
        )
        
        total_pairs = len(data_prep.midi_wav_map)
        logger.info(f"📈 Dataset: {total_pairs} MIDI/audio pairs")
        
        if total_pairs == 0:
            logger.error("❌ No valid pairs found!")
            return False
        
        # Test sequential processing with vectorized optimizations
        logger.info(f"🎵 Testing sequential mode with vectorized audio slicing...")
        
        init_time = time.perf_counter()
        processed_files = data_prep.create_audio_set(
            pad_before=0.02,
            pad_after=0.02,
            fix_length=5.0,
            batching=True,
            dir_path=PROCESSED_OUTPUT_DIR,
            num_batches=NUM_BATCHES,
            memory_limit_gb=MEMORY_LIMIT_GB,
            batch_size_multiplier=BATCH_SIZE_MULTIPLIER,
            enable_process_parallelization=False  # Sequential mode
        )
        
        total_time = time.perf_counter() - start_time
        processing_time = time.perf_counter() - init_time
        end_memory = psutil.Process().memory_info().rss / (1024**3)
        memory_delta = end_memory - start_memory
        
        # Analyze output
        output_files = list(Path(PROCESSED_OUTPUT_DIR).glob("*.pkl"))
        total_batches = len(output_files)
        total_size_mb = sum(f.stat().st_size for f in output_files) / (1024**2)
        
        # Performance metrics
        files_per_second = processed_files / processing_time if processing_time > 0 else 0
        
        logger.info(f"✅ Quick Test Results:")
        logger.info(f"   Files processed: {processed_files}/{total_pairs}")
        logger.info(f"   Success rate: {(processed_files/total_pairs)*100:.1f}%")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"   Processing time: {processing_time:.2f}s")
        logger.info(f"   Speed: {files_per_second:.2f} files/second")
        logger.info(f"   Memory: {start_memory:.2f}GB → {end_memory:.2f}GB ({memory_delta:+.2f}GB)")
        logger.info(f"   Batches created: {total_batches}")
        logger.info(f"   Total output: {total_size_mb:.1f}MB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Quick test failed: {e}", exc_info=True)
        return False

def main():
    """Main test runner."""
    logger.info("🧪 Quick Performance Test - Vectorized Audio Slicing")
    logger.info(f"📂 Dataset: {EGMD_DATASET_DIR}")
    logger.info(f"📂 Output: {PROCESSED_OUTPUT_DIR}")
    
    # System info
    system_memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = os.cpu_count()
    logger.info(f"💻 System: {system_memory_gb:.1f}GB RAM, {cpu_count} CPU cores")
    
    success = quick_performance_test()
    
    if success:
        logger.info("✅ Quick test completed successfully!")
    else:
        logger.error("❌ Quick test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
