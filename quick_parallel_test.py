#!/usr/bin/env python3
"""
Quick Parallel vs Sequential Test

A fast test to verify parallel processing is working with minimal overhead.
"""

import sys
import os
import logging
import time
import psutil
import shutil
from pathlib import Path

# Ensure the model_training package is discoverable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from model_training.data_preparation import data_preparation

# --- Test Configuration ---
EGMD_DATASET_DIR = "/Users/maple/Repos/chart-hero/datasets/e-gmd-v1.0.0"
TEST_OUTPUT_BASE = "/Users/maple/Repos/chart-hero/datasets/quick_parallel_test"
SEQUENTIAL_OUTPUT_DIR = f"{TEST_OUTPUT_BASE}/sequential"
PARALLEL_OUTPUT_DIR = f"{TEST_OUTPUT_BASE}/parallel"
FIXED_SEED = 42
SAMPLE_RATIO = 0.0003  # Ultra-small sample (0.03%)
MEMORY_LIMIT_GB = 16  # Higher memory limit to enable parallel processing
BATCH_SIZE_MULTIPLIER = 0.3
MAX_FILES_PER_TEST = 30  # Hard limit for ultra-fast test

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def setup_test_environment():
    """Set up clean test environment."""
    if os.path.exists(TEST_OUTPUT_BASE):
        shutil.rmtree(TEST_OUTPUT_BASE)
    
    os.makedirs(SEQUENTIAL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PARALLEL_OUTPUT_DIR, exist_ok=True)
    
    logger.info(f"✅ Test environment prepared")


def create_test_dataset():
    """Create a small test dataset."""
    logger.info(f"🎲 Creating test dataset (sample_ratio={SAMPLE_RATIO})")
    
    import numpy as np
    np.random.seed(FIXED_SEED)
    
    data_prep = data_preparation(
        directory_path=EGMD_DATASET_DIR,
        dataset='egmd',
        sample_ratio=SAMPLE_RATIO,
        diff_threshold=1.0
    )
    
    total_pairs = len(data_prep.midi_wav_map)
    logger.info(f"📊 Test dataset: {total_pairs} MIDI/audio pairs")
    
    if total_pairs == 0:
        raise ValueError("No valid pairs found!")
    
    # Limit files for ultra-fast testing
    if total_pairs > MAX_FILES_PER_TEST:
        logger.info(f"⚡ Limiting to {MAX_FILES_PER_TEST} files for ultra-fast test")
        data_prep.midi_wav_map = data_prep.midi_wav_map.head(MAX_FILES_PER_TEST)
        total_pairs = len(data_prep.midi_wav_map)
    
    return data_prep


def run_test(data_prep, mode, output_dir):
    """Run a single test (sequential or parallel)."""
    logger.info(f"🧪 Running {mode.upper()} test...")
    
    start_time = time.perf_counter()
    start_memory = psutil.Process().memory_info().rss / (1024**3)
    
    try:
        processed_files = data_prep.create_audio_set(
            pad_before=0.02,
            pad_after=0.02,
            fix_length=5.0,
            batching=False,  # No batching for speed
            dir_path=output_dir,
            num_batches=1,
            memory_limit_gb=MEMORY_LIMIT_GB,
            batch_size_multiplier=BATCH_SIZE_MULTIPLIER,
            enable_process_parallelization=(mode == 'parallel')
        )
        
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / (1024**3)
        
        results = {
            'mode': mode,
            'processed_files': processed_files,
            'total_pairs': len(data_prep.midi_wav_map),
            'success_rate': (processed_files / len(data_prep.midi_wav_map)) * 100,
            'total_time': end_time - start_time,
            'files_per_second': processed_files / (end_time - start_time),
            'memory_delta_gb': end_memory - start_memory,
        }
        
        logger.info(f"✅ {mode.upper()} Results:")
        logger.info(f"   Files: {results['processed_files']}/{results['total_pairs']} ({results['success_rate']:.1f}%)")
        logger.info(f"   Time: {results['total_time']:.2f}s")
        logger.info(f"   Speed: {results['files_per_second']:.2f} files/sec")
        logger.info(f"   Memory: {results['memory_delta_gb']:+.2f}GB")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ {mode} test failed: {e}")
        return None


def compare_results(sequential_results, parallel_results):
    """Compare the results."""
    if not sequential_results or not parallel_results:
        logger.error("❌ Cannot compare - one or both tests failed")
        return
    
    logger.info("\n" + "="*60)
    logger.info("📊 PERFORMANCE COMPARISON")
    logger.info("="*60)
    
    speed_improvement = (parallel_results['files_per_second'] / sequential_results['files_per_second'] - 1) * 100
    time_improvement = (1 - parallel_results['total_time'] / sequential_results['total_time']) * 100
    
    logger.info(f"🚀 Processing Speed:")
    logger.info(f"   Sequential: {sequential_results['files_per_second']:.2f} files/sec")
    logger.info(f"   Parallel:   {parallel_results['files_per_second']:.2f} files/sec")
    logger.info(f"   Improvement: {speed_improvement:+.1f}%")
    
    logger.info(f"⏱️  Total Time:")
    logger.info(f"   Sequential: {sequential_results['total_time']:.2f}s")
    logger.info(f"   Parallel:   {parallel_results['total_time']:.2f}s")
    logger.info(f"   Improvement: {time_improvement:+.1f}%")
    
    # Determine success
    if time_improvement > 20:
        logger.info("✅ EXCELLENT: Parallel processing shows significant improvement!")
        status = "EXCELLENT"
    elif time_improvement > 5:
        logger.info("✅ GOOD: Parallel processing shows noticeable improvement!")
        status = "GOOD"
    elif time_improvement > -5:
        logger.info("⚠️  MARGINAL: Parallel processing shows little difference")
        status = "MARGINAL"
    else:
        logger.info("❌ POOR: Parallel processing may not be working properly")
        status = "POOR"
    
    return status


def main():
    """Main test runner."""
    logger.info("⚡ Quick Parallel vs Sequential Test")
    logger.info(f"📂 Dataset: {EGMD_DATASET_DIR}")
    
    # System info
    system_memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = os.cpu_count()
    logger.info(f"💻 System: {system_memory_gb:.1f}GB RAM, {cpu_count} CPU cores")
    
    try:
        # Setup
        setup_test_environment()
        
        # Create test dataset
        data_prep = create_test_dataset()
        
        # Run sequential test
        sequential_results = run_test(data_prep, 'sequential', SEQUENTIAL_OUTPUT_DIR)
        
        # Reset dataset and run parallel test
        data_prep = create_test_dataset()
        parallel_results = run_test(data_prep, 'parallel', PARALLEL_OUTPUT_DIR)
        
        # Compare results
        status = compare_results(sequential_results, parallel_results)
        
        if status in ["EXCELLENT", "GOOD"]:
            logger.info("✅ Quick test completed successfully - parallel processing is working!")
        else:
            logger.warning("⚠️  Quick test completed but parallel processing may need attention")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
