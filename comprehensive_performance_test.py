#!/usr/bin/env python3
"""
Comprehensive Parallel vs Sequential Performance Test

This test creates a consistent, controlled dataset and runs proper comparisons
between sequential and parallel processing modes.
"""

import sys
import os
import logging
import time
import psutil
import shutil
import json
from pathlib import Path

# Ensure the model_training package is discoverable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from model_training.data_preparation import data_preparation

# --- Test Configuration ---
EGMD_DATASET_DIR = "/Users/maple/Repos/chart-hero/datasets/e-gmd-v1.0.0"
TEST_OUTPUT_BASE = "/Users/maple/Repos/chart-hero/datasets/performance_test"
SEQUENTIAL_OUTPUT_DIR = f"{TEST_OUTPUT_BASE}/sequential"
PARALLEL_OUTPUT_DIR = f"{TEST_OUTPUT_BASE}/parallel"
FIXED_SEED = 42  # For reproducible sampling
SAMPLE_RATIO = 0.0005  # 0.05% of dataset for ultra-fast test
NUM_BATCHES = 1  # Just 1 batch for quickest test
MEMORY_LIMIT_GB = 16  # Higher memory limit to enable parallel processing
BATCH_SIZE_MULTIPLIER = 0.3  # Even smaller batch size multiplier
MAX_FILES_PER_TEST = 50  # Hard limit on files processed per test

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def setup_test_environment():
    """Set up clean test environment."""
    # Clean up previous test outputs
    if os.path.exists(TEST_OUTPUT_BASE):
        logger.info(f"🧹 Cleaning up previous test data: {TEST_OUTPUT_BASE}")
        shutil.rmtree(TEST_OUTPUT_BASE)
    
    # Create fresh directories
    os.makedirs(SEQUENTIAL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PARALLEL_OUTPUT_DIR, exist_ok=True)
    
    logger.info(f"✅ Test environment prepared:")
    logger.info(f"   Sequential output: {SEQUENTIAL_OUTPUT_DIR}")
    logger.info(f"   Parallel output: {PARALLEL_OUTPUT_DIR}")


def create_consistent_dataset():
    """Create a consistent dataset for both tests using a fixed seed."""
    logger.info(f"🎲 Creating consistent dataset with seed={FIXED_SEED}, sample_ratio={SAMPLE_RATIO}")
    
    # Set random seed for reproducible sampling
    import numpy as np
    np.random.seed(FIXED_SEED)
    
    data_prep = data_preparation(
        directory_path=EGMD_DATASET_DIR,
        dataset='egmd',
        sample_ratio=SAMPLE_RATIO,
        diff_threshold=1.0
    )
    
    total_pairs = len(data_prep.midi_wav_map)
    logger.info(f"📊 Consistent dataset created: {total_pairs} MIDI/audio pairs")
    
    if total_pairs == 0:
        raise ValueError("No valid pairs found in dataset!")
    
    # Limit the number of files for ultra-fast testing
    if total_pairs > MAX_FILES_PER_TEST:
        logger.info(f"⚡ Limiting dataset to {MAX_FILES_PER_TEST} files for ultra-fast test")
        data_prep.midi_wav_map = data_prep.midi_wav_map.head(MAX_FILES_PER_TEST)
        total_pairs = len(data_prep.midi_wav_map)
    
    # Save the dataset mapping for reproducibility
    dataset_info = {
        'total_pairs': total_pairs,
        'sample_ratio': SAMPLE_RATIO,
        'seed': FIXED_SEED,
        'file_list': data_prep.midi_wav_map.to_dict('records')
    }
    
    with open(f"{TEST_OUTPUT_BASE}/dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2, default=str)
    
    return data_prep


def run_sequential_test(data_prep):
    """Run sequential processing test."""
    logger.info("🔄 Starting Sequential Processing Test")
    
    start_time = time.perf_counter()
    start_memory = psutil.Process().memory_info().rss / (1024**3)
    
    try:
        processed_files = data_prep.create_audio_set(
            pad_before=0.02,
            pad_after=0.02,
            fix_length=5.0,
            batching=True,
            dir_path=SEQUENTIAL_OUTPUT_DIR,
            num_batches=NUM_BATCHES,
            memory_limit_gb=MEMORY_LIMIT_GB,
            batch_size_multiplier=BATCH_SIZE_MULTIPLIER,
            enable_process_parallelization=False  # Sequential mode
        )
        
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / (1024**3)
        
        # Analyze output
        output_files = list(Path(SEQUENTIAL_OUTPUT_DIR).glob("*.pkl"))
        total_batches = len(output_files)
        total_size_mb = sum(f.stat().st_size for f in output_files) / (1024**2)
        
        results = {
            'mode': 'sequential',
            'processed_files': processed_files,
            'total_pairs': len(data_prep.midi_wav_map),
            'success_rate': (processed_files / len(data_prep.midi_wav_map)) * 100,
            'total_time': end_time - start_time,
            'files_per_second': processed_files / (end_time - start_time),
            'memory_start_gb': start_memory,
            'memory_end_gb': end_memory,
            'memory_delta_gb': end_memory - start_memory,
            'batches_created': total_batches,
            'total_output_mb': total_size_mb,
            'avg_mb_per_batch': total_size_mb / total_batches if total_batches > 0 else 0
        }
        
        logger.info("✅ Sequential Test Results:")
        logger.info(f"   Files processed: {results['processed_files']}/{results['total_pairs']}")
        logger.info(f"   Success rate: {results['success_rate']:.1f}%")
        logger.info(f"   Total time: {results['total_time']:.2f}s")
        logger.info(f"   Speed: {results['files_per_second']:.2f} files/second")
        logger.info(f"   Memory: {results['memory_start_gb']:.2f}GB → {results['memory_end_gb']:.2f}GB ({results['memory_delta_gb']:+.2f}GB)")
        logger.info(f"   Output: {results['batches_created']} batches, {results['total_output_mb']:.1f}MB total")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Sequential test failed: {e}", exc_info=True)
        return None


def run_parallel_test(data_prep):
    """Run parallel processing test."""
    logger.info("⚡ Starting Parallel Processing Test")
    
    start_time = time.perf_counter()
    start_memory = psutil.Process().memory_info().rss / (1024**3)
    
    try:
        processed_files = data_prep.create_audio_set(
            pad_before=0.02,
            pad_after=0.02,
            fix_length=5.0,
            batching=True,
            dir_path=PARALLEL_OUTPUT_DIR,
            num_batches=NUM_BATCHES,
            memory_limit_gb=MEMORY_LIMIT_GB,
            batch_size_multiplier=BATCH_SIZE_MULTIPLIER,
            enable_process_parallelization=True  # Parallel mode
        )
        
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / (1024**3)
        
        # Analyze output
        output_files = list(Path(PARALLEL_OUTPUT_DIR).glob("*.pkl"))
        total_batches = len(output_files)
        total_size_mb = sum(f.stat().st_size for f in output_files) / (1024**2)
        
        results = {
            'mode': 'parallel',
            'processed_files': processed_files,
            'total_pairs': len(data_prep.midi_wav_map),
            'success_rate': (processed_files / len(data_prep.midi_wav_map)) * 100,
            'total_time': end_time - start_time,
            'files_per_second': processed_files / (end_time - start_time),
            'memory_start_gb': start_memory,
            'memory_end_gb': end_memory,
            'memory_delta_gb': end_memory - start_memory,
            'batches_created': total_batches,
            'total_output_mb': total_size_mb,
            'avg_mb_per_batch': total_size_mb / total_batches if total_batches > 0 else 0
        }
        
        logger.info("✅ Parallel Test Results:")
        logger.info(f"   Files processed: {results['processed_files']}/{results['total_pairs']}")
        logger.info(f"   Success rate: {results['success_rate']:.1f}%")
        logger.info(f"   Total time: {results['total_time']:.2f}s")
        logger.info(f"   Speed: {results['files_per_second']:.2f} files/second")
        logger.info(f"   Memory: {results['memory_start_gb']:.2f}GB → {results['memory_end_gb']:.2f}GB ({results['memory_delta_gb']:+.2f}GB)")
        logger.info(f"   Output: {results['batches_created']} batches, {results['total_output_mb']:.1f}MB total")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Parallel test failed: {e}", exc_info=True)
        return None


def compare_results(sequential_results, parallel_results):
    """Compare and analyze the results."""
    if not sequential_results or not parallel_results:
        logger.error("❌ Cannot compare results - one or both tests failed")
        return
    
    logger.info("\n" + "="*80)
    logger.info("📊 COMPREHENSIVE PERFORMANCE COMPARISON")
    logger.info("="*80)
    
    # Performance comparison
    speed_improvement = (parallel_results['files_per_second'] / sequential_results['files_per_second'] - 1) * 100
    time_improvement = (1 - parallel_results['total_time'] / sequential_results['total_time']) * 100
    
    logger.info(f"🚀 Speed Comparison:")
    logger.info(f"   Sequential: {sequential_results['files_per_second']:.2f} files/sec")
    logger.info(f"   Parallel:   {parallel_results['files_per_second']:.2f} files/sec")
    logger.info(f"   Improvement: {speed_improvement:+.1f}%")
    
    logger.info(f"⏱️  Time Comparison:")
    logger.info(f"   Sequential: {sequential_results['total_time']:.2f}s")
    logger.info(f"   Parallel:   {parallel_results['total_time']:.2f}s")
    logger.info(f"   Improvement: {time_improvement:+.1f}%")
    
    logger.info(f"💾 Memory Comparison:")
    logger.info(f"   Sequential: {sequential_results['memory_delta_gb']:+.2f}GB")
    logger.info(f"   Parallel:   {parallel_results['memory_delta_gb']:+.2f}GB")
    
    logger.info(f"📦 Output Comparison:")
    logger.info(f"   Sequential: {sequential_results['total_output_mb']:.1f}MB ({sequential_results['batches_created']} batches)")
    logger.info(f"   Parallel:   {parallel_results['total_output_mb']:.1f}MB ({parallel_results['batches_created']} batches)")
    
    # Determine if parallel processing actually worked
    if parallel_results['total_time'] < sequential_results['total_time'] * 0.8:
        logger.info("✅ Parallel processing is working and shows significant improvement!")
    elif parallel_results['total_time'] < sequential_results['total_time'] * 0.95:
        logger.info("⚠️  Parallel processing shows modest improvement")
    else:
        logger.info("❌ Parallel processing may have fallen back to sequential mode")
    
    # Save comparison results
    comparison = {
        'sequential': sequential_results,
        'parallel': parallel_results,
        'comparison': {
            'speed_improvement_percent': speed_improvement,
            'time_improvement_percent': time_improvement,
            'parallel_is_faster': parallel_results['total_time'] < sequential_results['total_time'],
            'significant_improvement': time_improvement > 20
        }
    }
    
    with open(f"{TEST_OUTPUT_BASE}/comparison_results.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"📋 Detailed results saved to: {TEST_OUTPUT_BASE}/comparison_results.json")


def main():
    """Main test runner."""
    logger.info("🧪 Comprehensive Parallel vs Sequential Performance Test")
    logger.info(f"📂 Dataset: {EGMD_DATASET_DIR}")
    logger.info(f"📂 Output: {TEST_OUTPUT_BASE}")
    
    # System info
    system_memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = os.cpu_count()
    logger.info(f"💻 System: {system_memory_gb:.1f}GB RAM, {cpu_count} CPU cores")
    
    try:
        # Setup
        setup_test_environment()
        
        # Create consistent dataset
        data_prep = create_consistent_dataset()
        
        # Run sequential test
        sequential_results = run_sequential_test(data_prep)
        
        # Reset dataset for parallel test (ensure same starting conditions)
        data_prep = create_consistent_dataset()
        
        # Run parallel test  
        parallel_results = run_parallel_test(data_prep)
        
        # Compare results
        compare_results(sequential_results, parallel_results)
        
        logger.info("✅ Comprehensive performance test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
