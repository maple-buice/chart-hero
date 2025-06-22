#!/usr/bin/env python3
"""
Enhanced Performance Test for Optimized Data Preparation Pipeline
Tests vectorized audio slicing and optional parallelization improvements.
"""

import sys
import os
import logging
import time
import psutil
import gc
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure the model_training package is discoverable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from model_training.data_preparation import data_preparation

# --- Test Configuration ---
EGMD_DATASET_DIR = "/Users/maple/Repos/chart-hero/datasets/e-gmd-v1.0.0"
PROCESSED_OUTPUT_DIR = "/Users/maple/Repos/chart-hero/datasets/processed_optimized"
SAMPLE_RATIO = 0.01  # Slightly larger sample for optimization testing (1% of dataset)
NUM_BATCHES = 5  # More batches for better testing
MEMORY_LIMIT_GB = 16  # Higher memory limit to test parallelization
BATCH_SIZE_MULTIPLIER = 1.0

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'optimization_test_{time.strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

def test_sequential_optimizations():
    """Test the vectorized audio slicing optimizations in sequential mode."""
    logger.info("🚀 Testing Sequential Mode with Vectorized Optimizations")
    
    if os.path.exists(PROCESSED_OUTPUT_DIR):
        import shutil
        shutil.rmtree(PROCESSED_OUTPUT_DIR)
    os.makedirs(PROCESSED_OUTPUT_DIR)
    
    start_time = time.perf_counter()
    
    # Initialize data preparation
    data_prep = data_preparation(
        directory_path=EGMD_DATASET_DIR,
        dataset='egmd',
        sample_ratio=SAMPLE_RATIO,
        diff_threshold=1.0
    )
    
    total_pairs = len(data_prep.midi_wav_map)
    logger.info(f"📊 Dataset: {total_pairs} MIDI/audio pairs")
    
    # Run sequential processing with optimizations
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
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    # Analyze results
    output_files = list(Path(PROCESSED_OUTPUT_DIR).glob("*.pkl"))
    total_size_mb = sum(f.stat().st_size for f in output_files) / (1024**2)
    
    logger.info(f"✅ Sequential Test Results:")
    logger.info(f"   Time: {duration:.2f}s")
    logger.info(f"   Files processed: {processed_files}/{total_pairs}")
    logger.info(f"   Speed: {processed_files/duration:.2f} files/sec")
    logger.info(f"   Output size: {total_size_mb:.1f}MB")
    logger.info(f"   Batches created: {len(output_files)}")
    
    return {
        'mode': 'sequential',
        'duration': duration,
        'files_processed': processed_files,
        'total_pairs': total_pairs,
        'files_per_second': processed_files / duration,
        'output_size_mb': total_size_mb,
        'batches_created': len(output_files)
    }

def test_parallel_optimizations():
    """Test the improved parallel processing with vectorized optimizations."""
    logger.info("🚀 Testing Parallel Mode with Improved Memory Management")
    
    # Clean up from previous test
    if os.path.exists(PROCESSED_OUTPUT_DIR):
        import shutil
        shutil.rmtree(PROCESSED_OUTPUT_DIR)
    os.makedirs(PROCESSED_OUTPUT_DIR)
    
    start_time = time.perf_counter()
    
    # Initialize data preparation
    data_prep = data_preparation(
        directory_path=EGMD_DATASET_DIR,
        dataset='egmd',
        sample_ratio=SAMPLE_RATIO,
        diff_threshold=1.0
    )
    
    total_pairs = len(data_prep.midi_wav_map)
    logger.info(f"📊 Dataset: {total_pairs} MIDI/audio pairs")
    
    # Run parallel processing with optimizations
    processed_files = data_prep.create_audio_set(
        pad_before=0.02,
        pad_after=0.02,
        fix_length=5.0,
        batching=True,
        dir_path=PROCESSED_OUTPUT_DIR,
        num_batches=NUM_BATCHES,
        memory_limit_gb=MEMORY_LIMIT_GB,
        batch_size_multiplier=BATCH_SIZE_MULTIPLIER,
        enable_process_parallelization=True  # Parallel mode
    )
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    # Analyze results
    output_files = list(Path(PROCESSED_OUTPUT_DIR).glob("*.pkl"))
    total_size_mb = sum(f.stat().st_size for f in output_files) / (1024**2)
    
    logger.info(f"✅ Parallel Test Results:")
    logger.info(f"   Time: {duration:.2f}s")
    logger.info(f"   Files processed: {processed_files}/{total_pairs}")
    logger.info(f"   Speed: {processed_files/duration:.2f} files/sec")
    logger.info(f"   Output size: {total_size_mb:.1f}MB")
    logger.info(f"   Batches created: {len(output_files)}")
    
    return {
        'mode': 'parallel',
        'duration': duration,
        'files_processed': processed_files,
        'total_pairs': total_pairs,
        'files_per_second': processed_files / duration,
        'output_size_mb': total_size_mb,
        'batches_created': len(output_files)
    }

def compare_performance(sequential_results, parallel_results):
    """Compare performance between sequential and parallel modes."""
    logger.info("📈 Performance Comparison:")
    logger.info("="*60)
    
    seq_speed = sequential_results['files_per_second']
    par_speed = parallel_results['files_per_second']
    speedup = par_speed / seq_speed if seq_speed > 0 else 0
    
    seq_time = sequential_results['duration']
    par_time = parallel_results['duration']
    time_reduction = (seq_time - par_time) / seq_time * 100 if seq_time > 0 else 0
    
    logger.info(f"Sequential Mode:")
    logger.info(f"  Time: {seq_time:.2f}s")
    logger.info(f"  Speed: {seq_speed:.2f} files/sec")
    logger.info(f"  Files processed: {sequential_results['files_processed']}")
    
    logger.info(f"Parallel Mode:")
    logger.info(f"  Time: {par_time:.2f}s")
    logger.info(f"  Speed: {par_speed:.2f} files/sec")
    logger.info(f"  Files processed: {parallel_results['files_processed']}")
    
    logger.info(f"Performance Improvement:")
    logger.info(f"  Speedup: {speedup:.2f}x")
    logger.info(f"  Time reduction: {time_reduction:.1f}%")
    
    if speedup > 1.2:
        logger.info("🎉 Significant speedup achieved with parallelization!")
    elif speedup > 1.05:
        logger.info("✅ Modest speedup achieved with parallelization")
    else:
        logger.info("⚠️  Parallelization did not provide significant speedup")
        
    # Check for any processing differences
    if sequential_results['files_processed'] != parallel_results['files_processed']:
        logger.warning(f"⚠️  Different number of files processed: sequential={sequential_results['files_processed']}, parallel={parallel_results['files_processed']}")
    
    return {
        'speedup': speedup,
        'time_reduction_percent': time_reduction,
        'sequential': sequential_results,
        'parallel': parallel_results
    }

def main():
    """Main test runner for optimization comparison."""
    logger.info("🧪 Enhanced Performance Testing - Vectorized + Parallel Optimizations")
    logger.info(f"📂 Dataset: {EGMD_DATASET_DIR}")
    logger.info(f"📂 Output: {PROCESSED_OUTPUT_DIR}")
    logger.info(f"🔬 Sample ratio: {SAMPLE_RATIO} ({SAMPLE_RATIO*100:.1f}% of dataset)")
    logger.info(f"💾 Memory limit: {MEMORY_LIMIT_GB}GB")
    
    # System info
    system_memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = os.cpu_count()
    logger.info(f"💻 System: {system_memory_gb:.1f}GB RAM, {cpu_count} CPU cores")
    
    if not os.path.exists(EGMD_DATASET_DIR):
        logger.error(f"❌ Dataset directory not found: {EGMD_DATASET_DIR}")
        return
    
    try:
        # Test 1: Sequential mode with vectorized optimizations
        logger.info("\n" + "="*80)
        logger.info("TEST 1: Sequential Mode with Vectorized Audio Slicing")
        logger.info("="*80)
        sequential_results = test_sequential_optimizations()
        
        # Brief pause and cleanup
        time.sleep(2)
        gc.collect()
        
        # Test 2: Parallel mode with improved memory management
        logger.info("\n" + "="*80)
        logger.info("TEST 2: Parallel Mode with Improved Memory Management")
        logger.info("="*80)
        
        # Check if system meets parallel requirements
        if MEMORY_LIMIT_GB >= 16 and system_memory_gb >= 32:
            parallel_results = test_parallel_optimizations()
            
            # Compare results
            logger.info("\n" + "="*80)
            logger.info("PERFORMANCE COMPARISON")
            logger.info("="*80)
            comparison = compare_performance(sequential_results, parallel_results)
            
            # Save detailed results
            results_file = f"optimization_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
            import json
            with open(results_file, 'w') as f:
                json.dump(comparison, f, indent=2)
            logger.info(f"📄 Detailed results saved to: {results_file}")
            
        else:
            logger.warning("⚠️  System does not meet parallel processing requirements")
            logger.warning(f"   Required: memory_limit >= 16GB, system_memory >= 32GB")
            logger.warning(f"   Current: memory_limit = {MEMORY_LIMIT_GB}GB, system_memory = {system_memory_gb:.1f}GB")
            logger.info("✅ Sequential optimization test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return

    logger.info("🎉 Optimization testing completed!")

if __name__ == "__main__":
    main()
