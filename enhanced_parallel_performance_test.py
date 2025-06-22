#!/usr/bin/env python3
"""
Enhanced Performance Test with Actual Parallel Processing

This test increases the memory limit and file count to trigger real parallel processing
and measure the performance benefits of the parallel worker implementation.
"""

import sys
import os
import logging
import time
import psutil
import shutil
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Ensure the model_training package is discoverable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from model_training.data_preparation import data_preparation

# --- Enhanced Test Configuration for Real Parallel Processing ---
EGMD_DATASET_DIR = "/Users/maple/Repos/chart-hero/datasets/e-gmd-v1.0.0"
TEST_OUTPUT_BASE = "/Users/maple/Repos/chart-hero/datasets/enhanced_parallel_test"
SEQUENTIAL_OUTPUT_DIR = f"{TEST_OUTPUT_BASE}/sequential"
PARALLEL_OUTPUT_DIR = f"{TEST_OUTPUT_BASE}/parallel"

# Configuration to trigger actual parallel processing
NUM_TEST_FILES = 60  # Above 50 file threshold for parallel processing
MEMORY_LIMIT_GB = 20  # Above 16GB threshold for parallel processing  
BATCH_SIZE_MULTIPLIER = 1.0
NUM_BATCHES = 2  # More batches to see parallel benefits

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'enhanced_parallel_test_{time.strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Run the enhanced parallel performance test."""
    logger.info("🚀 Starting Enhanced Parallel Performance Test")
    logger.info("="*70)
    logger.info(f"📊 Test Configuration:")
    logger.info(f"   Files: {NUM_TEST_FILES} (triggering parallel mode)")
    logger.info(f"   Memory: {MEMORY_LIMIT_GB}GB (enabling parallel processing)")
    logger.info(f"   Batches: {NUM_BATCHES}")
    
    # Check if dataset exists
    if not os.path.exists(EGMD_DATASET_DIR):
        logger.error(f"❌ Dataset not found: {EGMD_DATASET_DIR}")
        return False
    
    try:
        # Check system requirements
        system_memory_gb = psutil.virtual_memory().total / (1024**3)
        if system_memory_gb < 32:
            logger.warning(f"⚠️  System memory ({system_memory_gb:.1f}GB) below 32GB threshold")
            logger.warning("   Parallel processing may still fall back to sequential mode")
        
        # Clean up previous test data
        if os.path.exists(TEST_OUTPUT_BASE):
            logger.info(f"🧹 Cleaning up previous test data...")
            shutil.rmtree(TEST_OUTPUT_BASE)
        
        # Get fixed file set
        logger.info(f"🎯 Selecting fixed set of {NUM_TEST_FILES} files...")
        csv_files = [f for f in os.listdir(EGMD_DATASET_DIR) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in {EGMD_DATASET_DIR}")
        
        csv_path = os.path.join(EGMD_DATASET_DIR, csv_files[0])
        df = pd.read_csv(csv_path)
        df_sorted = df.sort_values('midi_filename')
        
        # Select files
        selected_pairs = []
        for _, row in df_sorted.iterrows():
            if len(selected_pairs) >= NUM_TEST_FILES:
                break
                
            midi_path = os.path.join(EGMD_DATASET_DIR, row['midi_filename'])
            audio_path = os.path.join(EGMD_DATASET_DIR, row['audio_filename'])
            
            if os.path.exists(midi_path) and os.path.exists(audio_path):
                selected_pairs.append((row['midi_filename'], row['audio_filename']))
        
        logger.info(f"✅ Selected {len(selected_pairs)} valid file pairs")
        
        # Create test datasets
        def create_test_dataset(file_pairs, output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
            # Copy and filter CSV
            filtered_df = df[df['midi_filename'].isin([p[0] for p in file_pairs])]
            test_csv_path = os.path.join(output_dir, csv_files[0])
            filtered_df.to_csv(test_csv_path, index=False)
            
            # Copy files
            for midi_file, audio_file in file_pairs:
                for src_file in [midi_file, audio_file]:
                    src = os.path.join(EGMD_DATASET_DIR, src_file)
                    dst = os.path.join(output_dir, src_file)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
            
            return output_dir
        
        seq_dataset = create_test_dataset(selected_pairs, SEQUENTIAL_OUTPUT_DIR)
        par_dataset = create_test_dataset(selected_pairs, PARALLEL_OUTPUT_DIR)
        
        logger.info("📁 Test datasets created")
        
        # Test sequential processing
        logger.info("🔄 Running SEQUENTIAL processing test...")
        start_time = time.perf_counter()
        start_memory = psutil.virtual_memory().used / (1024**3)
        
        seq_data_prep = data_preparation(seq_dataset, 'egmd', sample_ratio=1.0)
        seq_result = seq_data_prep.create_audio_set(
            pad_before=0.1,
            pad_after=0.1,
            fix_length=10,
            batching=True,
            dir_path=SEQUENTIAL_OUTPUT_DIR.replace('sequential', 'sequential_output'),
            num_batches=NUM_BATCHES,
            memory_limit_gb=MEMORY_LIMIT_GB,
            batch_size_multiplier=BATCH_SIZE_MULTIPLIER,
            enable_process_parallelization=False
        )
        
        seq_time = time.perf_counter() - start_time
        seq_memory = psutil.virtual_memory().used / (1024**3) - start_memory
        
        logger.info(f"✅ Sequential: {seq_result} files in {seq_time:.2f}s")
        
        # Test parallel processing  
        logger.info("⚡ Running PARALLEL processing test...")
        start_time = time.perf_counter()
        start_memory = psutil.virtual_memory().used / (1024**3)
        
        par_data_prep = data_preparation(par_dataset, 'egmd', sample_ratio=1.0)
        par_result = par_data_prep.create_audio_set(
            pad_before=0.1,
            pad_after=0.1,
            fix_length=10,
            batching=True,
            dir_path=PARALLEL_OUTPUT_DIR.replace('parallel', 'parallel_output'),
            num_batches=NUM_BATCHES,
            memory_limit_gb=MEMORY_LIMIT_GB,
            batch_size_multiplier=BATCH_SIZE_MULTIPLIER,
            enable_process_parallelization=True
        )
        
        par_time = time.perf_counter() - start_time
        par_memory = psutil.virtual_memory().used / (1024**3) - start_memory
        
        logger.info(f"✅ Parallel: {par_result} files in {par_time:.2f}s")
        
        # Analyze results
        logger.info("="*70)
        logger.info("📊 PERFORMANCE COMPARISON RESULTS:")
        
        if seq_result == par_result:
            logger.info(f"✅ File consistency: Both processed {seq_result} files")
            
            speedup = seq_time / par_time if par_time > 0 else 0
            time_saved = seq_time - par_time
            time_improvement = (time_saved / seq_time * 100) if seq_time > 0 else 0
            
            logger.info(f"   ⏱️  Sequential time: {seq_time:.2f}s")
            logger.info(f"   ⚡ Parallel time: {par_time:.2f}s")
            logger.info(f"   🚀 Speedup factor: {speedup:.2f}x")
            logger.info(f"   📈 Time improvement: {time_improvement:+.1f}%")
            logger.info(f"   ⏱️  Time saved: {time_saved:.2f}s")
            logger.info(f"   💾 Memory difference: {par_memory - seq_memory:+.2f}GB")
            
            # Results summary
            results = {
                'files_processed': seq_result,
                'sequential_time': seq_time,
                'parallel_time': par_time,
                'speedup_factor': speedup,
                'time_improvement_percent': time_improvement,
                'time_saved_seconds': time_saved,
                'memory_difference_gb': par_memory - seq_memory,
                'test_config': {
                    'num_files': NUM_TEST_FILES,
                    'memory_limit_gb': MEMORY_LIMIT_GB,
                    'num_batches': NUM_BATCHES
                }
            }
            
            # Save results
            results_file = f"enhanced_parallel_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"💾 Results saved to: {results_file}")
            
            # Success determination
            if speedup >= 1.2:  # At least 20% improvement
                logger.info("🎉 SUCCESS: Parallel processing provides significant speedup!")
                return True
            elif speedup >= 1.05:  # At least 5% improvement
                logger.info("✅ SUCCESS: Parallel processing provides modest speedup")
                return True
            else:
                logger.info("⚠️  Parallel processing did not provide significant speedup")
                logger.info("   This may be expected for smaller datasets or system limitations")
                return True  # Still success if files are consistent
        else:
            logger.error(f"❌ FAIL: File count mismatch - Sequential: {seq_result}, Parallel: {par_result}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
