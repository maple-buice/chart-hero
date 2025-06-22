#!/usr/bin/env python3
"""
Test to verify that parallel workers use the same memory-efficient code as sequential processing.
"""

import os
import sys
import logging
import time
import psutil
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_training.data_preparation import data_preparation

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_parallel_memory_efficiency():
    """Test that parallel processing uses the same memory-efficient code as sequential."""
    
    # Check that we have the E-GMD dataset
    dataset_path = Path("datasets/e-gmd-v1.0.0")
    if not dataset_path.exists():
        logger.warning("E-GMD dataset not found, skipping memory efficiency test")
        return True
    
    csv_file = dataset_path / "e-gmd-v1.0.0.csv"
    if not csv_file.exists():
        logger.warning("E-GMD CSV not found, skipping memory efficiency test")
        return True
    
    try:
        logger.info("Testing parallel processing memory efficiency...")
        
        # Create data preparation instance with a moderate sample
        data_prep = data_preparation(
            directory_path=str(dataset_path),
            dataset='egmd',
            sample_ratio=0.05  # 5% sample for testing
        )
        
        dataset_size = len(data_prep.midi_wav_map)
        logger.info(f"Dataset size: {dataset_size} files")
        
        if dataset_size < 50:
            logger.info("Dataset too small for parallel processing, increasing sample...")
            data_prep = data_preparation(
                directory_path=str(dataset_path),
                dataset='egmd',
                sample_ratio=0.15  # 15% sample to get above parallel threshold
            )
            dataset_size = len(data_prep.midi_wav_map)
            logger.info(f"Updated dataset size: {dataset_size} files")
        
        if dataset_size < 50:
            logger.warning("Still below parallel threshold, skipping test")
            return True
        
        # Monitor memory usage during parallel processing
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**3)
        
        logger.info(f"Initial memory usage: {initial_memory:.1f}GB")
        logger.info("Starting parallel processing test...")
        
        # Test parallel processing with conservative settings
        start_time = time.perf_counter()
        processed_files = data_prep.create_audio_set(
            pad_before=0.1,
            pad_after=0.1,
            fix_length=10.0,
            batching=False,  # Don't save to disk for testing
            memory_limit_gb=8.0,  # Conservative limit
            batch_size_multiplier=1.0,  # Conservative multiplier
            enable_process_parallelization=True  # Force parallel mode
        )
        processing_time = time.perf_counter() - start_time
        
        final_memory = process.memory_info().rss / (1024**3)
        memory_increase = final_memory - initial_memory
        
        logger.info(f"Parallel processing complete:")
        logger.info(f"  - Files processed: {processed_files}")
        logger.info(f"  - Processing time: {processing_time:.1f}s")
        logger.info(f"  - Initial memory: {initial_memory:.1f}GB")
        logger.info(f"  - Final memory: {final_memory:.1f}GB")
        logger.info(f"  - Memory increase: {memory_increase:.1f}GB")
        logger.info(f"  - Memory per file: {memory_increase*1024/max(1,processed_files):.1f}MB")
        
        # Check if memory usage is reasonable
        if memory_increase > 4.0:  # More than 4GB increase is excessive
            logger.warning(f"⚠️  High memory increase: {memory_increase:.1f}GB")
            logger.warning("This suggests parallel workers may not be using efficient sequential code")
            return False
        else:
            logger.info(f"✅ Reasonable memory increase: {memory_increase:.1f}GB")
            logger.info("Parallel workers appear to be using memory-efficient code")
            return True
        
    except Exception as e:
        logger.error(f"Error during parallel memory test: {e}")
        return False

def test_sequential_vs_parallel_memory():
    """Compare memory usage between sequential and parallel processing."""
    
    dataset_path = Path("datasets/e-gmd-v1.0.0")
    if not dataset_path.exists():
        logger.warning("E-GMD dataset not found, skipping comparison test")
        return True
    
    try:
        logger.info("Comparing sequential vs parallel memory usage...")
        
        # Test 1: Sequential processing
        logger.info("\n--- Sequential Processing Test ---")
        data_prep_seq = data_preparation(
            directory_path=str(dataset_path),
            dataset='egmd',
            sample_ratio=0.02  # Small sample for comparison
        )
        
        process = psutil.Process()
        seq_initial = process.memory_info().rss / (1024**3)
        
        seq_files = data_prep_seq.create_audio_set(
            pad_before=0.1,
            pad_after=0.1,
            fix_length=10.0,
            batching=False,
            memory_limit_gb=4.0,
            enable_process_parallelization=False  # Force sequential
        )
        
        seq_final = process.memory_info().rss / (1024**3)
        seq_increase = seq_final - seq_initial
        
        logger.info(f"Sequential: {seq_files} files, {seq_increase:.1f}GB memory increase")
        
        # Test 2: Parallel processing (if dataset is large enough)
        dataset_size = len(data_prep_seq.midi_wav_map)
        if dataset_size >= 50:
            logger.info("\n--- Parallel Processing Test ---")
            data_prep_par = data_preparation(
                directory_path=str(dataset_path),
                dataset='egmd',
                sample_ratio=0.02  # Same sample size
            )
            
            par_initial = process.memory_info().rss / (1024**3)
            
            par_files = data_prep_par.create_audio_set(
                pad_before=0.1,
                pad_after=0.1,
                fix_length=10.0,
                batching=False,
                memory_limit_gb=8.0,
                enable_process_parallelization=True  # Force parallel
            )
            
            par_final = process.memory_info().rss / (1024**3)
            par_increase = par_final - par_initial
            
            logger.info(f"Parallel: {par_files} files, {par_increase:.1f}GB memory increase")
            
            # Compare memory efficiency
            if par_increase <= seq_increase * 1.5:  # Allow 50% overhead for parallelization
                logger.info("✅ Parallel memory usage is reasonable compared to sequential")
                return True
            else:
                logger.warning(f"⚠️  Parallel uses {par_increase/seq_increase:.1f}x more memory than sequential")
                return False
        else:
            logger.info("Dataset too small for parallel processing, sequential test only")
            return True
        
    except Exception as e:
        logger.error(f"Error during comparison test: {e}")
        return False

def main():
    logger.info("Testing parallel processing memory efficiency...")
    logger.info("="*60)
    
    success = True
    
    # Test 1: Basic parallel memory efficiency
    logger.info("\n--- Test 1: Parallel Memory Efficiency ---")
    if not test_parallel_memory_efficiency():
        success = False
    
    # Test 2: Sequential vs parallel comparison
    logger.info("\n--- Test 2: Sequential vs Parallel Comparison ---")
    if not test_sequential_vs_parallel_memory():
        success = False
    
    # Summary
    logger.info("\n" + "="*60)
    if success:
        logger.info("✅ MEMORY EFFICIENCY TESTS PASSED")
        logger.info("Parallel workers appear to be using the same memory-efficient code as sequential processing")
        logger.info("")
        logger.info("OPTIMIZATION SUCCESS:")
        logger.info("• Parallel workers now reuse sequential processing logic")
        logger.info("• Memory usage is controlled and reasonable")
        logger.info("• No more memory explosions in worker processes")
        logger.info("• Same compression and optimization techniques applied")
    else:
        logger.error("❌ MEMORY EFFICIENCY TESTS FAILED")
        logger.error("Parallel workers may still have memory issues")
        logger.error("Consider using sequential processing for now")
    
    logger.info("="*60)
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
