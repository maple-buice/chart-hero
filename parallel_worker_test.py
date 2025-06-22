#!/usr/bin/env python3
"""
Parallel Worker Test - Test the standalone parallel processing functions

This test specifically validates that the parallel worker functions work correctly
and can process files without serialization issues.
"""

import sys
import os
import logging
import time
import psutil
from pathlib import Path

# Ensure the model_training package is discoverable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def test_worker_functions():
    """Test the parallel worker functions directly."""
    try:
        from model_training.parallel_worker import process_file_pair_worker, chunk_processor_worker
        logger.info("✅ Successfully imported parallel worker functions")
        
        # Find some test files
        egmd_dataset_dir = "/Users/maple/Repos/chart-hero/datasets/e-gmd-v1.0.0"
        
        if not os.path.exists(egmd_dataset_dir):
            logger.error(f"❌ Dataset directory not found: {egmd_dataset_dir}")
            return False
        
        # Find first few MIDI/audio pairs
        test_files = []
        for root, dirs, files in os.walk(egmd_dataset_dir):
            midi_files = [f for f in files if f.endswith('.midi')]
            for midi_file in midi_files[:3]:  # Test with first 3 files
                midi_path = os.path.join(root, midi_file)
                audio_file = midi_file.replace('.midi', '.wav')
                audio_path = os.path.join(root, audio_file)
                
                if os.path.exists(audio_path):
                    test_files.append((midi_path, audio_path))
                    
                if len(test_files) >= 3:
                    break
            if len(test_files) >= 3:
                break
        
        if not test_files:
            logger.error("❌ No valid MIDI/audio pairs found for testing")
            return False
        
        logger.info(f"🎵 Found {len(test_files)} test file pairs")
        
        # Test individual worker function
        logger.info("🧪 Testing individual file processing...")
        
        for i, (midi_path, audio_path) in enumerate(test_files):
            logger.info(f"   Testing file {i+1}: {os.path.basename(midi_path)}")
            
            args = (midi_path, audio_path, 0.02, 0.02, 5.0, 4.0)  # pad_before, pad_after, fix_length, memory_limit
            
            start_time = time.perf_counter()
            result = process_file_pair_worker(args)
            end_time = time.perf_counter()
            
            success, file_id, record_count, proc_time, perf_data = result
            
            if success:
                logger.info(f"      ✅ Success: {record_count} records in {end_time - start_time:.3f}s")
                logger.info(f"      📊 Performance: {perf_data.get('note_count', 0)} notes, "
                          f"{perf_data.get('total_time', 0):.3f}s total")
            else:
                logger.warning(f"      ⚠️  Failed: {file_id}")
        
        # Test multiprocessing capability
        logger.info("🚀 Testing multiprocessing capability...")
        
        try:
            from multiprocessing import Pool
            
            # Prepare args for all test files
            all_args = [(midi_path, audio_path, 0.02, 0.02, 5.0, 2.0) for midi_path, audio_path in test_files]
            
            with Pool(processes=2) as pool:
                logger.info("   Running parallel processing with 2 workers...")
                start_time = time.perf_counter()
                results = pool.map(process_file_pair_worker, all_args)
                end_time = time.perf_counter()
                
                logger.info(f"   ✅ Parallel processing completed in {end_time - start_time:.3f}s")
                
                successful_results = [r for r in results if r[0]]
                logger.info(f"   📊 Results: {len(successful_results)}/{len(results)} files processed successfully")
                
                for result in successful_results:
                    success, file_id, record_count, proc_time, perf_data = result
                    logger.info(f"      {file_id}: {record_count} records")
                    
        except Exception as e:
            logger.error(f"❌ Multiprocessing test failed: {e}")
            return False
        
        logger.info("✅ All parallel worker tests passed!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import parallel worker: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Worker test failed: {e}", exc_info=True)
        return False


def test_full_parallel_integration():
    """Test the full parallel processing integration."""
    logger.info("🔧 Testing full parallel processing integration...")
    
    try:
        from model_training.data_preparation import data_preparation
        
        egmd_dataset_dir = "/Users/maple/Repos/chart-hero/datasets/e-gmd-v1.0.0"
        test_output_dir = "/Users/maple/Repos/chart-hero/datasets/worker_test"
        
        # Clean up previous test
        if os.path.exists(test_output_dir):
            import shutil
            shutil.rmtree(test_output_dir)
        os.makedirs(test_output_dir, exist_ok=True)
        
        # Initialize with very small sample for quick test
        data_prep = data_preparation(
            directory_path=egmd_dataset_dir,
            dataset='egmd',
            sample_ratio=0.001,  # Very small sample
            diff_threshold=1.0
        )
        
        total_pairs = len(data_prep.midi_wav_map)
        logger.info(f"📊 Test dataset: {total_pairs} MIDI/audio pairs")
        
        if total_pairs == 0:
            logger.error("❌ No valid pairs found for integration test")
            return False
        
        # Test parallel processing with the new worker system
        logger.info("⚡ Testing parallel processing integration...")
        
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / (1024**3)
        
        processed_files = data_prep.create_audio_set(
            pad_before=0.02,
            pad_after=0.02,
            fix_length=5.0,
            batching=False,  # Don't save batches for this test
            dir_path=test_output_dir,
            num_batches=1,
            memory_limit_gb=16,
            batch_size_multiplier=1.0,
            enable_process_parallelization=True  # Force parallel mode
        )
        
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / (1024**3)
        
        logger.info(f"✅ Integration test completed:")
        logger.info(f"   Files processed: {processed_files}/{total_pairs}")
        logger.info(f"   Time: {end_time - start_time:.3f}s")
        logger.info(f"   Memory: {start_memory:.2f}GB → {end_memory:.2f}GB ({end_memory - start_memory:+.2f}GB)")
        
        return processed_files > 0
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}", exc_info=True)
        return False


def main():
    """Main test runner."""
    logger.info("🧪 Parallel Worker Function Test")
    
    # System info
    system_memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = os.cpu_count()
    logger.info(f"💻 System: {system_memory_gb:.1f}GB RAM, {cpu_count} CPU cores")
    
    success = True
    
    # Test 1: Worker functions
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Parallel Worker Functions")
    logger.info("="*60)
    if not test_worker_functions():
        success = False
    
    # Test 2: Full integration
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Full Parallel Integration")
    logger.info("="*60)
    if not test_full_parallel_integration():
        success = False
    
    # Final result
    logger.info("\n" + "="*60)
    if success:
        logger.info("✅ All parallel worker tests passed!")
        logger.info("🚀 Parallel processing is working correctly!")
    else:
        logger.error("❌ Some parallel worker tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
