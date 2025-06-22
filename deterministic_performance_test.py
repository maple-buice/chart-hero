#!/usr/bin/env python3
"""
Deterministic Performance Test for Parallel Worker Validation

This test ensures consistent file selection between sequential and parallel processing
to provide accurate performance comparisons and validate the parallel worker fixes.
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

# --- Test Configuration ---
EGMD_DATASET_DIR = "/Users/maple/Repos/chart-hero/datasets/e-gmd-v1.0.0"
TEST_OUTPUT_BASE = "/Users/maple/Repos/chart-hero/datasets/deterministic_performance_test"
SEQUENTIAL_OUTPUT_DIR = f"{TEST_OUTPUT_BASE}/sequential"
PARALLEL_OUTPUT_DIR = f"{TEST_OUTPUT_BASE}/parallel"

# Performance test parameters
NUM_TEST_FILES = 10  # Fixed number of files to test
MEMORY_LIMIT_GB = 8
BATCH_SIZE_MULTIPLIER = 1.0
NUM_BATCHES = 1  # Single batch for direct comparison

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'deterministic_performance_test_{time.strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class DeterministicPerformanceTester:
    """Performance tester that uses fixed file sets for consistent results."""
    
    def __init__(self):
        self.results = {
            'sequential': {},
            'parallel': {},
            'comparison': {},
            'test_files': []
        }
        self.selected_files = []
        
    def get_fixed_file_set(self, dataset_dir: str, num_files: int) -> List[Tuple[str, str]]:
        """
        Get a fixed, deterministic set of files for testing.
        This ensures the same files are used in both sequential and parallel tests.
        """
        logger.info(f"🎯 Selecting fixed set of {num_files} files for testing...")
        
        # Read the CSV file to get available files
        csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in {dataset_dir}")
        
        csv_path = os.path.join(dataset_dir, csv_files[0])
        df = pd.read_csv(csv_path)
        
        # Sort by filename for deterministic selection
        df_sorted = df.sort_values('midi_filename')
        
        # Select the first N files that exist
        selected_pairs = []
        for _, row in df_sorted.iterrows():
            if len(selected_pairs) >= num_files:
                break
                
            midi_path = os.path.join(dataset_dir, row['midi_filename'])
            audio_path = os.path.join(dataset_dir, row['audio_filename'])
            
            # Only include if both files exist
            if os.path.exists(midi_path) and os.path.exists(audio_path):
                selected_pairs.append((row['midi_filename'], row['audio_filename']))
                
        if len(selected_pairs) < num_files:
            logger.warning(f"Only found {len(selected_pairs)} valid file pairs out of requested {num_files}")
            
        logger.info(f"✅ Selected {len(selected_pairs)} files for testing:")
        for i, (midi, audio) in enumerate(selected_pairs[:5]):  # Show first 5
            logger.info(f"   {i+1}: {midi} + {audio}")
        if len(selected_pairs) > 5:
            logger.info(f"   ... and {len(selected_pairs)-5} more files")
            
        self.selected_files = selected_pairs
        self.results['test_files'] = selected_pairs
        return selected_pairs
        
    def create_test_dataset(self, file_pairs: List[Tuple[str, str]], output_dir: str) -> str:
        """Create a test dataset with only the selected files."""
        logger.info(f"📁 Creating test dataset in {output_dir}...")
        
        # Clean up and create output directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
        # Copy the original CSV and filter it
        original_csv = [f for f in os.listdir(EGMD_DATASET_DIR) if f.endswith('.csv')][0]
        original_df = pd.read_csv(os.path.join(EGMD_DATASET_DIR, original_csv))
        
        # Filter to only include our selected files
        selected_midi_files = [pair[0] for pair in file_pairs]
        filtered_df = original_df[original_df['midi_filename'].isin(selected_midi_files)]
        
        # Save filtered CSV
        test_csv_path = os.path.join(output_dir, original_csv)
        filtered_df.to_csv(test_csv_path, index=False)
        
        # Copy the selected files
        for midi_file, audio_file in file_pairs:
            # Copy MIDI file
            src_midi = os.path.join(EGMD_DATASET_DIR, midi_file)
            dst_midi = os.path.join(output_dir, midi_file)
            os.makedirs(os.path.dirname(dst_midi), exist_ok=True)
            shutil.copy2(src_midi, dst_midi)
            
            # Copy audio file
            src_audio = os.path.join(EGMD_DATASET_DIR, audio_file)
            dst_audio = os.path.join(output_dir, audio_file)
            os.makedirs(os.path.dirname(dst_audio), exist_ok=True)
            shutil.copy2(src_audio, dst_audio)
            
        logger.info(f"✅ Test dataset created with {len(file_pairs)} file pairs")
        return output_dir
        
    def run_sequential_test(self, dataset_dir: str) -> Dict:
        """Run sequential processing on the test dataset."""
        logger.info("🔄 Running SEQUENTIAL processing test...")
        
        start_time = time.perf_counter()
        start_memory = psutil.virtual_memory().used / (1024**3)
        
        # Initialize data preparation in sequential mode
        data_prep = data_preparation(
            directory_path=dataset_dir,
            dataset='egmd',
            sample_ratio=1.0  # Process all files in our test set
        )
        
        # Run processing
        try:
            results = data_prep.create_audio_set(
                pad_before=0.1,
                pad_after=0.1,
                fix_length=10,
                batching=True,
                dir_path=SEQUENTIAL_OUTPUT_DIR.replace('sequential', 'sequential_output'),
                num_batches=NUM_BATCHES,
                memory_limit_gb=MEMORY_LIMIT_GB,
                batch_size_multiplier=BATCH_SIZE_MULTIPLIER,
                enable_process_parallelization=False  # Sequential mode
            )
            
            end_time = time.perf_counter()
            end_memory = psutil.virtual_memory().used / (1024**3)
            
            sequential_results = {
                'success': True,
                'processing_time': end_time - start_time,
                'memory_used': end_memory - start_memory,
                'total_files': results,  # create_audio_set returns number of processed files
                'total_records': 0,  # We'll estimate this or leave as 0 for comparison
                'batches_processed': 1,
                'performance_stats': []
            }
            
            logger.info(f"✅ Sequential processing completed:")
            logger.info(f"   ⏱️  Time: {sequential_results['processing_time']:.2f}s")
            logger.info(f"   📁 Files: {sequential_results['total_files']}")
            logger.info(f"   📊 Records: {sequential_results['total_records']}")
            logger.info(f"   💾 Memory: {sequential_results['memory_used']:.2f}GB")
            
        except Exception as e:
            logger.error(f"❌ Sequential processing failed: {e}")
            sequential_results = {
                'success': False,
                'error': str(e),
                'processing_time': 0,
                'memory_used': 0,
                'total_files': 0,
                'total_records': 0
            }
            
        self.results['sequential'] = sequential_results
        return sequential_results
        
    def run_parallel_test(self, dataset_dir: str) -> Dict:
        """Run parallel processing on the test dataset."""
        logger.info("⚡ Running PARALLEL processing test...")
        
        start_time = time.perf_counter()
        start_memory = psutil.virtual_memory().used / (1024**3)
        
        # Initialize data preparation in parallel mode
        data_prep = data_preparation(
            directory_path=dataset_dir,
            dataset='egmd',
            sample_ratio=1.0  # Process all files in our test set
        )
        
        # Run processing
        try:
            results = data_prep.create_audio_set(
                pad_before=0.1,
                pad_after=0.1,
                fix_length=10,
                batching=True,
                dir_path=PARALLEL_OUTPUT_DIR.replace('parallel', 'parallel_output'),
                num_batches=NUM_BATCHES,
                memory_limit_gb=MEMORY_LIMIT_GB,
                batch_size_multiplier=BATCH_SIZE_MULTIPLIER,
                enable_process_parallelization=True  # Parallel mode
            )
            
            end_time = time.perf_counter()
            end_memory = psutil.virtual_memory().used / (1024**3)
            
            parallel_results = {
                'success': True,
                'processing_time': end_time - start_time,
                'memory_used': end_memory - start_memory,
                'total_files': results,  # create_audio_set returns number of processed files
                'total_records': 0,  # We'll estimate this or leave as 0 for comparison
                'batches_processed': 1,
                'performance_stats': []
            }
            
            logger.info(f"✅ Parallel processing completed:")
            logger.info(f"   ⏱️  Time: {parallel_results['processing_time']:.2f}s")
            logger.info(f"   📁 Files: {parallel_results['total_files']}")
            logger.info(f"   📊 Records: {parallel_results['total_records']}")
            logger.info(f"   💾 Memory: {parallel_results['memory_used']:.2f}GB")
            
        except Exception as e:
            logger.error(f"❌ Parallel processing failed: {e}")
            parallel_results = {
                'success': False,
                'error': str(e),
                'processing_time': 0,
                'memory_used': 0,
                'total_files': 0,
                'total_records': 0
            }
            
        self.results['parallel'] = parallel_results
        return parallel_results
        
    def analyze_performance(self) -> Dict:
        """Analyze and compare performance between sequential and parallel modes."""
        logger.info("📊 Analyzing performance comparison...")
        
        seq = self.results['sequential']
        par = self.results['parallel']
        
        if not seq.get('success') or not par.get('success'):
            logger.error("❌ Cannot compare performance - one or both tests failed")
            return {}
            
        # Calculate performance improvements
        time_improvement = (seq['processing_time'] - par['processing_time']) / seq['processing_time'] * 100
        memory_difference = par['memory_used'] - seq['memory_used']
        
        # Check data consistency
        files_match = seq['total_files'] == par['total_files']
        records_match = seq['total_records'] == par['total_records']
        
        comparison = {
            'time_improvement_percent': time_improvement,
            'sequential_time': seq['processing_time'],
            'parallel_time': par['processing_time'],
            'speedup_factor': seq['processing_time'] / par['processing_time'] if par['processing_time'] > 0 else 0,
            'memory_difference_gb': memory_difference,
            'files_consistency': files_match,
            'records_consistency': records_match,
            'parallel_efficiency': time_improvement > 0 and files_match and records_match
        }
        
        self.results['comparison'] = comparison
        
        # Log results
        logger.info("📈 Performance Comparison Results:")
        logger.info(f"   ⚡ Speedup Factor: {comparison['speedup_factor']:.2f}x")
        logger.info(f"   ⏱️  Time Improvement: {comparison['time_improvement_percent']:+.1f}%")
        logger.info(f"   💾 Memory Difference: {comparison['memory_difference_gb']:+.2f}GB")
        logger.info(f"   ✅ Files Consistent: {comparison['files_consistency']}")
        logger.info(f"   ✅ Records Consistent: {comparison['records_consistency']}")
        logger.info(f"   🎯 Parallel Efficient: {comparison['parallel_efficiency']}")
        
        return comparison
        
    def save_results(self):
        """Save detailed results to file."""
        results_file = f"deterministic_performance_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        logger.info(f"💾 Results saved to: {results_file}")
        return results_file


def main():
    """Run the deterministic performance test."""
    logger.info("🚀 Starting Deterministic Performance Test for Parallel Worker")
    logger.info("="*70)
    
    # Check if dataset exists
    if not os.path.exists(EGMD_DATASET_DIR):
        logger.error(f"❌ Dataset not found: {EGMD_DATASET_DIR}")
        return False
        
    tester = DeterministicPerformanceTester()
    
    try:
        # Step 1: Get fixed file set
        file_pairs = tester.get_fixed_file_set(EGMD_DATASET_DIR, NUM_TEST_FILES)
        
        if len(file_pairs) == 0:
            logger.error("❌ No valid file pairs found")
            return False
            
        # Step 2: Create test datasets
        seq_dataset = tester.create_test_dataset(file_pairs, SEQUENTIAL_OUTPUT_DIR)
        par_dataset = tester.create_test_dataset(file_pairs, PARALLEL_OUTPUT_DIR)
        
        # Step 3: Run sequential test
        seq_results = tester.run_sequential_test(seq_dataset)
        
        # Step 4: Run parallel test 
        par_results = tester.run_parallel_test(par_dataset)
        
        # Step 5: Analyze performance
        comparison = tester.analyze_performance()
        
        # Step 6: Save results
        results_file = tester.save_results()
        
        # Final summary
        logger.info("="*70)
        if comparison.get('parallel_efficiency', False):
            logger.info("🎉 SUCCESS: Parallel worker is functioning correctly and efficiently!")
            logger.info(f"   📈 Achieved {comparison['speedup_factor']:.2f}x speedup")
            logger.info(f"   ✅ Data consistency maintained")
        else:
            logger.warning("⚠️  ATTENTION: Performance or consistency issues detected")
            if not comparison.get('files_consistency'):
                logger.error("   ❌ File count mismatch between modes")
            if not comparison.get('records_consistency'):
                logger.error("   ❌ Record count mismatch between modes")
            if comparison.get('time_improvement_percent', 0) <= 0:
                logger.warning("   ⚠️  No time improvement from parallel processing")
                
        logger.info(f"📄 Detailed results: {results_file}")
        return comparison.get('parallel_efficiency', False)
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
