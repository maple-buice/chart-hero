#!/usr/bin/env python3
"""
Test script to demonstrate callback integration with parallel and sequential processing.

This script shows how to use custom progress callbacks with the data preparation pipeline.
"""

import os
import sys
import logging
import time
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_training.data_preparation import EgmdDataPrep

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProgressTracker:
    """Custom progress tracker that can be used as a callback."""
    
    def __init__(self, name="Processing"):
        self.name = name
        self.start_time = time.time()
        self.last_update = 0
        self.update_interval = 5.0  # Update every 5 seconds
        self.total_callbacks = 0
        
    def __call__(self, current, total, details):
        """
        Progress callback function.
        
        Args:
            current: Current progress count
            total: Total items to process
            details: Dictionary with additional information
        """
        self.total_callbacks += 1
        current_time = time.time()
        
        # Only update every few seconds to avoid spam
        if current_time - self.last_update >= self.update_interval or current == total:
            self.last_update = current_time
            
            elapsed = current_time - self.start_time
            progress_percent = (current / total) * 100 if total > 0 else 0
            
            # Calculate ETA
            if current > 0:
                avg_time_per_item = elapsed / current
                remaining_items = total - current
                eta_seconds = remaining_items * avg_time_per_item
                eta_str = f"{eta_seconds/60:.1f}m" if eta_seconds > 60 else f"{eta_seconds:.1f}s"
            else:
                eta_str = "calculating..."
            
            # Format the progress message
            mode = details.get('mode', 'unknown')
            if mode == 'parallel':
                worker_info = f"Workers: {details.get('workers_active', 0)}/{details.get('total_workers', 0)}"
                current_worker = details.get('worker_id', 'unknown')
                message = f"[{self.name}] {progress_percent:.1f}% ({current}/{total}) | {worker_info} | Worker {current_worker} | ETA: {eta_str}"
            else:
                current_file = details.get('current_file', 'unknown')
                records = details.get('total_records', 0)
                memory_gb = details.get('memory_gb', 0)
                message = f"[{self.name}] {progress_percent:.1f}% ({current}/{total}) | Records: {records} | Memory: {memory_gb:.1f}GB | File: {current_file[:30]}... | ETA: {eta_str}"
            
            logger.info(message)
            
            # Final completion message
            if current == total and details.get('completed', current == total):
                total_time = elapsed / 60
                files_per_min = current / (elapsed / 60) if elapsed > 0 else 0
                logger.info(f"[{self.name}] ✅ COMPLETED in {total_time:.1f} minutes | "
                          f"Rate: {files_per_min:.1f} files/min | Total callbacks: {self.total_callbacks}")

def test_callback_integration():
    """Test callback integration with both sequential and parallel modes."""
    
    logger.info("=" * 80)
    logger.info("TESTING CALLBACK INTEGRATION WITH DATA PREPARATION")
    logger.info("=" * 80)
    
    # Check if we have any MIDI/audio data to work with
    test_dirs = [
        "/Users/maple/Repos/chart-hero/test_data",
        "/Users/maple/Repos/chart-hero/data",
        "/content/drive/MyDrive/chart-hero/data"  # Colab path
    ]
    
    data_dir = None
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            data_dir = test_dir
            break
    
    if not data_dir:
        logger.warning("No test data directory found. Creating minimal test setup...")
        
        # Create a minimal test directory with dummy files for demonstration
        data_dir = "/Users/maple/Repos/chart-hero/test_callback_data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Create some dummy MIDI and WAV files (empty for demo)
        for i in range(5):
            midi_file = os.path.join(data_dir, f"test_song_{i}.mid")
            wav_file = os.path.join(data_dir, f"test_song_{i}.wav")
            
            # Create empty files for demonstration
            if not os.path.exists(midi_file):
                with open(midi_file, 'wb') as f:
                    f.write(b'')  # Empty MIDI file (won't process correctly but shows callback)
            if not os.path.exists(wav_file):
                with open(wav_file, 'wb') as f:
                    f.write(b'')  # Empty WAV file
        
        logger.info(f"Created test data in: {data_dir}")
    
    logger.info(f"Using data directory: {data_dir}")
    
    try:
        # Initialize data preparation
        data_prep = EgmdDataPrep(data_dir)
        logger.info(f"Found {len(data_prep.midi_wav_map)} MIDI/WAV pairs")
        
        if len(data_prep.midi_wav_map) == 0:
            logger.warning("No valid MIDI/WAV pairs found. Callback demo will be limited.")
            return
        
        # Test 1: Sequential processing with callback
        logger.info("\n" + "="*50)
        logger.info("TEST 1: SEQUENTIAL PROCESSING WITH CALLBACK")
        logger.info("="*50)
        
        sequential_tracker = ProgressTracker("Sequential")
        
        try:
            result_sequential = data_prep.create_audio_set(
                pad_before=0.01,
                pad_after=0.01,
                fix_length=1.0,  # Short length for quick test
                batching=True,
                dir_path=os.path.join(data_dir, "sequential_output"),
                memory_limit_gb=4,
                enable_process_parallelization=False,  # Force sequential
                progress_callback=sequential_tracker
            )
            logger.info(f"Sequential processing completed: {result_sequential} files processed")
        except Exception as e:
            logger.error(f"Sequential processing failed: {e}")
        
        # Test 2: Parallel processing with callback (if conditions are met)
        logger.info("\n" + "="*50)
        logger.info("TEST 2: PARALLEL PROCESSING WITH CALLBACK")
        logger.info("="*50)
        
        parallel_tracker = ProgressTracker("Parallel")
        
        try:
            result_parallel = data_prep.create_audio_set(
                pad_before=0.01,
                pad_after=0.01,
                fix_length=1.0,  # Short length for quick test
                batching=True,
                dir_path=os.path.join(data_dir, "parallel_output"),
                memory_limit_gb=16,  # Higher memory for parallel
                enable_process_parallelization=True,  # Try parallel
                progress_callback=parallel_tracker
            )
            logger.info(f"Parallel processing completed: {result_parallel} files processed")
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
        
        # Test 3: Demonstrate callback error handling
        logger.info("\n" + "="*50)
        logger.info("TEST 3: CALLBACK ERROR HANDLING")
        logger.info("="*50)
        
        def failing_callback(current, total, details):
            """A callback that raises an error to test error handling."""
            if current == 2:  # Fail on second call
                raise ValueError("Simulated callback error")
            logger.info(f"Callback working: {current}/{total}")
        
        try:
            result_error = data_prep.create_audio_set(
                pad_before=0.01,
                pad_after=0.01,
                fix_length=0.5,  # Even shorter for error test
                batching=False,  # No batching for simpler test
                enable_process_parallelization=False,
                progress_callback=failing_callback
            )
            logger.info(f"Error handling test completed: {result_error} files processed")
        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
        
    except Exception as e:
        logger.error(f"Test setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_callback_integration()
