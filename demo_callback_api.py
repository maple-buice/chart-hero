#!/usr/bin/env python3
"""
Demo script showing callback API usage with the enhanced data preparation pipeline.

This script demonstrates the callback functionality without requiring real audio/MIDI data.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_progress_callback():
    """
    Demonstrate the progress callback API format.
    
    This shows the expected callback signature and the data structure
    that callbacks will receive from both sequential and parallel processing.
    """
    
    logger.info("=" * 60)
    logger.info("DATA PREPARATION CALLBACK API DEMONSTRATION")
    logger.info("=" * 60)
    
    logger.info("\\nCallback Function Signature:")
    logger.info("  def progress_callback(current, total, details):")
    logger.info("    # current: int - Current progress (files processed)")
    logger.info("    # total: int - Total files to process") 
    logger.info("    # details: dict - Additional information")
    
    logger.info("\\nSequential Mode Details Dictionary:")
    sequential_details = {
        'mode': 'sequential',
        'current_file': 'song_123.mid',
        'batch_records': 1250,
        'total_records': 5400,
        'files_processed': 54,
        'memory_gb': 2.8,
        'is_worker': False,
        'pid': 12345
    }
    
    for key, value in sequential_details.items():
        logger.info(f"    {key}: {value}")
    
    logger.info("\\nParallel Mode Details Dictionary:")
    parallel_details = {
        'mode': 'parallel',
        'worker_id': 'worker_2',
        'workers_active': 3,
        'total_workers': 3,
        'current_pid': 12347,
        'worker_files': 18,
        'worker_records': 890,
        'completed': False,
        'final_results': False
    }
    
    for key, value in parallel_details.items():
        logger.info(f"    {key}: {value}")
    
    logger.info("\\nExample Callback Implementation:")
    logger.info("```python")
    logger.info("def my_progress_callback(current, total, details):")
    logger.info("    progress_percent = (current / total) * 100")
    logger.info("    mode = details.get('mode', 'unknown')")
    logger.info("    ")
    logger.info("    if mode == 'sequential':")
    logger.info("        current_file = details.get('current_file', 'unknown')")
    logger.info("        records = details.get('total_records', 0)")
    logger.info("        print(f'Sequential: {progress_percent:.1f}% | {current_file} | {records} records')")
    logger.info("    ")
    logger.info("    elif mode == 'parallel':")
    logger.info("        workers = f\"{details.get('workers_active', 0)}/{details.get('total_workers', 0)}\"")
    logger.info("        worker_id = details.get('worker_id', 'unknown')")
    logger.info("        print(f'Parallel: {progress_percent:.1f}% | Workers: {workers} | Current: {worker_id}')")
    logger.info("```")
    
    logger.info("\\nUsage Example:")
    logger.info("```python")
    logger.info("from model_training.data_preparation import EgmdDataPrep")
    logger.info("")
    logger.info("data_prep = EgmdDataPrep('/path/to/data')")
    logger.info("")
    logger.info("# Sequential with callback")
    logger.info("result = data_prep.create_audio_set(")
    logger.info("    pad_before=0.02,")
    logger.info("    pad_after=0.02,")
    logger.info("    batching=True,")
    logger.info("    enable_process_parallelization=False,")
    logger.info("    progress_callback=my_progress_callback")
    logger.info(")")
    logger.info("")
    logger.info("# Parallel with callback (if conditions are met)")
    logger.info("result = data_prep.create_audio_set(")
    logger.info("    pad_before=0.02,")
    logger.info("    pad_after=0.02,")
    logger.info("    batching=True,")
    logger.info("    memory_limit_gb=16,")
    logger.info("    enable_process_parallelization=True,")
    logger.info("    progress_callback=my_progress_callback")
    logger.info(")")
    logger.info("```")

def demo_callback_classes():
    """Demonstrate some useful callback class implementations."""
    
    logger.info("\\n" + "=" * 60)
    logger.info("EXAMPLE CALLBACK IMPLEMENTATIONS")
    logger.info("=" * 60)
    
    class SimpleProgressLogger:
        """Simple callback that logs progress every N items."""
        
        def __init__(self, log_interval=10):
            self.log_interval = log_interval
            self.start_time = time.time()
        
        def __call__(self, current, total, details):
            if current % self.log_interval == 0 or current == total:
                elapsed = time.time() - self.start_time
                rate = current / elapsed if elapsed > 0 else 0
                mode = details.get('mode', 'unknown')
                logger.info(f"[{mode.upper()}] Progress: {current}/{total} ({rate:.1f} files/sec)")
    
    class DetailedProgressTracker:
        """Detailed callback that tracks timing and memory usage."""
        
        def __init__(self):
            self.start_time = time.time()
            self.checkpoints = []
        
        def __call__(self, current, total, details):
            now = time.time()
            elapsed = now - self.start_time
            
            checkpoint = {
                'timestamp': now,
                'current': current,
                'total': total,
                'elapsed': elapsed,
                'mode': details.get('mode'),
                'memory_gb': details.get('memory_gb', 0)
            }
            self.checkpoints.append(checkpoint)
            
            # Log every 25% or on completion
            progress_pct = (current / total) * 100
            if progress_pct % 25 < (100 / total) or current == total:
                logger.info(f"Checkpoint: {progress_pct:.1f}% complete in {elapsed:.1f}s")
        
        def get_summary(self):
            """Return processing summary."""
            if not self.checkpoints:
                return "No data collected"
            
            final = self.checkpoints[-1]
            total_time = final['elapsed']
            total_files = final['current']
            avg_rate = total_files / total_time if total_time > 0 else 0
            
            return f"Processed {total_files} files in {total_time:.1f}s (avg: {avg_rate:.2f} files/sec)"
    
    class PerformanceMonitor:
        """Callback that monitors processing performance and detects issues."""
        
        def __init__(self, slow_threshold=2.0):
            self.slow_threshold = slow_threshold  # seconds per file
            self.last_time = time.time()
            self.last_current = 0
            self.slow_periods = []
        
        def __call__(self, current, total, details):
            now = time.time()
            files_processed = current - self.last_current
            time_elapsed = now - self.last_time
            
            if files_processed > 0 and time_elapsed > 0:
                rate = files_processed / time_elapsed
                time_per_file = time_elapsed / files_processed
                
                if time_per_file > self.slow_threshold:
                    self.slow_periods.append({
                        'start_file': self.last_current,
                        'end_file': current,
                        'time_per_file': time_per_file,
                        'details': details
                    })
                    mode = details.get('mode', 'unknown')
                    logger.warning(f"[{mode.upper()}] Slow processing detected: "
                                 f"{time_per_file:.2f}s per file (threshold: {self.slow_threshold}s)")
            
            self.last_time = now
            self.last_current = current
        
        def get_slow_periods_summary(self):
            """Return summary of slow processing periods."""
            if not self.slow_periods:
                return "No slow periods detected"
            
            total_slow_periods = len(self.slow_periods)
            avg_slow_time = sum(p['time_per_file'] for p in self.slow_periods) / total_slow_periods
            
            return f"Detected {total_slow_periods} slow periods, avg {avg_slow_time:.2f}s per file"
    
    logger.info("\\nExample callback classes demonstrated above:")
    logger.info("  1. SimpleProgressLogger - Basic progress logging")
    logger.info("  2. DetailedProgressTracker - Timing and checkpoint tracking")
    logger.info("  3. PerformanceMonitor - Performance issue detection")
    
    logger.info("\\nThese can be used as:")
    logger.info("  progress_callback = SimpleProgressLogger(log_interval=25)")
    logger.info("  progress_callback = DetailedProgressTracker()")
    logger.info("  progress_callback = PerformanceMonitor(slow_threshold=3.0)")

def demo_integration_points():
    """Show where callbacks are integrated in the processing pipeline."""
    
    logger.info("\\n" + "=" * 60)
    logger.info("CALLBACK INTEGRATION POINTS")
    logger.info("=" * 60)
    
    logger.info("\\nCallbacks are triggered at these points:")
    logger.info("\\n1. SEQUENTIAL MODE:")
    logger.info("   - After each file is processed")
    logger.info("   - Provides current file info, memory usage, record counts")
    logger.info("   - Called from main processing thread")
    
    logger.info("\\n2. PARALLEL MODE:")
    logger.info("   - Worker processes send progress updates via queue")
    logger.info("   - Main process aggregates worker progress")
    logger.info("   - Provides worker status, PIDs, and aggregated progress")
    logger.info("   - Called from main thread (thread-safe)")
    
    logger.info("\\n3. ERROR HANDLING:")
    logger.info("   - Callback errors are caught and logged as warnings")
    logger.info("   - Processing continues even if callback fails")
    logger.info("   - No impact on core data processing reliability")
    
    logger.info("\\n4. MEMORY SAFETY:")
    logger.info("   - Callbacks receive copies of data, not references")
    logger.info("   - Callback execution time is not included in performance metrics")
    logger.info("   - Callbacks cannot interfere with worker process memory management")

if __name__ == "__main__":
    logger.info("🚀 ENHANCED DATA PREPARATION CALLBACK SYSTEM")
    demo_progress_callback()
    demo_callback_classes()
    demo_integration_points()
    
    logger.info("\\n" + "=" * 60)
    logger.info("✅ CALLBACK SYSTEM DEMONSTRATION COMPLETE")
    logger.info("=" * 60)
    logger.info("\\nThe data preparation pipeline now supports comprehensive")
    logger.info("progress callbacks for both sequential and parallel processing modes.")
    logger.info("\\nUse test_callback_integration.py to see the callbacks in action!")
