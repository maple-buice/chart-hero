#!/usr/bin/env python3
"""
Parallel processing worker module for audio data preparation.

This module contains standalone functions that REUSE the exact same logic
from the sequential processing to ensure memory efficiency and consistency.
"""

import os
import gc
import time
import psutil
import logging
import warnings
import threading
from typing import Tuple, Optional, Dict, Any, List
import pandas as pd
import numpy as np
from tqdm import tqdm

# Progress reporting system - workers report to main process
class ProgressReporter:
    """Thread-safe progress reporting for worker processes."""
    
    def __init__(self, worker_id: str, total_files: int, callback_queue=None):
        self.worker_id = worker_id
        self.total_files = total_files
        self.processed_files = 0
        self.total_records = 0
        self.callback_queue = callback_queue
        self.start_time = time.perf_counter()
    
    def update(self, files_processed: int = 0, records_added: int = 0):
        """Update progress and report to main process via callback queue."""
        if files_processed > 0:
            self.processed_files += files_processed
        if records_added > 0:
            self.total_records += records_added
        
        # Send progress update to main process if queue available
        if self.callback_queue:
            progress_data = {
                'worker_id': self.worker_id,
                'processed_files': self.processed_files,
                'total_files': self.total_files,
                'total_records': self.total_records,
                'elapsed_time': time.perf_counter() - self.start_time,
                'current_pid': os.getpid()
            }
            try:
                self.callback_queue.put(('progress', progress_data), block=False)
            except:
                pass  # Queue full, skip this update
    
    def log(self, level: str, message: str):
        """Send log message to main process."""
        if self.callback_queue:
            try:
                self.callback_queue.put(('log', {
                    'worker_id': self.worker_id,
                    'level': level,
                    'message': message,
                    'pid': os.getpid()
                }), block=False)
            except:
                pass
    
    def close(self):
        """Signal completion to main process."""
        if self.callback_queue:
            try:
                self.callback_queue.put(('complete', {
                    'worker_id': self.worker_id,
                    'processed_files': self.processed_files,
                    'total_records': self.total_records,
                    'total_time': time.perf_counter() - self.start_time
                }), block=False)
            except:
                pass

# Suppress warnings and reduce log verbosity in worker processes
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore')  # Suppress all warnings in workers

# Aggressively suppress ALL logging in worker processes
logging.getLogger().setLevel(logging.CRITICAL + 1)  # Disable all logging
for logger_name in ['model_training.data_preparation', 'librosa', 'soundfile', '__main__']:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL + 1)

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL + 1)  # Complete silence in workers


def chunk_processor_worker(args: Tuple) -> Tuple[int, int, float, List[Dict], List[Dict]]:
    """
    Process a chunk of file pairs using the EXACT SAME logic as sequential processing.
    
    This ensures memory efficiency by reusing the proven sequential code instead
    of implementing parallel-specific logic that might have memory leaks.
    
    Args:
        args: Tuple containing (file_chunk_data, processing_params, directory_path, callback_queue)
    
    Returns:
        Tuple of (processed_files, total_records, processing_time, performance_stats, record_data)
    """
    try:
        # Unpack args - callback_queue is optional for backward compatibility
        if len(args) == 4:
            file_chunk, processing_params, directory_path, callback_queue = args
        else:
            file_chunk, processing_params, directory_path = args
            callback_queue = None
            
        chunk_start_time = time.perf_counter()
        
        # Import the sequential processing function directly
        # This ensures we use EXACTLY the same memory-efficient code
        from functools import partial
        
        # Create a minimal data_preparation instance just to access the methods
        # but with aggressive memory cleanup
        temp_data_prep = create_minimal_data_prep_worker(directory_path)
        
        # Create the same processing function as sequential mode
        process_func = partial(
            temp_data_prep._process_file_pair_worker_wrapper,
            pad_before=processing_params['pad_before'],
            pad_after=processing_params['pad_after'],
            fix_length=processing_params['fix_length'],
            memory_limit_gb=processing_params['memory_limit_gb']
        )
        
        processed_files = 0
        total_records = 0
        performance_stats = []
        all_records = []
        
        # Calculate unique worker position based on PID
        current_pid = os.getpid()
        # Use last 2 digits of PID for better uniqueness
        worker_num = current_pid % 100
        worker_id = f"worker_{current_pid}"
        
        # Handle both DataFrame (for testing) and list (for production parallel processing)
        if isinstance(file_chunk, pd.DataFrame):
            # DataFrame format - iterate using iterrows()
            file_iterator = enumerate(file_chunk.iterrows())
            def extract_row(i, item):
                index, row = item
                return row
            total_files = len(file_chunk)
        else:
            # List format - iterate directly over dictionaries
            file_iterator = enumerate(file_chunk)
            def extract_row(i, item):
                return pd.Series(item)  # Convert dict to Series for compatibility
            total_files = len(file_chunk)
        
        # Create progress reporter for this worker
        progress_reporter = ProgressReporter(worker_id, total_files, callback_queue)
        
        for i, item in file_iterator:
            try:
                file_start_time = time.perf_counter()
                
                # Extract row data in consistent format
                row = extract_row(i, item)
                
                # Check if row has required columns
                if 'midi_filename' not in row:
                    progress_reporter.log('error', f"Row missing midi_filename. Available columns: {list(row.index)}")
                    continue
                
                # Use the exact same processing logic as sequential
                result = process_func(row)
                
                file_time = time.perf_counter() - file_start_time
                
                if result is not None and not result.empty:
                    processed_files += 1
                    records_count = len(result)
                    total_records += records_count
                    
                    # Convert DataFrame to records list efficiently (same as sequential)
                    for _, record in result.iterrows():
                        all_records.append(record.to_dict())
                    
                    performance_stats.append({
                        'file_id': os.path.basename(row.get('midi_filename', 'unknown')),
                        'records': records_count,
                        'time': file_time,
                        'perf': {
                            'total_time': file_time,
                            'note_count': records_count,
                        }
                    })
                    
                    # Clear result immediately (same as sequential)
                    del result
                    
                    # Report progress to main process
                    progress_reporter.update(files_processed=1, records_added=records_count)
                else:
                    progress_reporter.log('warning', f"Failed to process: {row.get('midi_filename', 'unknown')}")
                    
            except Exception as e:
                progress_reporter.log('error', f"Error processing row: {e}")
                continue
        
        # Signal completion to main process
        progress_reporter.close()
        
        # Clean up the temporary data_prep instance
        del temp_data_prep
        gc.collect()
        
        chunk_time = time.perf_counter() - chunk_start_time
        progress_reporter.log('info', f"Worker chunk complete: {processed_files} files, {total_records} records, {chunk_time:.1f}s")
        
        return processed_files, total_records, chunk_time, performance_stats, all_records
        
    except Exception as e:
        # Send error to main process if possible
        if 'callback_queue' in locals() and callback_queue:
            try:
                callback_queue.put(('log', {
                    'worker_id': worker_id if 'worker_id' in locals() else 'unknown',
                    'level': 'error',
                    'message': f"Chunk processor failed: {e}",
                    'pid': os.getpid()
                }), block=False)
            except:
                pass
        
        # Clean up on error
        gc.collect()
        return 0, 0, 0, [], []


def create_minimal_data_prep_worker(directory_path: str):
    """
    Create a minimal data_preparation instance for worker processes.
    
    This creates only what's needed to access the _process_file_pair method
    without the full initialization overhead that includes duration calculation.
    """
    # Import here to avoid circular imports
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Import the main class but avoid full initialization
    from model_training.data_preparation import data_preparation
    
    # Create a minimal instance with just what's needed for processing
    class MinimalDataPrep:
        def __init__(self, directory_path):
            self.directory_path = directory_path
            # Track main process ID for tqdm management (worker-aware)
            self._main_process_id = os.getpid()  # In worker, this will be worker PID
            
        def _process_file_pair(self, row, pad_before, pad_after, fix_length, memory_limit_gb):
            """Call the original _process_file_pair method without full init."""
            # Create a temporary full instance with disabled progress bars
            from model_training.data_preparation import data_preparation
            
            # Ensure progress bars are disabled in worker processes
            temp_full_instance = data_preparation(
                self.directory_path, 
                'egmd', 
                sample_ratio=1.0,  # Use full dataset mapping from main process
                disable_progress=True  # Critical: disable all progress bars
            )
            return temp_full_instance._process_file_pair(row, pad_before, pad_after, fix_length, memory_limit_gb)
            
        def _process_file_pair_worker_wrapper(self, row, pad_before, pad_after, fix_length, memory_limit_gb):
            """Wrapper to call the main _process_file_pair method."""
            return self._process_file_pair(row, pad_before, pad_after, fix_length, memory_limit_gb)
    
    return MinimalDataPrep(directory_path)


# Legacy compatibility functions (redirecting to the new unified approach)

def process_file_pair_with_data_worker(args: Tuple) -> Optional[List[Dict]]:
    """Legacy function - now redirects to unified chunk processor for consistency."""
    try:
        midi_filename, audio_filename, directory_path, processing_params = args
        
        # Create a single-row chunk to process
        import pandas as pd
        single_file_chunk = pd.DataFrame([{
            'midi_filename': midi_filename,
            'audio_filename': audio_filename,
            'track_id': os.path.basename(midi_filename).replace('.mid', '')
        }])
        
        # Use the unified chunk processor
        processed_files, total_records, _, _, record_data = chunk_processor_worker((
            single_file_chunk, processing_params, directory_path
        ))
        
        return record_data if record_data else None
        
    except Exception as e:
        logger.error(f"Legacy worker failed: {e}")
        return None


def process_file_pair_worker(args: Tuple) -> Tuple[bool, str, int, float, Dict[str, Any]]:
    """Legacy worker - redirects to unified approach for consistency."""
    try:
        midi_path, audio_path, pad_before, pad_after, fix_length, memory_limit_gb = args
        
        file_start_time = time.perf_counter()
        
        # Extract filenames and directory
        directory_path = os.path.dirname(midi_path)
        midi_filename = os.path.relpath(midi_path, directory_path)
        audio_filename = os.path.relpath(audio_path, directory_path)
        
        processing_params = {
            'pad_before': pad_before,
            'pad_after': pad_after,
            'fix_length': fix_length,
            'memory_limit_gb': memory_limit_gb
        }
        
        records = process_file_pair_with_data_worker((
            midi_filename,
            audio_filename,
            directory_path,
            processing_params
        ))
        
        file_time = time.perf_counter() - file_start_time
        file_id = os.path.basename(midi_path)
        
        if records is not None and len(records) > 0:
            perf_data = {
                'total_time': file_time,
                'note_count': len(records),
                'memory_used': len(records) * 0.1  # Estimate
            }
            return True, file_id, len(records), file_time, perf_data
        else:
            return False, file_id, 0, file_time, {}
            
    except Exception as e:
        logger.error(f"Legacy worker failed: {e}")
        return False, str(args[0]) if args else "unknown", 0, 0, {}
