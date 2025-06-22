#!/usr/bin/env python3
"""
Example of how to integrate callback-based progress reporting 
with the existing parallel processing system.
"""

import sys
import os
import time
import queue
from multiprocessing import Queue, Pool
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tqdm import tqdm
from model_training.parallel_worker import chunk_processor_worker
import pandas as pd

def create_progress_manager(callback_queue: Queue, num_workers: int):
    """Create a progress manager that handles worker callbacks."""
    worker_bars = {}
    completed_workers = 0
    
    def update_progress():
        nonlocal completed_workers
        
        try:
            while True:
                message_type, data = callback_queue.get_nowait()
                
                if message_type == 'progress':
                    worker_id = data['worker_id']
                    
                    # Create progress bar for new worker
                    if worker_id not in worker_bars:
                        position = len(worker_bars)
                        worker_bars[worker_id] = tqdm(
                            total=data['total_files'],
                            desc=f"Worker-{worker_id.split('_')[1][-2:]}",
                            position=position,
                            leave=True,
                            ncols=100,
                            colour='cyan',
                            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]{postfix}'
                        )
                    
                    # Update progress bar
                    pbar = worker_bars[worker_id]
                    pbar.n = data['processed_files']
                    pbar.set_postfix({
                        'Files': data['processed_files'],
                        'Records': data['total_records'],
                        'PID': str(data['current_pid'])[-4:]  # Last 4 digits
                    })
                    pbar.refresh()
                    
                elif message_type == 'log':
                    # Print log messages without disrupting progress bars
                    level = data.get('level', 'info').upper()
                    message = data.get('message', '')
                    print(f"\n[{level}] {message}")
                    
                    # Refresh all bars after log message
                    for pbar in worker_bars.values():
                        pbar.refresh()
                    
                elif message_type == 'complete':
                    # Worker completed
                    worker_id = data['worker_id']
                    if worker_id in worker_bars:
                        pbar = worker_bars[worker_id]
                        pbar.n = pbar.total  # Ensure 100%
                        pbar.close()
                    completed_workers += 1
                    print(f"\n✅ Worker completed: {data['processed_files']} files, {data['total_records']} records")
                    
        except queue.Empty:
            pass
        except Exception as e:
            print(f"\nError in progress manager: {e}")
    
    return update_progress, lambda: completed_workers >= num_workers

def demo_parallel_with_callbacks():
    """Demonstrate parallel processing with callback-based progress."""
    print("🚀 Demo: Parallel processing with callback-based progress")
    print("="*60)
    
    # Create some test data
    test_files = [
        {'midi_filename': f'test_{i}.mid', 'audio_filename': f'test_{i}.wav'}
        for i in range(12)  # 12 files total
    ]
    
    # Split into chunks for 3 workers
    chunk_size = 4
    chunks = [test_files[i:i+chunk_size] for i in range(0, len(test_files), chunk_size)]
    
    processing_params = {
        'pad_before': 0.1,
        'pad_after': 0.1,
        'fix_length': 10.0,
        'memory_limit_gb': 8.0
    }
    
    # Create callback queue
    callback_queue = Queue()
    
    # Prepare arguments for workers (add callback queue)
    worker_args = [
        (pd.DataFrame(chunk), processing_params, "datasets/e-gmd-v1.0.0", callback_queue)
        for chunk in chunks
    ]
    
    # Create progress manager
    update_progress, is_complete = create_progress_manager(callback_queue, len(chunks))
    
    print(f"Starting {len(chunks)} workers with {len(test_files)} total files...")
    
    # Start workers with callbacks
    with Pool(processes=len(chunks)) as pool:
        # Start async processing
        result = pool.map_async(chunk_processor_worker, worker_args)
        
        # Monitor progress in main thread
        while not result.ready() or not is_complete():
            update_progress()
            time.sleep(0.1)  # Check for updates every 100ms
        
        # Get final results
        update_progress()  # Final update
        worker_results = result.get()
    
    print(f"\n🎉 Parallel processing completed!")
    print(f"Results: {worker_results}")

if __name__ == "__main__":
    # Note: This is a demo - the actual files don't exist
    print("📝 Note: This is a demo showing the callback architecture.")
    print("For real testing, use the actual data preparation script.")
    print("The key improvement is that ALL progress bars are managed by the main process!")
