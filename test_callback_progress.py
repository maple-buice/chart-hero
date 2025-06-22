#!/usr/bin/env python3
"""
Test the new callback-based progress reporting system.
Main process handles all progress bars and logging.
"""

import sys
import os
import time
import threading
import queue
from multiprocessing import Process, Queue
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tqdm import tqdm
from model_training.parallel_worker import ProgressReporter

def simulate_worker_with_callback(worker_id: int, num_items: int, callback_queue: Queue):
    """Simulate a worker process that reports progress via callback."""
    # Create progress reporter
    reporter = ProgressReporter(f"worker_{worker_id}", num_items, callback_queue)
    
    for i in range(num_items):
        # Simulate work
        time.sleep(0.3)
        
        # Report progress
        reporter.update(files_processed=1, records_added=i*10)
        
        # Occasional log messages
        if i % 3 == 0:
            reporter.log('info', f"Worker {worker_id} processed item {i+1}")
    
    reporter.close()

def main_process_progress_handler(callback_queue: Queue, num_workers: int):
    """Main process handles all progress bars and logging."""
    print("🧪 Testing callback-based progress reporting")
    print("="*60)
    
    # Track worker progress bars
    worker_bars = {}
    completed_workers = 0
    
    while completed_workers < num_workers:
        try:
            # Check for messages from workers
            message_type, data = callback_queue.get(timeout=0.1)
            
            if message_type == 'progress':
                worker_id = data['worker_id']
                
                # Create progress bar for new worker
                if worker_id not in worker_bars:
                    worker_bars[worker_id] = tqdm(
                        total=data['total_files'],
                        desc=f"Worker-{worker_id.split('_')[1]}",
                        position=len(worker_bars),
                        leave=True,
                        ncols=100,
                        colour='cyan'
                    )
                
                # Update progress bar
                pbar = worker_bars[worker_id]
                pbar.n = data['processed_files']
                pbar.set_postfix({
                    'Files': data['processed_files'],
                    'Records': data['total_records'],
                    'PID': data['current_pid']
                })
                pbar.refresh()
                
            elif message_type == 'log':
                # Handle log messages
                level = data.get('level', 'info').upper()
                message = data.get('message', '')
                worker_id = data.get('worker_id', 'unknown')
                print(f"[{level}] {worker_id}: {message}")
                
            elif message_type == 'complete':
                # Worker completed
                worker_id = data['worker_id']
                if worker_id in worker_bars:
                    worker_bars[worker_id].close()
                completed_workers += 1
                print(f"✅ {worker_id} completed! ({data['processed_files']} files, {data['total_records']} records)")
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error handling callback: {e}")
    
    print("\n🎉 All workers completed!")
    print("Progress bars were managed by main process - no conflicts!")

def test_callback_progress():
    """Test the callback-based progress system."""
    # Create callback queue
    callback_queue = Queue()
    
    # Start workers
    num_workers = 3
    workers = []
    for i in range(num_workers):
        worker = Process(
            target=simulate_worker_with_callback,
            args=(i+1, 5, callback_queue)  # Each processes 5 items
        )
        workers.append(worker)
        worker.start()
    
    # Handle progress in main process
    main_process_progress_handler(callback_queue, num_workers)
    
    # Wait for all workers to complete
    for worker in workers:
        worker.join()

if __name__ == "__main__":
    test_callback_progress()
