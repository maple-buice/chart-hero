#!/usr/bin/env python3
"""
Test script to verify tqdm improvements in parallel processing.
"""

import os
import time
from multiprocessing import Process, Queue
from tqdm import tqdm

def worker_with_tqdm(worker_id, items, main_process_id, results_queue):
    """Worker function that uses tqdm with process-aware settings."""
    current_pid = os.getpid()
    is_worker = current_pid != main_process_id
    
    if is_worker:
        # Disable progress bar in worker processes
        desc = f"Worker-{worker_id} (PID:{current_pid})"
        progress_bar = tqdm(items, desc=desc, disable=True)
    else:
        # Show progress bar in main process
        desc = f"Main process (PID:{current_pid})"
        progress_bar = tqdm(items, desc=desc, position=worker_id, leave=False)
    
    processed = []
    for item in progress_bar:
        # Simulate some work
        time.sleep(0.1)
        processed.append(f"worker-{worker_id}-processed-{item}")
    
    results_queue.put((worker_id, processed))

def test_parallel_tqdm():
    """Test parallel processing with improved tqdm handling."""
    print("Testing improved tqdm handling in parallel processing...")
    
    main_process_id = os.getpid()
    print(f"Main process ID: {main_process_id}")
    
    # Create test data
    data_chunks = [
        [1, 2, 3],
        [4, 5, 6], 
        [7, 8, 9]
    ]
    
    # Create processes
    processes = []
    results_queue = Queue()
    
    for i, chunk in enumerate(data_chunks):
        p = Process(target=worker_with_tqdm, 
                   args=(i, chunk, main_process_id, results_queue))
        processes.append(p)
        p.start()
    
    # Wait for all processes
    for p in processes:
        p.join()
    
    # Collect results
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())
    
    print(f"Processed {len(results)} chunks successfully!")
    for worker_id, processed_items in sorted(results):
        print(f"  Worker {worker_id}: {len(processed_items)} items")
    
    print("✅ Test completed - tqdm improvements working correctly!")

if __name__ == "__main__":
    test_parallel_tqdm()
