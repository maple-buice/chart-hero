#!/usr/bin/env python3
"""
Test the new persistent progress bar system with multiple workers.
"""

import sys
import os
import time
import threading
from multiprocessing import Process
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tqdm import tqdm
from model_training.parallel_worker import create_persistent_progress_bar, register_progress_bar, start_progress_refresher

def simulate_worker(worker_id: int, num_items: int):
    """Simulate a worker process with a progress bar."""
    position = worker_id + 10
    desc = f"Worker-{worker_id}"
    
    # Create persistent progress bar
    progress_bar = create_persistent_progress_bar(
        worker_id=f"worker_{worker_id}",
        total=num_items,
        desc=desc,
        position=position
    )
    
    for i in range(num_items):
        # Simulate work
        time.sleep(0.2)
        
        # Update progress
        progress_bar.update(1)
        progress_bar.set_postfix({
            'Item': i+1,
            'Worker': worker_id,
            'Status': 'Processing'
        })
        
        # Simulate occasional log messages that might interfere
        if i % 5 == 0:
            print(f"[LOG] Worker {worker_id} processed item {i+1}")
    
    progress_bar.close()
    print(f"✅ Worker {worker_id} completed!")

def test_persistent_progress_bars():
    """Test multiple progress bars with log interference."""
    print("🧪 Testing persistent progress bars with log interference")
    print("="*60)
    
    # Start the background refresher
    start_progress_refresher()
    
    # Create multiple worker threads
    workers = []
    for i in range(3):  # 3 workers
        worker = threading.Thread(
            target=simulate_worker, 
            args=(i+1, 10)  # Each processes 10 items
        )
        workers.append(worker)
        worker.start()
    
    # Wait for all workers to complete
    for worker in workers:
        worker.join()
    
    print("\n🎉 All workers completed!")
    print("You should have seen 3 persistent progress bars that stayed visible")
    print("even when log messages appeared between them.")

if __name__ == "__main__":
    test_persistent_progress_bars()
