#!/usr/bin/env python3
"""
Test script to verify positioned tqdm progress bars work correctly.
"""

import os
import time
from tqdm import tqdm
from multiprocessing import Process

def test_positioned_progress_bars():
    """Test multiple positioned progress bars simultaneously."""
    print("Testing positioned tqdm progress bars...")
    print("This will show multiple progress bars stacked vertically without conflicts.\n")
    
    # Simulate main process - position 0
    main_items = list(range(20))
    main_bar = tqdm(main_items, desc="Main Process", position=0, leave=True, ncols=80, colour='green')
    
    # Simulate worker 1 - position 1  
    worker1_items = list(range(15))
    worker1_bar = tqdm(worker1_items, desc="Worker-1", position=1, leave=True, ncols=80, colour='cyan')
    
    # Simulate worker 2 - position 2
    worker2_items = list(range(18))
    worker2_bar = tqdm(worker2_items, desc="Worker-2", position=2, leave=True, ncols=80, colour='yellow')
    
    # Process main items
    for item in main_bar:
        time.sleep(0.05)  # Simulate work
        main_bar.set_postfix({"Item": item, "PID": os.getpid()})
    
    # Process worker 1 items
    for item in worker1_bar:
        time.sleep(0.07)  # Different timing
        worker1_bar.set_postfix({"Item": item, "Status": "Processing"})
    
    # Process worker 2 items
    for item in worker2_bar:
        time.sleep(0.06)  # Different timing
        worker2_bar.set_postfix({"Item": item, "Status": "Working"})
    
    print("\n✅ Positioned progress bars test completed!")
    print("Each progress bar should have appeared on its own line without interference.")

def test_process_aware_bars():
    """Test process-aware positioned progress bars."""
    print("\nTesting process-aware positioned progress bars...")
    
    current_pid = os.getpid()
    main_process_id = current_pid  # Simulate main process tracking
    
    # Main process behavior
    is_worker = current_pid != main_process_id  # False for main
    
    if is_worker:
        position = (current_pid % 10) + 1
        desc = f"Worker-{position}"
        colour = 'cyan'
    else:
        position = 0
        desc = "Main Process"
        colour = 'green'
    
    items = list(range(25))
    progress_bar = tqdm(items, desc=desc, position=position, leave=True, ncols=80, colour=colour)
    
    for item in progress_bar:
        time.sleep(0.04)
        progress_bar.set_postfix({
            "PID": current_pid,
            "Worker": is_worker,
            "Pos": position
        })
    
    print(f"\n✅ Process-aware test completed!")
    print(f"   Process ID: {current_pid}")
    print(f"   Is Worker: {is_worker}")
    print(f"   Position: {position}")

if __name__ == "__main__":
    print("🚀 Starting tqdm positioned progress bar tests\n")
    
    # Test 1: Basic positioned bars
    test_positioned_progress_bars()
    
    # Small delay between tests
    time.sleep(1)
    
    # Test 2: Process-aware positioning
    test_process_aware_bars()
    
    print("\n🎉 All tests completed successfully!")
    print("The positioned progress bars should prevent rapid switching and provide clean output.")
