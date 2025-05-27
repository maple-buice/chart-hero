#!/usr/bin/env python3
"""
Script to verify process detection in monitoring.
This script mimics a transformer training process and checks if it can be detected.
"""

import os
import sys
import time
import subprocess
import signal
import psutil

def run_test_process():
    """Run a test process that mimics a transformer training process"""
    print("Starting test process that mimics transformer training...")
    # Use the actual test_monitoring.py script which should be detectable
    test_process = subprocess.Popen(
        ["python3", "test_monitoring.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return test_process

def run_monitoring(pid=None):
    """Run the monitoring script with various detection methods"""
    base_cmd = ["python3", "monitor_training.py", "--interval", "1", "--compact"]
    
    if pid:
        # Direct PID monitoring
        cmd = base_cmd + ["--watch-pid", str(pid)]
        mode = "PID-specific"
    else:
        # Auto-detection
        cmd = base_cmd + ["--auto-detect", "--enable-debug"]
        mode = "auto-detection"
    
    print(f"Starting monitoring with {mode}...")
    monitor_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return monitor_process

def check_process_detection(test_pid, monitor_process, duration=10):
    """Check if the monitoring process can detect the test process"""
    print(f"Checking if test process (PID: {test_pid}) is detected for {duration} seconds...")
    
    start_time = time.time()
    detected = False
    
    while time.time() - start_time < duration:
        # Read output from monitoring process
        try:
            output = monitor_process.stdout.readline().decode('utf-8')
            if output:
                print(f"Monitor output: {output.strip()}")
                if str(test_pid) in output:
                    detected = True
                    print(f"Success! Test process PID {test_pid} was detected in monitoring output.")
                    break
        except Exception as e:
            print(f"Error reading monitor output: {e}")
        
        time.sleep(0.5)
    
    return detected

def main():
    print("=== Process Detection Verification ===")
    
    # Run the test process
    test_process = run_test_process()
    test_pid = test_process.pid
    print(f"Test process started with PID: {test_pid}")
    
    try:
        # First, test direct PID monitoring
        monitor_process = run_monitoring(pid=test_pid)
        pid_detection = check_process_detection(test_pid, monitor_process)
        monitor_process.terminate()
        
        # Give processes time to terminate
        time.sleep(2)
        
        # Then, test auto-detection
        monitor_process = run_monitoring()
        auto_detection = check_process_detection(test_pid, monitor_process)
        monitor_process.terminate()
        
        # Summary
        print("\n=== Detection Results ===")
        print(f"Direct PID detection: {'✅ Successful' if pid_detection else '❌ Failed'}")
        print(f"Auto-detection: {'✅ Successful' if auto_detection else '❌ Failed'}")
        
        if not auto_detection:
            print("\nRecommendation: Use direct PID monitoring with --watch-pid instead of auto-detection")
            # Get command line of the test process to help diagnose
            try:
                proc = psutil.Process(test_pid)
                print(f"\nProcess command line: {' '.join(proc.cmdline())}")
            except:
                pass
    
    finally:
        # Clean up processes
        if test_process:
            test_process.terminate()
            try:
                test_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                test_process.kill()
        
        if monitor_process:
            monitor_process.terminate()
            try:
                monitor_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor_process.kill()

if __name__ == "__main__":
    main()
