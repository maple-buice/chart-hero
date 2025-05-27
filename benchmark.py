#!/usr/bin/env python3
"""
Benchmark script for testing system resource monitoring
This script performs various operations to generate predictable resource loads
"""

import argparse
import numpy as np
import time
import threading
import os
import sys
import shutil
import psutil
import random
from pathlib import Path

# Try to import PyTorch if available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not found, GPU benchmarks will be skipped")

# Global variables
should_exit = False
temp_dir = Path("benchmark_temp")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='System benchmark for resource monitoring testing')
    parser.add_argument('--duration', type=int, default=60, help='Benchmark duration in seconds')
    parser.add_argument('--cpu-load', type=float, default=50, help='Target CPU load percentage (0-100)')
    parser.add_argument('--memory-load', type=float, default=50, help='Target memory load percentage (0-100)')
    parser.add_argument('--disk-load', action='store_true', help='Generate disk I/O load')
    parser.add_argument('--network-load', action='store_true', help='Generate network I/O load')
    parser.add_argument('--gpu-load', action='store_true', help='Generate GPU load (requires PyTorch)')
    parser.add_argument('--all', action='store_true', help='Enable all load types')
    
    args = parser.parse_args()
    
    # If --all is specified, enable all load types
    if args.all:
        args.disk_load = True
        args.network_load = True
        args.gpu_load = True
    
    return args

def cpu_worker(target_percent=50):
    """CPU worker that tries to maintain a specific load percentage"""
    print(f"Starting CPU worker targeting {target_percent}% load")
    
    # Convert target percentage to a value between 0 and 1
    target = target_percent / 100.0
    
    while not should_exit:
        # Start time
        start_time = time.time()
        
        # Calculate how long to busy-wait vs sleep
        end_time = start_time + 1.0  # 1 second cycle
        busy_time = target
        sleep_time = 1.0 - busy_time
        
        # Busy-wait phase - perform CPU-intensive calculations
        busy_end = start_time + busy_time
        while time.time() < busy_end and not should_exit:
            # CPU-intensive operation
            np.random.random((500, 500)) @ np.random.random((500, 500))
        
        # Sleep phase
        if not should_exit and sleep_time > 0:
            time.sleep(sleep_time)

def memory_worker(target_percent=50):
    """Memory worker that allocates a percentage of available memory"""
    print(f"Starting memory worker targeting {target_percent}% load")
    
    # Get total system memory and calculate target usage
    total_memory = psutil.virtual_memory().total
    target_bytes = int((total_memory * target_percent) / 100)
    
    # Calculate how many 100MB chunks to allocate
    chunk_size = 100 * 1024 * 1024  # 100MB
    num_chunks = target_bytes // chunk_size
    
    print(f"Allocating {num_chunks} chunks of {chunk_size/(1024*1024)}MB (total: {target_bytes/(1024*1024*1024):.2f}GB)")
    
    # Hold references to allocated memory to prevent garbage collection
    memory_blocks = []
    
    try:
        # Gradually allocate memory to avoid system freeze
        for i in range(num_chunks):
            if should_exit:
                break
                
            # Allocate a chunk of memory
            block = bytearray(chunk_size)
            # Write some data to ensure it's actually allocated
            for j in range(0, len(block), 4096):
                block[j] = 1
                
            memory_blocks.append(block)
            
            if i % 10 == 0:
                print(f"Allocated {i+1}/{num_chunks} chunks ({(i+1)*chunk_size/(1024*1024*1024):.2f}GB)")
                time.sleep(0.1)  # Brief pause to allow system to respond
        
        print(f"Memory allocation complete. Holding {len(memory_blocks)} chunks")
        
        # Hold the memory until exit signal
        while not should_exit:
            # Periodically touch the memory to ensure it's not swapped out
            for i, block in enumerate(memory_blocks):
                if i % 10 == 0:  # Only touch every 10th block to reduce CPU usage
                    block[0] = random.randint(0, 255)
            time.sleep(1)
            
    except MemoryError:
        print("Memory allocation failed - system limit reached")
    finally:
        # Clear the memory blocks
        print("Releasing memory...")
        memory_blocks.clear()

def disk_worker():
    """Generate disk I/O load by writing and reading files"""
    print("Starting disk I/O worker")
    
    # Create a temporary directory for the benchmark
    os.makedirs(temp_dir, exist_ok=True)
    
    file_size = 100 * 1024 * 1024  # 100MB
    write_chunk = b'X' * 1024 * 1024  # 1MB chunks for writing
    
    try:
        while not should_exit:
            # Create random filename
            filename = temp_dir / f"benchmark_{random.randint(1, 1000)}.tmp"
            
            # Write phase
            print(f"Writing {file_size/(1024*1024)}MB to {filename}")
            with open(filename, 'wb') as f:
                for _ in range(file_size // len(write_chunk)):
                    if should_exit:
                        break
                    f.write(write_chunk)
                    # Small sleep to prevent 100% CPU usage
                    time.sleep(0.01)
            
            # Read phase
            if not should_exit and filename.exists():
                print(f"Reading from {filename}")
                with open(filename, 'rb') as f:
                    while chunk := f.read(1024 * 1024):  # Read in 1MB chunks
                        if should_exit:
                            break
                        # Do something with the data to ensure it's read
                        data_sum = sum(chunk[:1000])  # Just compute something
                        # Small sleep to prevent 100% CPU usage
                        time.sleep(0.01)
            
            # Delete the file
            if filename.exists():
                os.unlink(filename)
            
            # Brief pause between cycles
            time.sleep(0.5)
            
    finally:
        # Clean up temporary files
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                print(f"Removed temporary directory {temp_dir}")
            except Exception as e:
                print(f"Error cleaning up temporary directory: {e}")

def network_worker():
    """Simulate network activity by downloading data"""
    print("Starting network I/O worker")
    
    # Try to use curl or wget if available
    has_curl = shutil.which("curl") is not None
    has_wget = shutil.which("wget") is not None
    
    if not (has_curl or has_wget):
        print("Neither curl nor wget found. Using Python's urllib instead.")
    
    # List of URLs to download (use large files that are commonly available)
    urls = [
        "https://speed.hetzner.de/100MB.bin",  # 100MB test file
        "https://proof.ovh.net/files/100Mb.dat"  # Another 100MB test file
    ]
    
    try:
        while not should_exit:
            url = random.choice(urls)
            output_file = temp_dir / f"download_{random.randint(1, 1000)}.tmp"
            
            # Create temp dir if it doesn't exist
            os.makedirs(temp_dir, exist_ok=True)
            
            print(f"Downloading from {url}")
            
            # Use appropriate download method
            try:
                if has_curl:
                    # Using curl with progress and a timeout
                    os.system(f"curl -o {output_file} -L --max-time 10 {url} > /dev/null 2>&1")
                elif has_wget:
                    # Using wget with a timeout
                    os.system(f"wget -O {output_file} --timeout=10 {url} > /dev/null 2>&1")
                else:
                    # Using Python's urllib
                    import urllib.request
                    with urllib.request.urlopen(url, timeout=10) as response, open(output_file, 'wb') as out_file:
                        shutil.copyfileobj(response, out_file)
            except Exception as e:
                print(f"Download error: {e}")
            
            # Delete the downloaded file
            if output_file.exists():
                os.unlink(output_file)
            
            # Brief pause between downloads
            time.sleep(2)
            
    finally:
        # Clean up
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

def gpu_worker():
    """Generate GPU load using PyTorch operations"""
    if not HAS_TORCH:
        print("PyTorch not available, skipping GPU benchmark")
        return
    
    print("Starting GPU worker")
    
    # Check if MPS (Apple Silicon) or CUDA is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU device available, using CPU instead")
        device = torch.device("cpu")
    
    # Create some large tensors and perform operations
    try:
        # Start with moderately sized tensors
        size = 2000
        print(f"Creating tensors of size {size}x{size}")
        
        # Create tensors on the GPU
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        print(f"Starting GPU operations")
        while not should_exit:
            # Matrix multiplication (heavy GPU operation)
            c = torch.matmul(a, b)
            
            # More operations to keep the GPU busy
            d = torch.nn.functional.relu(c)
            e = torch.mean(d, dim=1)
            f = torch.cat([e.unsqueeze(0) for _ in range(10)], dim=0)
            
            # Force computation
            result = f.sum().item()
            
            # Sleep briefly to prevent 100% CPU usage in the Python process
            time.sleep(0.05)
            
    except Exception as e:
        print(f"GPU worker error: {e}")
    finally:
        # Clean up
        if device.type != "cpu":
            # Clear CUDA/MPS cache if applicable
            if device.type == "cuda" and hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
            elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

def main():
    global should_exit
    
    # Parse arguments
    args = parse_arguments()
    
    print("=== System Resource Benchmark ===")
    print(f"Duration: {args.duration} seconds")
    print(f"CPU Load: {args.cpu_load}%")
    print(f"Memory Load: {args.memory_load}%")
    print(f"Disk I/O: {'Enabled' if args.disk_load else 'Disabled'}")
    print(f"Network I/O: {'Enabled' if args.network_load else 'Disabled'}")
    print(f"GPU Load: {'Enabled' if args.gpu_load else 'Disabled'}")
    print("================================")
    
    # Start workers based on arguments
    workers = []
    
    # CPU worker
    cpu_thread = threading.Thread(target=cpu_worker, args=(args.cpu_load,))
    cpu_thread.daemon = True
    workers.append(cpu_thread)
    
    # Memory worker
    memory_thread = threading.Thread(target=memory_worker, args=(args.memory_load,))
    memory_thread.daemon = True
    workers.append(memory_thread)
    
    # Disk I/O worker
    if args.disk_load:
        disk_thread = threading.Thread(target=disk_worker)
        disk_thread.daemon = True
        workers.append(disk_thread)
    
    # Network I/O worker
    if args.network_load:
        network_thread = threading.Thread(target=network_worker)
        network_thread.daemon = True
        workers.append(network_thread)
    
    # GPU worker
    if args.gpu_load and HAS_TORCH:
        gpu_thread = threading.Thread(target=gpu_worker)
        gpu_thread.daemon = True
        workers.append(gpu_thread)
    
    # Start all workers
    for worker in workers:
        worker.start()
    
    # Run for the specified duration
    start_time = time.time()
    try:
        while time.time() - start_time < args.duration:
            elapsed = time.time() - start_time
            remaining = args.duration - elapsed
            print(f"Benchmark running... {elapsed:.1f}s elapsed, {remaining:.1f}s remaining", end="\r")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    finally:
        # Signal workers to exit
        should_exit = True
        print("\nStopping benchmark...")
        
        # Wait for workers to finish (with timeout)
        for worker in workers:
            worker.join(timeout=5.0)
        
        # Clean up temporary directory if it exists
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Error cleaning up temporary directory: {e}")
        
        print("Benchmark complete")

if __name__ == "__main__":
    main()
