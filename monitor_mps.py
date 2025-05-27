#!/usr/bin/env python3
"""
Script to monitor MPS (Apple Silicon) GPU memory usage for PyTorch.
This script doesn't require sudo access and can be run alongside training.
"""

import torch
import time
import os
import psutil
from datetime import datetime

def format_size(bytes):
    """Format bytes to a human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024
    return f"{bytes:.2f} PB"

def get_torch_mps_memory():
    """Get MPS memory usage from PyTorch"""
    if not torch.backends.mps.is_available():
        return None
    
    try:
        stats = {}
        if hasattr(torch.mps, 'current_allocated_memory'):
            stats['allocated'] = torch.mps.current_allocated_memory()
        if hasattr(torch.mps, 'driver_allocated_memory'):
            stats['driver'] = torch.mps.driver_allocated_memory()
        if hasattr(torch.mps, 'max_memory_allocated'):
            stats['peak'] = torch.mps.max_memory_allocated()
        return stats
    except Exception as e:
        print(f"Error getting MPS memory: {e}")
        return None

def get_python_training_process():
    """Find the Python process running the transformer training"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
        try:
            if proc.info['cmdline'] and 'train_transformer.py' in ' '.join(proc.info['cmdline']):
                return {
                    'pid': proc.info['pid'],
                    'cpu_percent': proc.info['cpu_percent'],
                    'memory': proc.info['memory_info'].rss if proc.info['memory_info'] else 0,
                    'memory_human': format_size(proc.info['memory_info'].rss) if proc.info['memory_info'] else 'N/A',
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None

def main():
    print("=== PyTorch MPS Memory Monitor ===")
    print("Press Ctrl+C to exit")
    print()
    
    # Initialize process stats
    for _ in range(5):  # Sample a few times to get more accurate CPU usage
        psutil.cpu_percent(interval=0.1)
    
    try:
        while True:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Get MPS memory stats
            mps_stats = get_torch_mps_memory()
            
            # Get training process stats
            process_stats = get_python_training_process()
            
            # Clear screen
            os.system('clear')
            
            print(f"=== PyTorch MPS Memory Monitor === [{timestamp}]")
            
            # Display MPS memory stats
            if mps_stats:
                print("\nMPS Memory Usage:")
                for key, value in mps_stats.items():
                    print(f"  {key.capitalize()}: {format_size(value)}")
            else:
                print("\nMPS Memory: Not available")
            
            # Display training process stats
            if process_stats:
                print(f"\nTraining Process (PID {process_stats['pid']}):")
                print(f"  CPU: {process_stats['cpu_percent']:.1f}%")
                print(f"  Memory: {process_stats['memory_human']}")
            else:
                print("\nNo training process found")
            
            # Free up some memory
            torch.mps.empty_cache() if torch.backends.mps.is_available() else None
            
            time.sleep(3)
            
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
