#!/usr/bin/env python3
"""
Script to monitor system resources during training.
Run this in a separate terminal window while training is running.
"""

import psutil
import time
import os
import subprocess
import platform
import argparse
import signal
from datetime import datetime
from collections import deque
import csv
from pathlib import Path
import threading
import json
import sys

# Constants
MAX_HISTORY_POINTS = 60  # Keep a minute of history (with 1s interval)
DEFAULT_INTERVAL = 2  # Default update interval in seconds
MAX_PROCESSES_SHOWN = 10  # Maximum number of processes to show in the terminal
CONFIG_FILE_PATH = os.path.join(os.path.expanduser("~"), ".monitor_training_config.json")

# Configuration file functions
def get_config_path():
    """Get path to configuration file"""
    return CONFIG_FILE_PATH

def load_config():
    """Load configuration from file"""
    config_path = get_config_path()
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Config file {config_path} is invalid. Using defaults.")
    return {}

def save_config(config):
    """Save configuration to file"""
    config_path = get_config_path()
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False

# Global variables for storing historical data
history = {
    'timestamps': deque(maxlen=MAX_HISTORY_POINTS),
    'cpu': deque(maxlen=MAX_HISTORY_POINTS),
    'memory': deque(maxlen=MAX_HISTORY_POINTS),
    'gpu': deque(maxlen=MAX_HISTORY_POINTS),
    'disk_io': deque(maxlen=MAX_HISTORY_POINTS),
    'network': deque(maxlen=MAX_HISTORY_POINTS),
    'process_data': {}  # Dict of pid -> {'cpu': deque, 'memory': deque}
}

# Global flags
should_exit = False
log_file = None
log_writer = None

def get_gpu_usage_apple_silicon():
    """Get GPU usage on Apple Silicon Macs"""
    # First try to use PyTorch MPS info if available
    try:
        import torch
        if torch.backends.mps.is_available():
            result = {}
            
            # Force garbage collection to get accurate memory readings
            import gc
            gc.collect()
            
            # Try to clear MPS cache
            try:
                torch.mps.empty_cache()  # Clear any unused memory
            except (AttributeError, RuntimeError) as e:
                pass  # Silently handle if this fails
            
            # Create a small tensor to make sure MPS is initialized
            try:
                dummy = torch.zeros(1, device="mps")
            except RuntimeError as e:
                result["error"] = f"MPS initialization error: {str(e)}"
                return result
            
            # Try to get MPS memory stats
            memory_stats = {}
            
            # Basic MPS memory metrics
            if hasattr(torch.mps, 'current_allocated_memory'):
                allocated = torch.mps.current_allocated_memory()
                memory_stats["allocated"] = allocated
                # Get approximate total GPU memory (varies by device)
                if 'M1_MAX' in platform.machine() or 'M1_ULTRA' in platform.machine():
                    estimated_total = 32 * 1024 * 1024 * 1024  # 32GB
                elif 'M2' in platform.machine():
                    estimated_total = 24 * 1024 * 1024 * 1024  # 24GB typical for M2
                else:
                    estimated_total = 16 * 1024 * 1024 * 1024  # 16GB is typical for M1
                
                # Calculate percentage usage
                estimated_percent = min(100, (allocated / estimated_total) * 100)
                result["percent"] = estimated_percent
                result["allocated"] = allocated
                result["total_estimate"] = estimated_total
            
            # Additional memory metrics if available
            if hasattr(torch.mps, 'driver_allocated_memory'):
                driver = torch.mps.driver_allocated_memory()
                memory_stats["driver_allocated"] = driver
                result["driver_allocated"] = driver
            
            if hasattr(torch.mps, 'max_memory_allocated'):
                peak = torch.mps.max_memory_allocated()
                memory_stats["max_allocated"] = peak
                result["max_allocated"] = peak
            
            # Count active tensors on MPS device
            try:
                mps_tensors = [obj for obj in gc.get_objects() 
                              if isinstance(obj, torch.Tensor) 
                              and obj.device.type == 'mps']
                
                tensor_count = len(mps_tensors)
                memory_stats["tensor_count"] = tensor_count
                result["tensor_count"] = tensor_count
                
                # Calculate estimated memory used by Python-visible tensors
                if tensor_count > 0:
                    total_tensor_bytes = sum(t.nelement() * t.element_size() for t in mps_tensors)
                    memory_stats["tensor_memory"] = total_tensor_bytes
                    result["tensor_memory"] = total_tensor_bytes
            except Exception:
                pass  # Silently handle tensor counting errors
            
            # Add all memory stats to result
            result["memory_stats"] = memory_stats
            
            return result
            
    except (ImportError, AttributeError, RuntimeError) as e:
        # Silently handle PyTorch MPS errors
        pass
    
    # Try using powermetrics without sudo (will not work in most cases)
    try:
        # Try without sudo first, will likely fail but worth a try
        cmd = ["powermetrics", "-n", "1", "-i", "500", "--show-gpu-power", "--show-gpu-utilization"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        
        lines = result.stdout.strip().split('\n')
        gpu_percent = None
        gpu_power = None
        
        for line in lines:
            # Look for GPU utilization
            if "GPU " in line and "%" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if "%" in part:
                        try:
                            gpu_percent = float(part.strip("%"))
                        except ValueError:
                            pass
            
            # Look for GPU power usage
            if "GPU energy" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if "mW" in part:
                        try:
                            gpu_power = float(part.strip("mW"))
                        except ValueError:
                            pass
        
        return {"percent": gpu_percent, "power_mw": gpu_power}
    except (subprocess.SubprocessError, FileNotFoundError):
        # Silently fail, don't print error
        return None

def format_size(bytes):
    """Format bytes to a human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024
    return f"{bytes:.2f} PB"

def get_disk_io():
    """Get disk I/O statistics"""
    io = psutil.disk_io_counters()
    if io:
        return {
            "read_bytes": io.read_bytes,
            "write_bytes": io.write_bytes,
            "read_count": io.read_count,
            "write_count": io.write_count
        }
    return None

def get_network_io():
    """Get network I/O statistics"""
    io = psutil.net_io_counters()
    if io:
        return {
            "bytes_sent": io.bytes_sent,
            "bytes_recv": io.bytes_recv,
            "packets_sent": io.packets_sent,
            "packets_recv": io.packets_recv
        }
    return None

def get_disk_usage():
    """Get disk usage for the current directory"""
    try:
        usage = psutil.disk_usage(os.getcwd())
        return {
            "total": usage.total,
            "used": usage.used,
            "free": usage.free,
            "percent": usage.percent
        }
    except Exception as e:
        print(f"Error getting disk usage: {e}")
        return None

def get_python_processes():
    """Get all Python processes and their resource usage"""
    python_processes = []
    all_processes = []
    
    print("\nScanning processes...")
    
    # Force psutil to update process info before iterating
    psutil.cpu_percent(interval=0.1)
    
    # Get all process information
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
        try:
            # Get basic process info
            proc_name = proc.info['name'] if proc.info['name'] else "Unknown"
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            
            # Print all processes for debugging
            print(f"Process: PID={proc.info['pid']}, Name={proc_name}, Cmd={cmdline[:30]}...")
            
            # Enhanced Python process detection
            is_python = False
            if proc_name and ('python' in proc_name.lower() or 'python3' in proc_name.lower()):
                is_python = True
                print(f"Found Python process: {proc_name}")
            elif cmdline and ('python' in cmdline.lower() or 'python3' in cmdline.lower()):
                is_python = True
                print(f"Found Python process via cmdline: {cmdline[:50]}")
            # Check for common Python interpreters
            elif proc_name in ['Python', 'python', 'python3', 'ipython']:
                is_python = True
                print(f"Found Python interpreter: {proc_name}")
            
            # Try to access additional information if we're not sure
            if not is_python and (proc_name.startswith('py') or '.py' in cmdline):
                try:
                    # Try to get executable path
                    exe = proc.exe()
                    if 'python' in exe.lower():
                        is_python = True
                        print(f"Found Python process via executable: {exe}")
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pass
                
            if is_python:
                # Skip this monitoring script
                if 'monitor_training.py' in cmdline:
                    print(f"Skipping monitor script PID={proc.info['pid']}")
                    continue
                    
                # Get the process with full info to ensure accurate metrics
                try:
                    proc_detailed = psutil.Process(proc.info['pid'])
                    proc_detailed.cpu_percent(interval=0.1)  # Prime the cpu_percent calculation
                    time.sleep(0.1)  # Give it a moment to collect data
                    
                    memory_info = proc_detailed.memory_info() if hasattr(proc_detailed, 'memory_info') else None
                    cpu_percent = proc_detailed.cpu_percent(interval=0) or 0.0
                    threads_count = len(proc_detailed.threads()) if hasattr(proc_detailed, 'threads') else 0
                    
                    proc_info = {
                        'pid': proc.info['pid'],
                        'cmdline': cmdline[:50] + '...' if len(cmdline) > 50 else cmdline,
                        'cpu_percent': cpu_percent,
                        'memory': memory_info.rss if memory_info else 0,
                        'memory_human': format_size(memory_info.rss) if memory_info else 'N/A',
                        'threads': threads_count
                    }
                    
                    # Store historical data for this process
                    pid = proc.info['pid']
                    if pid not in history['process_data']:
                        history['process_data'][pid] = {
                            'cpu': deque(maxlen=MAX_HISTORY_POINTS),
                            'memory': deque(maxlen=MAX_HISTORY_POINTS),
                            'cmdline': proc_info['cmdline']
                        }
                    
                    history['process_data'][pid]['cpu'].append(proc_info['cpu_percent'])
                    history['process_data'][pid]['memory'].append(proc_info['memory'])
                    
                    python_processes.append(proc_info)
                    print(f"Added process PID={proc.info['pid']} to monitored list")
                
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    print(f"Error getting detailed process info: {e}")
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"Error accessing process: {e}")
            pass
    
    print(f"Found {len(python_processes)} Python processes to monitor")
    return python_processes

def calculate_io_rates(previous, current, interval):
    """Calculate I/O rates given previous and current measurements"""
    if not previous or not current:
        return None
    
    return {
        "read_rate": (current["read_bytes"] - previous["read_bytes"]) / interval,
        "write_rate": (current["write_bytes"] - previous["write_bytes"]) / interval,
        "read_iops": (current["read_count"] - previous["read_count"]) / interval,
        "write_iops": (current["write_count"] - previous["write_count"]) / interval
    }

def calculate_network_rates(previous, current, interval):
    """Calculate network rates given previous and current measurements"""
    if not previous or not current:
        return None
    
    return {
        "upload": (current["bytes_sent"] - previous["bytes_sent"]) / interval,
        "download": (current["bytes_recv"] - previous["bytes_recv"]) / interval,
        "packets_sent_rate": (current["packets_sent"] - previous["packets_sent"]) / interval,
        "packets_recv_rate": (current["packets_recv"] - previous["packets_recv"]) / interval
    }

def print_memory_trend(process_data):
    """Print a simple ASCII chart of memory trend"""
    if len(process_data['memory']) < 2:
        return "Not enough data"
    
    # Use last 10 points or what's available
    points = list(process_data['memory'])[-10:]
    
    # Normalize to fit in terminal
    min_val = min(points)
    max_val = max(points)
    
    if max_val == min_val:
        normalized = [5 for _ in points]
    else:
        normalized = [int(((p - min_val) / (max_val - min_val)) * 10) for p in points]
    
    # Create simple bar chart
    chart = ""
    for n in normalized:
        chart += "█" * n + " " * (10 - n)
        chart += " "
    
    trend = points[-1] - points[0]
    
    if trend > 0:
        indicator = "▲"  # Increasing
    elif trend < 0:
        indicator = "▼"  # Decreasing
    else:
        indicator = "●"  # Stable
    
    return f"{chart} {indicator} {format_size(points[-1])}"

# Constants for terminal colors
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'

def get_color_for_percent(percent):
    """Return appropriate color based on percentage value"""
    if percent is None:
        return Colors.RESET
    if percent < 30:
        return Colors.GREEN
    elif percent < 70:
        return Colors.YELLOW
    else:
        return Colors.RED

def get_trend_indicator(values, num_points=5):
    """Return a trend indicator (↑, ↓, →) based on recent values"""
    if not values or len(values) < num_points:
        return " "
    
    # Get the most recent values
    recent = list(values)[-num_points:]
    
    # Calculate trend
    if len(recent) >= 2:
        first = sum(recent[:num_points//2]) / (num_points//2)
        last = sum(recent[num_points//2:]) / (len(recent) - num_points//2)
        
        if last > first * 1.05:  # 5% increase
            return "↑"
        elif last < first * 0.95:  # 5% decrease
            return "↓"
    
    return "→"

def create_bar_chart(percent, width=20, use_color=True):
    """Create a simple ASCII bar chart"""
    if percent is None:
        return "[" + " " * width + "] N/A"
    
    # Ensure percent is between 0 and 100
    percent = max(0, min(100, percent))
    
    # Calculate filled and empty portions
    filled_width = int(width * percent / 100)
    empty_width = width - filled_width
    
    # Create bar with or without color
    if use_color:
        color = get_color_for_percent(percent)
        bar = "[" + color + "█" * filled_width + Colors.RESET + " " * empty_width + f"] {percent:.1f}%"
    else:
        bar = "[" + "█" * filled_width + " " * empty_width + f"] {percent:.1f}%"
    
    return bar

def write_to_log(data):
    """Write data to CSV log file"""
    global log_writer
    if log_writer:
        timestamp = datetime.now().isoformat()
        
        # Write system metrics
        row = {
            'timestamp': timestamp,
            'type': 'system',
            'cpu_percent': data.get('cpu_percent', ''),
            'memory_percent': data.get('memory_percent', ''),
            'memory_used': data.get('memory_used', ''),
            'gpu_percent': data.get('gpu_percent', {}).get('percent', '') if isinstance(data.get('gpu_percent'), dict) else '',
            'gpu_power': data.get('gpu_percent', {}).get('power_mw', '') if isinstance(data.get('gpu_percent'), dict) else '',
            'disk_read_rate': data.get('disk_io_rate', {}).get('read_rate', '') if data.get('disk_io_rate') else '',
            'disk_write_rate': data.get('disk_io_rate', {}).get('write_rate', '') if data.get('disk_io_rate') else '',
            'network_download': data.get('network_rate', {}).get('download', '') if data.get('network_rate') else '',
            'network_upload': data.get('network_rate', {}).get('upload', '') if data.get('network_rate') else '',
        }
        log_writer.writerow(row)
        
        # Write process metrics
        for proc in data.get('python_processes', []):
            row = {
                'timestamp': timestamp,
                'type': 'process',
                'pid': proc.get('pid', ''),
                'cpu_percent': proc.get('cpu_percent', ''),
                'memory': proc.get('memory', ''),
                'cmdline': proc.get('cmdline', '')
            }
            log_writer.writerow(row)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global should_exit
    should_exit = True
    print("\nShutting down monitoring...")

def start_logging(log_path):
    """Start logging to a CSV file"""
    global log_file, log_writer
    
    try:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = open(log_path, 'w', newline='')
        log_writer = csv.DictWriter(log_file, fieldnames=[
            'timestamp', 'type', 'pid', 'cpu_percent', 'memory_percent', 
            'memory_used', 'memory', 'gpu_percent', 'gpu_power', 
            'disk_read_rate', 'disk_write_rate', 'network_download', 
            'network_upload', 'cmdline'
        ])
        log_writer.writeheader()
        print(f"Logging to {log_path}")
        return True
    except Exception as e:
        print(f"Error setting up logging: {e}")
        return False

def main():
    # Load config
    config = load_config()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Monitor system resources during ML training')
    parser.add_argument('--interval', type=int, default=config.get('interval', DEFAULT_INTERVAL), 
                        help='Update interval in seconds')
    parser.add_argument('--log', type=str, default=config.get('log_path', None),
                        help='Log data to the specified CSV file')
    parser.add_argument('--processes', type=int, default=config.get('max_processes', MAX_PROCESSES_SHOWN), 
                        help='Maximum number of processes to show')
    parser.add_argument('--save-config', action='store_true', 
                        help='Save current settings as default configuration')
    parser.add_argument('--auto-detect', action='store_true', default=config.get('auto_detect', False),
                        help='Auto-detect training processes')
    parser.add_argument('--enable-debug', action='store_true', default=config.get('debug', False),
                        help='Enable debug output')
    parser.add_argument('--watch-pid', type=int, help='Watch a specific process ID')
    parser.add_argument('--no-color', action='store_true', default=config.get('no_color', False),
                        help='Disable colored output')
    parser.add_argument('--compact', action='store_true', default=config.get('compact', False),
                        help='Use compact display format')
    parser.add_argument('--viz-width', type=int, default=config.get('viz_width', 20),
                        help='Width of visualization bars')
    parser.add_argument('--no-process-scan', action='store_true', default=config.get('no_process_scan', False),
                        help='Disable Python process scanning (show only system metrics)')
    args = parser.parse_args()
    
    # Save configuration if requested
    if args.save_config:
        config_to_save = {
            'interval': args.interval,
            'log_path': args.log,
            'max_processes': args.processes,
            'auto_detect': args.auto_detect,
            'debug': args.enable_debug,
            'no_color': args.no_color,
            'compact': args.compact,
            'viz_width': args.viz_width
        }
        save_config(config_to_save)
        print(f"Configuration saved. Run without --save-config to use it.")
        if len(sys.argv) == 2:  # If only --save-config was provided
            return
    
    # Set up signal handling for clean exit
    signal.signal(signal.SIGINT, signal_handler)
    
    # Set up logging if requested
    if args.log:
        if not start_logging(args.log):
            return
    
    print("=== Training Resource Monitor ===")
    print("Press Ctrl+C to exit")
    print()
    
    # Initialize CPU percent
    psutil.cpu_percent(interval=None)
    
    # Initialize I/O counters
    prev_disk_io = get_disk_io()
    prev_network_io = get_network_io()
    
    try:
        # Run the process detection once with verbose output for debugging
        debug_iteration = args.enable_debug
        
        while not should_exit:
            timestamp = datetime.now().strftime("%H:%M:%S")
            history['timestamps'].append(timestamp)
            
            # System-wide metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            history['cpu'].append(cpu_percent)
            history['memory'].append(memory.percent)
            
            # GPU metrics (only works on Apple Silicon with sudo)
            gpu_percent = get_gpu_usage_apple_silicon()
            if gpu_percent and isinstance(gpu_percent, dict) and gpu_percent.get('percent') is not None:
                history['gpu'].append(gpu_percent['percent'])
            
            # Disk and Network I/O
            current_disk_io = get_disk_io()
            current_network_io = get_network_io()
            disk_usage = get_disk_usage()
            
            # Calculate rates
            disk_io_rate = calculate_io_rates(prev_disk_io, current_disk_io, args.interval)
            network_rate = calculate_network_rates(prev_network_io, current_network_io, args.interval)
            
            if disk_io_rate:
                history['disk_io'].append(disk_io_rate['write_rate'] + disk_io_rate['read_rate'])
            if network_rate:
                history['network'].append(network_rate['upload'] + network_rate['download'])
            
            # Update previous values
            prev_disk_io = current_disk_io
            prev_network_io = current_network_io
            
            # Python processes - if debugging, capture stdout
            if debug_iteration:
                import io
                import sys
                old_stdout = sys.stdout
                sys.stdout = mystdout = io.StringIO()
                
                python_processes = get_python_processes()
                
                sys.stdout = old_stdout
                debug_output = mystdout.getvalue()
                debug_iteration = False  # Only debug the first iteration
            elif not args.no_process_scan:
                # Restore normal process detection
                def get_python_processes_normal():
                    """Get all Python processes without verbose output"""
                    python_processes = []
                    
                    # Force psutil to update process info before iterating
                    psutil.cpu_percent(interval=0.1)
                    
                    # If watching a specific PID, only include that one
                    if args.watch_pid:
                        try:
                            proc = psutil.Process(args.watch_pid)
                            # Get the process with full info
                            proc_detailed = psutil.Process(args.watch_pid)
                            memory_info = proc_detailed.memory_info() if hasattr(proc_detailed, 'memory_info') else None
                            cmdline = ' '.join(proc.cmdline()) if proc.cmdline() else ''
                            
                            proc_info = {
                                'pid': args.watch_pid,
                                'cmdline': cmdline[:50] + '...' if len(cmdline) > 50 else cmdline,
                                'cpu_percent': proc_detailed.cpu_percent(interval=0) or 0.0,
                                'memory': memory_info.rss if memory_info else 0,
                                'memory_human': format_size(memory_info.rss) if memory_info else 'N/A',
                                'threads': len(proc_detailed.threads()) if hasattr(proc_detailed, 'threads') else 0
                            }
                            
                            # Store historical data
                            pid = args.watch_pid
                            if pid not in history['process_data']:
                                history['process_data'][pid] = {
                                    'cpu': deque(maxlen=MAX_HISTORY_POINTS),
                                    'memory': deque(maxlen=MAX_HISTORY_POINTS),
                                    'cmdline': proc_info['cmdline']
                                }
                            
                            history['process_data'][pid]['cpu'].append(proc_info['cpu_percent'])
                            history['process_data'][pid]['memory'].append(proc_info['memory'])
                            
                            python_processes.append(proc_info)
                            return python_processes
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            print(f"Error: Could not access process with PID {args.watch_pid}")
                    
                    # Normal process scanning
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
                        try:
                            proc_name = proc.info['name'] if proc.info['name'] else "Unknown"
                            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                            
                            # Enhanced Python process detection
                            is_python = False
                            if proc_name and ('python' in proc_name.lower() or 'python3' in proc_name.lower()):
                                is_python = True
                                if args.enable_debug:
                                    print(f"Found Python process by name: {proc_name} (PID: {proc.info['pid']})")
                            elif cmdline and ('python' in cmdline.lower() or 'python3' in cmdline.lower()):
                                is_python = True
                                if args.enable_debug:
                                    print(f"Found Python process by cmdline: {cmdline[:50]} (PID: {proc.info['pid']})")
                            # Check for common Python interpreters
                            elif proc_name in ['Python', 'python', 'python3', 'ipython']:
                                is_python = True
                                if args.enable_debug:
                                    print(f"Found Python interpreter: {proc_name} (PID: {proc.info['pid']})")
                            
                            # Auto-detect training processes
                            if args.auto_detect and not is_python:
                                # Keywords related to ML training
                                ml_keywords = [
                                    'train', 'learning', 'tensorflow', 'pytorch', 'torch', 'model', 
                                    'deep', 'neural', 'ai', 'ml', 'cuda', 'gpu', 'mps', 'transformer',
                                    'inference', 'predict', 'keras', 'scikit', 'sklearn', 'xgboost',
                                    'lightgbm', 'catboost', 'huggingface', 'bert', 'gpt', 'llm',
                                    'benchmark', 'test_monitoring', 'monitoring_test'
                                ]
                                
                                # Check for machine learning frameworks
                                ml_frameworks = [
                                    'tensorflow', 'torch', 'pytorch', 'keras', 'sklearn', 'xgboost',
                                    'lightgbm', 'transformers', 'jax', 'theano', 'caffe', 'paddlepaddle',
                                    'mxnet', 'onnx', 'fastai', 'lightning'
                                ]
                                
                                # Project-specific scripts to detect
                                project_scripts = [
                                    'train_transformer.py', 
                                    'transformer_model.py',
                                    'model_training',
                                    'chart-hero',
                                    'train.py',
                                    'transformer_data.py'
                                ]
                                
                                # First check for ML frameworks in command line
                                is_ml_framework = any(framework in cmdline.lower() for framework in ml_frameworks)
                                
                                # Then check for any ML keywords
                                is_ml_keyword = any(keyword in cmdline.lower() for keyword in ml_keywords)
                                
                                # Also check project-specific scripts
                                is_project_script = any(script in cmdline for script in project_scripts)
                                
                                # Also check common benchmark or test files
                                is_benchmark = 'benchmark.py' in cmdline or 'test_monitoring.py' in cmdline
                                
                                # If any check passes, treat as Python process for monitoring
                                if is_ml_framework or is_ml_keyword or is_benchmark or is_project_script:
                                    is_python = True
                                    if args.enable_debug:
                                        if is_benchmark:
                                            print(f"Auto-detected benchmark/test process: {cmdline[:50]} (PID: {proc.info['pid']})")
                                        elif is_project_script:
                                            print(f"Auto-detected project script: {cmdline[:50]} (PID: {proc.info['pid']})")
                                        else:
                                            print(f"Auto-detected ML process: {cmdline[:50]} (PID: {proc.info['pid']})")
                            
                            # Try to access additional information if we're not sure
                            if not is_python and (proc_name.startswith('py') or '.py' in cmdline):
                                try:
                                    # Try to get executable path
                                    exe = proc.exe()
                                    if 'python' in exe.lower():
                                        is_python = True
                                except (psutil.AccessDenied, psutil.NoSuchProcess):
                                    pass
                            
                            if is_python and 'monitor_training.py' not in cmdline:
                                # Get the process with full info to ensure accurate metrics
                                try:
                                    proc_detailed = psutil.Process(proc.info['pid'])
                                    
                                    memory_info = proc_detailed.memory_info() if hasattr(proc_detailed, 'memory_info') else None
                                    cpu_percent = proc_detailed.cpu_percent(interval=0) or 0.0
                                    threads_count = len(proc_detailed.threads()) if hasattr(proc_detailed, 'threads') else 0
                                    
                                    proc_info = {
                                        'pid': proc.info['pid'],
                                        'cmdline': cmdline[:50] + '...' if len(cmdline) > 50 else cmdline,
                                        'cpu_percent': cpu_percent,
                                        'memory': memory_info.rss if memory_info else 0,
                                        'memory_human': format_size(memory_info.rss) if memory_info else 'N/A',
                                        'threads': threads_count
                                    }
                                    
                                    # Store historical data
                                    pid = proc.info['pid']
                                    if pid not in history['process_data']:
                                        history['process_data'][pid] = {
                                            'cpu': deque(maxlen=MAX_HISTORY_POINTS),
                                            'memory': deque(maxlen=MAX_HISTORY_POINTS),
                                            'cmdline': proc_info['cmdline']
                                        }
                                    
                                    history['process_data'][pid]['cpu'].append(proc_info['cpu_percent'])
                                    history['process_data'][pid]['memory'].append(proc_info['memory'])
                                    
                                    python_processes.append(proc_info)
                                
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    return python_processes
                
                python_processes = get_python_processes_normal()
                # Restore normal process detection
                def get_python_processes_normal():
                    """Get all Python processes without verbose output"""
                    python_processes = []
                    
                    # Force psutil to update process info before iterating
                    psutil.cpu_percent(interval=0.1)
                    
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
                        try:
                            proc_name = proc.info['name'] if proc.info['name'] else "Unknown"
                            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                            
                            # Enhanced Python process detection
                            is_python = False
                            if proc_name and ('python' in proc_name.lower() or 'python3' in proc_name.lower()):
                                is_python = True
                            elif cmdline and ('python' in cmdline.lower() or 'python3' in cmdline.lower()):
                                is_python = True
                            # Check for common Python interpreters
                            elif proc_name in ['Python', 'python', 'python3', 'ipython']:
                                is_python = True
                            
                            # Try to access additional information if we're not sure
                            if not is_python and (proc_name.startswith('py') or '.py' in cmdline):
                                try:
                                    # Try to get executable path
                                    exe = proc.exe()
                                    if 'python' in exe.lower():
                                        is_python = True
                                except (psutil.AccessDenied, psutil.NoSuchProcess):
                                    pass
                            
                            if is_python and 'monitor_training.py' not in cmdline:
                                # Get the process with full info to ensure accurate metrics
                                try:
                                    proc_detailed = psutil.Process(proc.info['pid'])
                                    
                                    memory_info = proc_detailed.memory_info() if hasattr(proc_detailed, 'memory_info') else None
                                    cpu_percent = proc_detailed.cpu_percent(interval=0) or 0.0
                                    threads_count = len(proc_detailed.threads()) if hasattr(proc_detailed, 'threads') else 0
                                    
                                    proc_info = {
                                        'pid': proc.info['pid'],
                                        'cmdline': cmdline[:50] + '...' if len(cmdline) > 50 else cmdline,
                                        'cpu_percent': cpu_percent,
                                        'memory': memory_info.rss if memory_info else 0,
                                        'memory_human': format_size(memory_info.rss) if memory_info else 'N/A',
                                        'threads': threads_count
                                    }
                                    
                                    # Store historical data
                                    pid = proc.info['pid']
                                    if pid not in history['process_data']:
                                        history['process_data'][pid] = {
                                            'cpu': deque(maxlen=MAX_HISTORY_POINTS),
                                            'memory': deque(maxlen=MAX_HISTORY_POINTS),
                                            'cmdline': proc_info['cmdline']
                                        }
                                    
                                    history['process_data'][pid]['cpu'].append(proc_info['cpu_percent'])
                                    history['process_data'][pid]['memory'].append(proc_info['memory'])
                                    
                                    python_processes.append(proc_info)
                                
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    return python_processes
                
                python_processes = get_python_processes_normal()
            
            # Prepare data for logging
            log_data = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used': memory.used,
                'gpu_percent': gpu_percent,
                'disk_io_rate': disk_io_rate,
                'network_rate': network_rate,
                'python_processes': python_processes
            }
            
            # Write to log if enabled
            if args.log:
                write_to_log(log_data)
            
            # Update the display
            os.system('clear' if os.name == 'posix' else 'cls')
            print(f"=== System Resource Monitor === (Press Ctrl+C to exit) ===")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Interval: {args.interval}s")
            
            # System metrics
            print("\n=== System Metrics ===")
            
            # Use our new visualization functions
            use_color = not args.no_color
            bar_width = args.viz_width
            
            # CPU Usage with bar chart
            cpu_color = get_color_for_percent(cpu_percent)
            cpu_trend = get_trend_indicator(history['cpu'])
            cpu_bar = create_bar_chart(cpu_percent, width=bar_width, use_color=use_color)
            
            if not args.compact:
                print(f"CPU Usage: {cpu_bar} {cpu_trend}")
            else:
                print(f"CPU: {cpu_color if use_color else ''}{cpu_percent:.1f}%{Colors.RESET if use_color else ''} {cpu_trend}")
            
            # Memory Usage with bar chart
            mem_color = get_color_for_percent(memory.percent)
            mem_trend = get_trend_indicator(history['memory'])
            mem_bar = create_bar_chart(memory.percent, width=bar_width, use_color=use_color)
            
            if not args.compact:
                print(f"Memory Usage: {mem_bar} {mem_trend} ({format_size(memory.used)} of {format_size(memory.total)})")
            else:
                print(f"Mem: {mem_color if use_color else ''}{memory.percent:.1f}%{Colors.RESET if use_color else ''} {mem_trend} ({format_size(memory.used)})")
            
            # GPU Usage with bar chart (if available)
            if gpu_percent is not None and isinstance(gpu_percent, dict) and 'percent' in gpu_percent:
                gpu_pct = gpu_percent['percent']
                gpu_color = get_color_for_percent(gpu_pct)
                gpu_trend = get_trend_indicator(history['gpu'])
                gpu_bar = create_bar_chart(gpu_pct, width=bar_width, use_color=use_color)
                
                if not args.compact:
                    print(f"GPU Usage: {gpu_bar} {gpu_trend}")
                else:
                    print(f"GPU: {gpu_color if use_color else ''}{gpu_pct:.1f}%{Colors.RESET if use_color else ''} {gpu_trend}")
                
                # Additional GPU info
                if not args.compact and 'power_mw' in gpu_percent and gpu_percent['power_mw'] is not None:
                    print(f"GPU Power: {gpu_percent['power_mw']:.2f} mW")
            else:
                print("GPU Usage: Not available")
            
            # Disk I/O rates
            if disk_io_rate:
                read_mb = disk_io_rate['read_rate'] / (1024 * 1024)
                write_mb = disk_io_rate['write_rate'] / (1024 * 1024)
                
                if not args.compact:
                    print(f"Disk I/O: Read {read_mb:.2f} MB/s, Write {write_mb:.2f} MB/s")
                else:
                    print(f"Disk: R:{read_mb:.1f}MB/s W:{write_mb:.1f}MB/s")
            
            # Network I/O rates
            if network_rate:
                down_mb = network_rate['download'] / (1024 * 1024)
                up_mb = network_rate['upload'] / (1024 * 1024)
                
                # Add network visualization with bar charts
                if not args.compact:
                    # Calculate percentage for visualization (assume 100Mbps = 12.5MB/s is 100%)
                    down_percent = min(100, (down_mb / 12.5) * 100) 
                    up_percent = min(100, (up_mb / 12.5) * 100)
                    
                    down_color = get_color_for_percent(down_percent)
                    up_color = get_color_for_percent(up_percent)
                    
                    down_bar = create_bar_chart(down_percent, width=bar_width//2, use_color=use_color)
                    up_bar = create_bar_chart(up_percent, width=bar_width//2, use_color=use_color)
                    
                    down_trend = get_trend_indicator(history['network'])
                    
                    print(f"Network: Down {down_bar} {down_mb:.2f} MB/s, Up {up_bar} {up_mb:.2f} MB/s {down_trend}")
                else:
                    print(f"Net: D:{down_mb:.1f}MB/s U:{up_mb:.1f}MB/s")
            
            # Get PyTorch MPS memory info if available
            try:
                import torch
                if torch.backends.mps.is_available():
                    print("\nPyTorch MPS Memory:")
                    
                    # Get detailed GPU information directly from our enhanced function
                    gpu_details = get_gpu_usage_apple_silicon()
                    
                    if gpu_details and isinstance(gpu_details, dict):
                        stats = []
                        
                        # Check if we have memory stats
                        if "allocated" in gpu_details:
                            # Memory usage with bar chart if percent is available
                            if "percent" in gpu_details:
                                gpu_pct = gpu_details["percent"]
                                gpu_bar = create_bar_chart(gpu_pct, width=args.viz_width, use_color=not args.no_color)
                                print(f"MPS Memory: {gpu_bar}")
                            
                            # Display allocated memory
                            allocated = gpu_details["allocated"]
                            stats.append(f"Allocated: {format_size(allocated)}")
                            
                            # Display total estimated memory if available
                            if "total_estimate" in gpu_details:
                                total = gpu_details["total_estimate"]
                                stats.append(f"Est. Total: {format_size(total)}")
                        
                        # Add driver allocated memory if available
                        if "driver_allocated" in gpu_details:
                            driver = gpu_details["driver_allocated"]
                            stats.append(f"Driver: {format_size(driver)}")
                        
                        # Add max allocated memory if available
                        if "max_allocated" in gpu_details:
                            peak = gpu_details["max_allocated"]
                            stats.append(f"Peak: {format_size(peak)}")
                        
                        # Add tensor information if available
                        if "tensor_count" in gpu_details:
                            count = gpu_details["tensor_count"]
                            stats.append(f"Tensors: {count}")
                            
                            if "tensor_memory" in gpu_details:
                                tensor_mem = gpu_details["tensor_memory"]
                                stats.append(f"Tensor Mem: {format_size(tensor_mem)}")
                        
                        # Display error if any
                        if "error" in gpu_details:
                            stats.append(f"Error: {gpu_details['error']}")
                        
                        # Display the collected stats
                        if stats:
                            print(", ".join(stats))
                        else:
                            print("No MPS memory statistics available")
                            
                        # If debugging is enabled, show more detailed MPS info
                        if args.enable_debug:
                            print("\nMPS Debug Information:")
                            # List all available functions
                            available_functions = [f for f in dir(torch.mps) if not f.startswith('_')]
                            print(f"Available MPS functions: {', '.join(available_functions)}")
                            
                            # List active tensors
                            try:
                                import gc
                                mps_tensors = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor) and obj.device.type == 'mps']
                                if mps_tensors:
                                    print(f"Active MPS Tensors: {len(mps_tensors)}")
                                    for i, t in enumerate(mps_tensors[:5]):  # Show up to 5 tensors
                                        print(f"  Tensor {i+1}: shape={t.shape}, dtype={t.dtype}, size={format_size(t.nelement() * t.element_size())}")
                                    if len(mps_tensors) > 5:
                                        print(f"  ... and {len(mps_tensors)-5} more tensors")
                            except Exception as e:
                                print(f"Error listing MPS tensors: {str(e)}")
                    else:
                        print("MPS memory information not available")
                        
            except (ImportError, AttributeError) as e:
                print(f"\nPyTorch MPS not available: {str(e)}")
                pass
            
            # Display Python processes (sorted by CPU usage)
            print("\nPython Processes (with memory trend):")
            if python_processes:
                if not args.compact:
                    print(f"{'PID':<7} {'CPU%':<7} {'Memory':<10} {'Memory Trend (10s)':<30} {'Threads':<8} {'Command'}")
                    print("-" * 100)
                    
                    for proc in sorted(python_processes, key=lambda x: x['cpu_percent'], reverse=True)[:args.processes]:
                        pid = proc['pid']
                        if pid in history['process_data'] and len(history['process_data'][pid]['memory']) > 0:
                            trend = print_memory_trend(history['process_data'][pid])
                            
                            # Add color based on CPU usage
                            cpu_color = get_color_for_percent(proc['cpu_percent'])
                            cpu_str = f"{cpu_color if not args.no_color else ''}{proc['cpu_percent']:<7.1f}{Colors.RESET if not args.no_color else ''}"
                            
                            print(f"{pid:<7} {cpu_str} {proc['memory_human']:<10} {trend:<30} {proc['threads']:<8} {proc['cmdline']}")
                else:
                    # Compact display for processes
                    for i, proc in enumerate(sorted(python_processes, key=lambda x: x['cpu_percent'], reverse=True)[:args.processes]):
                        pid = proc['pid']
                        if pid in history['process_data'] and len(history['process_data'][pid]['memory']) > 0:
                            cpu_color = get_color_for_percent(proc['cpu_percent'])
                            cpu_str = f"{cpu_color if not args.no_color else ''}{proc['cpu_percent']:.1f}%{Colors.RESET if not args.no_color else ''}"
                            
                            # Get trend indicators
                            cpu_trend = get_trend_indicator(history['process_data'][pid]['cpu'])
                            mem_trend = get_trend_indicator(history['process_data'][pid]['memory'])
                            
                            cmd = proc['cmdline'].split()[-1] if proc['cmdline'] else "unknown"
                            print(f"P{i+1}: {pid} | CPU:{cpu_str}{cpu_trend} | Mem:{proc['memory_human']}{mem_trend} | {cmd}")
            else:
                print("No Python processes found.")
                
                # If we're debugging and no processes found, show some debug info
                if args.enable_debug and not debug_iteration:
                    print("\nProcess Detection Debug Information:")
                    debug_info = []
                    debug_info.append("No Python processes detected")
                    print("\n".join(debug_info))
            
            # Display logging status
            if args.log:
                print(f"\nLogging to: {args.log}")
                
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        pass
    finally:
        if log_file:
            log_file.close()
            print(f"\nLog file saved to {args.log}")
        print("\nExiting...")

if __name__ == "__main__":
    main()
