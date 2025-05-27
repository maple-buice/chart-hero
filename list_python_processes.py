#!/usr/bin/env python3
"""
List Python processes that would be detected by the monitoring script.
This helps diagnose process detection issues.
"""

import os
import sys
import psutil
import argparse

def format_size(bytes):
    """Format bytes to a human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024
    return f"{bytes:.2f} PB"

def list_python_processes(detailed=False):
    """List all Python processes with relevant details"""
    python_processes = []
    
    print("\n=== Python Processes ===")
    print(f"{'PID':<8} {'CPU%':<8} {'Memory':<12} {'Command'}")
    print("-" * 80)
    
    # Define detection keywords
    ml_keywords = [
        'train', 'learning', 'tensorflow', 'pytorch', 'torch', 'model', 
        'deep', 'neural', 'ai', 'ml', 'cuda', 'gpu', 'mps', 'transformer',
        'inference', 'predict', 'keras', 'scikit', 'sklearn', 'xgboost',
        'lightgbm', 'catboost', 'huggingface', 'bert', 'gpt', 'llm',
        'benchmark', 'test_monitoring', 'monitoring_test'
    ]
    
    ml_frameworks = [
        'tensorflow', 'torch', 'pytorch', 'keras', 'sklearn', 'xgboost',
        'lightgbm', 'transformers', 'jax', 'theano', 'caffe', 'paddlepaddle',
        'mxnet', 'onnx', 'fastai', 'lightning'
    ]
    
    project_scripts = [
        'train_transformer.py', 
        'transformer_model.py',
        'model_training',
        'chart-hero',
        'train.py',
        'transformer_data.py'
    ]
    
    # Force psutil to update process info before iterating
    psutil.cpu_percent(interval=0.1)
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
        try:
            proc_name = proc.info['name'] if proc.info['name'] else "Unknown"
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            
            # Basic Python process detection
            is_python = False
            detection_reason = ""
            
            if proc_name and ('python' in proc_name.lower() or 'python3' in proc_name.lower()):
                is_python = True
                detection_reason = "Python in name"
            elif cmdline and ('python' in cmdline.lower() or 'python3' in cmdline.lower()):
                is_python = True
                detection_reason = "Python in command line"
            elif proc_name in ['Python', 'python', 'python3', 'ipython']:
                is_python = True
                detection_reason = "Python interpreter name"
            
            # Try to get executable path for additional verification
            if not is_python and (proc_name.startswith('py') or '.py' in cmdline):
                try:
                    exe = proc.exe()
                    if 'python' in exe.lower():
                        is_python = True
                        detection_reason = "Python executable"
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pass
            
            # Check for ML-related keywords
            is_ml_process = False
            ml_reason = ""
            
            if any(framework in cmdline.lower() for framework in ml_frameworks):
                is_ml_process = True
                ml_reason = "ML framework detected"
            elif any(keyword in cmdline.lower() for keyword in ml_keywords):
                is_ml_process = True
                ml_reason = "ML keyword detected"
            elif any(script in cmdline for script in project_scripts):
                is_ml_process = True
                ml_reason = "Project script detected"
            elif 'benchmark.py' in cmdline or 'test_monitoring.py' in cmdline:
                is_ml_process = True
                ml_reason = "Benchmark/test script"
            
            # Get memory info
            memory_info = proc.info.get('memory_info')
            memory_str = format_size(memory_info.rss) if memory_info else 'N/A'
            
            # Output process info
            if is_python or is_ml_process:
                cpu_percent = proc.info.get('cpu_percent', 0.0)
                cmd_display = cmdline[:60] + '...' if len(cmdline) > 60 else cmdline
                
                print(f"{proc.info['pid']:<8} {cpu_percent:<8.1f} {memory_str:<12} {cmd_display}")
                
                if detailed:
                    print(f"  Detection: {'✓' if is_python else '✗'} Python process ({detection_reason})")
                    print(f"  ML Process: {'✓' if is_ml_process else '✗'} ({ml_reason})")
                    print(f"  Full command: {cmdline}")
                    print()
                
                python_processes.append(proc.info['pid'])
        
        except (psutil.NoSuchProcess, psutil.AccessDenied, Exception) as e:
            if detailed:
                print(f"Error accessing process: {e}")
    
    print(f"\nFound {len(python_processes)} Python/ML processes")
    return python_processes

def main():
    parser = argparse.ArgumentParser(description='List Python processes for monitoring diagnosis')
    parser.add_argument('--detailed', action='store_true', help='Show detailed process information')
    args = parser.parse_args()
    
    print("=== Python Process Detection Diagnostic Tool ===")
    print("This tool shows which processes would be detected by monitor_training.py")
    
    list_python_processes(detailed=args.detailed)
    
    print("\nUsage Tips:")
    print("1. If your training process isn't listed above, use --watch-pid with monitor_training.py")
    print("2. To monitor a specific process: python monitor_training.py --watch-pid <PID>")
    print("3. For detailed process info, run this script with --detailed")

if __name__ == "__main__":
    main()
