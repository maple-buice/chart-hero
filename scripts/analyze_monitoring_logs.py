#!/usr/bin/env python3
"""
Analyze training resource usage logs collected by monitor_training.py
This script creates visualizations and provides insights from resource monitoring data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime

def load_log(log_path):
    """Load the CSV log file into pandas DataFrame"""
    if not os.path.exists(log_path):
        print(f"Error: Log file {log_path} not found.")
        return None
    
    try:
        df = pd.read_csv(log_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"Error loading log file: {e}")
        return None

def analyze_system_metrics(df):
    """Analyze and plot system-wide metrics"""
    # Filter for system records
    system_df = df[df['type'] == 'system'].copy()
    
    if system_df.empty:
        print("No system metrics found in log file.")
        return
    
    # Set timestamp as index
    system_df.set_index('timestamp', inplace=True)
    
    # Plot CPU, Memory, and GPU usage
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # CPU Usage
    axes[0].plot(system_df.index, system_df['cpu_percent'], 'b-', linewidth=2)
    axes[0].set_title('CPU Usage Over Time')
    axes[0].set_ylabel('CPU Usage (%)')
    axes[0].grid(True)
    
    # Memory Usage
    axes[1].plot(system_df.index, system_df['memory_percent'], 'r-', linewidth=2)
    axes[1].set_title('Memory Usage Over Time')
    axes[1].set_ylabel('Memory Usage (%)')
    axes[1].grid(True)
    
    # GPU Usage (if available)
    if 'gpu_percent' in system_df.columns and not system_df['gpu_percent'].isna().all():
        axes[2].plot(system_df.index, system_df['gpu_percent'], 'g-', linewidth=2)
        axes[2].set_title('GPU Usage Over Time')
        axes[2].set_ylabel('GPU Usage (%)')
    else:
        axes[2].text(0.5, 0.5, 'No GPU data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[2].transAxes)
        axes[2].set_title('GPU Usage (No Data)')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('system_usage.png')
    print(f"System metrics chart saved as system_usage.png")
    
    # Plot I/O metrics
    if ('disk_read_rate' in system_df.columns and 'disk_write_rate' in system_df.columns and
        'network_download' in system_df.columns and 'network_upload' in system_df.columns):
        
        # Convert to MB/s for easier visualization
        for col in ['disk_read_rate', 'disk_write_rate', 'network_download', 'network_upload']:
            system_df[f'{col}_mb'] = system_df[col] / (1024 * 1024)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Disk I/O
        axes[0].plot(system_df.index, system_df['disk_read_rate_mb'], 'b-', label='Read')
        axes[0].plot(system_df.index, system_df['disk_write_rate_mb'], 'r-', label='Write')
        axes[0].set_title('Disk I/O Over Time')
        axes[0].set_ylabel('MB/s')
        axes[0].legend()
        axes[0].grid(True)
        
        # Network I/O
        axes[1].plot(system_df.index, system_df['network_download_mb'], 'g-', label='Download')
        axes[1].plot(system_df.index, system_df['network_upload_mb'], 'm-', label='Upload')
        axes[1].set_title('Network I/O Over Time')
        axes[1].set_ylabel('MB/s')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('io_usage.png')
        print(f"I/O metrics chart saved as io_usage.png")

def analyze_process_metrics(df):
    """Analyze and plot process-specific metrics"""
    # Filter for process records
    process_df = df[df['type'] == 'process'].copy()
    
    if process_df.empty:
        print("No process metrics found in log file.")
        return
    
    # Get unique PIDs
    pids = process_df['pid'].unique()
    
    # Find the most resource-intensive processes
    top_processes = []
    for pid in pids:
        pid_df = process_df[process_df['pid'] == pid]
        avg_cpu = pid_df['cpu_percent'].mean()
        max_cpu = pid_df['cpu_percent'].max()
        avg_memory = pid_df['memory'].mean()
        max_memory = pid_df['memory'].max()
        cmdline = pid_df['cmdline'].iloc[0] if not pid_df['cmdline'].isna().all() else 'Unknown'
        
        top_processes.append({
            'pid': pid,
            'avg_cpu': avg_cpu,
            'max_cpu': max_cpu,
            'avg_memory': avg_memory,
            'max_memory': max_memory,
            'cmdline': cmdline
        })
    
    # Sort by average CPU usage
    top_processes.sort(key=lambda x: x['avg_cpu'], reverse=True)
    
    # Plot top 3 processes by CPU usage
    if len(top_processes) > 0:
        top_n = min(3, len(top_processes))
        top_pids = [p['pid'] for p in top_processes[:top_n]]
        
        fig, axes = plt.subplots(top_n, 2, figsize=(12, 4 * top_n), sharex=True)
        
        for i, pid in enumerate(top_pids):
            pid_df = process_df[process_df['pid'] == pid]
            pid_df.set_index('timestamp', inplace=True)
            
            # If only one process, axes needs special handling
            if top_n == 1:
                ax_cpu = axes[0]
                ax_mem = axes[1]
            else:
                ax_cpu = axes[i, 0]
                ax_mem = axes[i, 1]
            
            # Process name for the title
            proc_name = pid_df['cmdline'].iloc[0]
            if len(proc_name) > 30:
                proc_name = proc_name[:27] + '...'
            
            # CPU Usage
            ax_cpu.plot(pid_df.index, pid_df['cpu_percent'], 'b-')
            ax_cpu.set_title(f'CPU Usage - PID {pid} ({proc_name})')
            ax_cpu.set_ylabel('CPU Usage (%)')
            ax_cpu.grid(True)
            
            # Memory Usage
            # Convert bytes to MB for better visibility
            memory_mb = pid_df['memory'] / (1024 * 1024)
            ax_mem.plot(pid_df.index, memory_mb, 'r-')
            ax_mem.set_title(f'Memory Usage - PID {pid} ({proc_name})')
            ax_mem.set_ylabel('Memory Usage (MB)')
            ax_mem.grid(True)
        
        plt.tight_layout()
        plt.savefig('process_usage.png')
        print(f"Process metrics chart saved as process_usage.png")
    
    # Print summary of top processes
    print("\nTop Processes by Average CPU Usage:")
    print(f"{'PID':<7} {'Avg CPU%':<10} {'Max CPU%':<10} {'Avg Memory':<15} {'Max Memory':<15} {'Command'}")
    print("-" * 80)
    
    for proc in top_processes[:5]:  # Show top 5
        avg_mem_human = format_size(proc['avg_memory'])
        max_mem_human = format_size(proc['max_memory'])
        cmdline = proc['cmdline']
        if len(cmdline) > 30:
            cmdline = cmdline[:27] + '...'
        
        print(f"{proc['pid']:<7} {proc['avg_cpu']:<10.1f} {proc['max_cpu']:<10.1f} {avg_mem_human:<15} {max_mem_human:<15} {cmdline}")

def analyze_gpu_efficiency(df):
    """Analyze GPU efficiency metrics"""
    # Filter for system and process records
    system_df = df[df['type'] == 'system'].copy()
    process_df = df[df['type'] == 'process'].copy()
    
    if system_df.empty or 'gpu_percent' not in system_df.columns or system_df['gpu_percent'].isna().all():
        print("No GPU data available for efficiency analysis")
        return
    
    # Calculate basic GPU metrics
    avg_gpu = system_df['gpu_percent'].mean()
    max_gpu = system_df['gpu_percent'].max()
    min_gpu = system_df['gpu_percent'].min()
    std_gpu = system_df['gpu_percent'].std()
    
    # Calculate GPU utilization stability (lower std dev relative to mean indicates more stable usage)
    stability_score = 100 - (std_gpu / max(0.1, avg_gpu) * 100)
    stability_score = max(0, min(100, stability_score))  # Clamp between 0-100
    
    print("\nGPU Efficiency Analysis:")
    print(f"{'Average GPU Usage':<25} {avg_gpu:.2f}%")
    print(f"{'Peak GPU Usage':<25} {max_gpu:.2f}%")
    print(f"{'Min GPU Usage':<25} {min_gpu:.2f}%")
    print(f"{'GPU Usage Std Dev':<25} {std_gpu:.2f}%")
    print(f"{'GPU Stability Score':<25} {stability_score:.2f}/100")
    
    # Calculate efficiency metrics if process data is available
    if not process_df.empty and 'cpu_percent' in process_df.columns:
        avg_cpu_per_process = process_df['cpu_percent'].mean()
        
        # GPU/CPU Efficiency Ratio - higher is better
        # This measures how effectively we're using the GPU relative to CPU
        gpu_cpu_ratio = avg_gpu / max(0.1, avg_cpu_per_process)
        
        print(f"\nCPU/GPU Efficiency:")
        print(f"{'Average CPU Usage':<25} {avg_cpu_per_process:.2f}%")
        print(f"{'GPU/CPU Ratio':<25} {gpu_cpu_ratio:.2f}")
        
        # Interpret the results
        print("\nEfficiency Interpretation:")
        
        if avg_gpu < 30:
            print("- GPU is significantly UNDERUTILIZED (avg < 30%)")
            print("  Recommendation: Increase batch size or model complexity")
        elif avg_gpu < 70:
            print("- GPU is MODERATELY UTILIZED (30-70%)")
            print("  Recommendation: Current parameters are reasonable, but could be optimized")
        else:
            print("- GPU is WELL UTILIZED (>70%)")
            
        if gpu_cpu_ratio < 0.5:
            print("- CPU-bound workload: CPU usage is much higher than GPU")
            print("  Recommendation: Optimize CPU operations, especially data loading")
        elif gpu_cpu_ratio > 2.0:
            print("- GPU-bound workload: GPU is the primary bottleneck")
            print("  Recommendation: Consider GPU memory optimization techniques")
        else:
            print("- Balanced CPU/GPU utilization")
    
    # Plot GPU utilization over time
    plt.figure(figsize=(12, 6))
    system_df.set_index('timestamp', inplace=True)
    plt.plot(system_df.index, system_df['gpu_percent'], 'g-', linewidth=2)
    plt.title('GPU Utilization Over Time')
    plt.ylabel('GPU Usage (%)')
    plt.xlabel('Time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('gpu_efficiency.png')
    print(f"\nGPU efficiency chart saved as gpu_efficiency.png")

def format_size(bytes):
    """Format bytes to a human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024
    return f"{bytes:.2f} PB"

def main():
    parser = argparse.ArgumentParser(description='Analyze training resource monitoring logs')
    parser.add_argument('log_file', help='Path to the CSV log file generated by monitor_training.py')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots (text summary only)')
    parser.add_argument('--output', type=str, default=None, help='Output directory for analysis results')
    args = parser.parse_args()
    
    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        # Change working directory to output directory for saving plots
        os.chdir(args.output)
    
    # Load log data
    df = load_log(args.log_file)
    if df is None:
        return
    
    # Print basic statistics
    duration = df['timestamp'].max() - df['timestamp'].min()
    print(f"=== Resource Usage Analysis ===")
    print(f"Log file: {args.log_file}")
    print(f"Monitoring duration: {duration}")
    print(f"Data points: {len(df)}")
    
    # Generate visualizations if not disabled
    if not args.no_plots:
        try:
            analyze_system_metrics(df)
            analyze_process_metrics(df)
            analyze_gpu_efficiency(df)
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    # Text summary of system metrics
    system_df = df[df['type'] == 'system']
    if not system_df.empty:
        print("\nSystem Metrics Summary:")
        metrics = [
            ('CPU Usage (%)', 'cpu_percent'),
            ('Memory Usage (%)', 'memory_percent'),
            ('GPU Usage (%)', 'gpu_percent'),
        ]
        
        for label, col in metrics:
            if col in system_df.columns and not system_df[col].isna().all():
                avg = system_df[col].mean()
                peak = system_df[col].max()
                print(f"{label:<20} Avg: {avg:.2f}, Peak: {peak:.2f}")

if __name__ == "__main__":
    main()
