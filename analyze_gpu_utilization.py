#!/usr/bin/env python3
"""
Enhanced analysis script for ML training resource monitoring data
Specifically focuses on GPU (MPS) utilization for Apple Silicon Macs
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from datetime import datetime
import json
import seaborn as sns
from pathlib import Path

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

def analyze_gpu_metrics(df, output_dir):
    """Analyze GPU metrics with a focus on MPS utilization"""
    # Filter for system records that have GPU metrics
    gpu_df = df[df['type'] == 'system'].copy()
    
    if gpu_df.empty:
        print("No system metrics found in log file.")
        return
    
    # Check if GPU metrics exist
    if 'gpu_memory_used' not in gpu_df.columns:
        print("No GPU memory metrics found in log file.")
        return
    
    # Set timestamp as index and sort
    gpu_df.set_index('timestamp', inplace=True)
    gpu_df = gpu_df.sort_index()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate metrics
    if 'gpu_memory_total' in gpu_df.columns and gpu_df['gpu_memory_total'].max() > 0:
        # Calculate memory utilization as percentage
        gpu_df['gpu_memory_percent'] = 100 * gpu_df['gpu_memory_used'] / gpu_df['gpu_memory_total']
    
    # Plot GPU memory usage
    plt.figure(figsize=(12, 6))
    plt.plot(gpu_df.index, gpu_df['gpu_memory_used'] / (1024*1024), 'g-', linewidth=2)
    plt.title('GPU Memory Usage Over Time')
    plt.ylabel('Memory Usage (MB)')
    plt.xlabel('Time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gpu_memory_usage.png'))
    
    # If we have utilization data, plot it
    if 'gpu_utilization' in gpu_df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(gpu_df.index, gpu_df['gpu_utilization'], 'r-', linewidth=2)
        plt.title('GPU Utilization Over Time')
        plt.ylabel('Utilization (%)')
        plt.xlabel('Time')
        plt.ylim(0, 100)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gpu_utilization.png'))
    
    # Create a summary of GPU metrics
    summary = {
        'max_gpu_memory_mb': gpu_df['gpu_memory_used'].max() / (1024*1024),
        'avg_gpu_memory_mb': gpu_df['gpu_memory_used'].mean() / (1024*1024),
        'min_gpu_memory_mb': gpu_df['gpu_memory_used'].min() / (1024*1024),
    }
    
    if 'gpu_utilization' in gpu_df.columns:
        summary['max_gpu_utilization'] = gpu_df['gpu_utilization'].max()
        summary['avg_gpu_utilization'] = gpu_df['gpu_utilization'].mean()
        summary['min_gpu_utilization'] = gpu_df['gpu_utilization'].min()
    
    if 'gpu_memory_percent' in gpu_df.columns:
        summary['max_gpu_memory_percent'] = gpu_df['gpu_memory_percent'].max()
        summary['avg_gpu_memory_percent'] = gpu_df['gpu_memory_percent'].mean()
    
    # Save summary to file
    with open(os.path.join(output_dir, 'gpu_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nGPU Usage Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.2f}")
    
    return summary

def analyze_process_metrics(df, output_dir):
    """Analyze process-specific metrics"""
    # Filter for process records
    process_df = df[df['type'] == 'process'].copy()
    
    if process_df.empty:
        print("No process metrics found in log file.")
        return
    
    # Set timestamp as index
    process_df.set_index('timestamp', inplace=True)
    
    # Plot process CPU and memory usage
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # CPU Usage
    axes[0].plot(process_df.index, process_df['cpu_percent'], 'b-', linewidth=2)
    axes[0].set_title('Process CPU Usage Over Time')
    axes[0].set_ylabel('CPU Usage (%)')
    axes[0].grid(True)
    
    # Memory Usage
    axes[1].plot(process_df.index, process_df['memory_percent'], 'r-', linewidth=2)
    axes[1].set_title('Process Memory Usage Over Time')
    axes[1].set_ylabel('Memory Usage (%)')
    axes[1].set_xlabel('Time')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'process_usage.png'))
    
    # Calculate summary statistics
    summary = {
        'max_process_cpu': process_df['cpu_percent'].max(),
        'avg_process_cpu': process_df['cpu_percent'].mean(),
        'max_process_memory': process_df['memory_percent'].max(),
        'avg_process_memory': process_df['memory_percent'].mean(),
    }
    
    # Save summary to file
    with open(os.path.join(output_dir, 'process_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nProcess Usage Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.2f}")
    
    return summary

def generate_report(system_summary, process_summary, gpu_summary, output_dir):
    """Generate a comprehensive HTML report"""
    report_path = os.path.join(output_dir, 'resource_utilization_report.html')
    
    # Combine all summaries
    all_metrics = {}
    if system_summary:
        all_metrics.update(system_summary)
    if process_summary:
        all_metrics.update(process_summary)
    if gpu_summary:
        all_metrics.update(gpu_summary)
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Training Resource Utilization Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; }}
            .metric {{ margin-bottom: 5px; }}
            .metric-name {{ font-weight: bold; }}
            .section {{ margin-bottom: 30px; }}
            .image-container {{ margin: 20px 0; }}
            img {{ max-width: 100%; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>ML Training Resource Utilization Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h2>Key Metrics Summary</h2>
    """
    
    # Add metrics
    for key, value in all_metrics.items():
        html_content += f"""
            <div class="metric">
                <span class="metric-name">{key}:</span> {value:.2f}
            </div>
        """
    
    # Add images
    html_content += """
        </div>
        
        <div class="section">
            <h2>GPU Utilization</h2>
            <div class="image-container">
                <img src="gpu_memory_usage.png" alt="GPU Memory Usage">
            </div>
    """
    
    # Add GPU utilization if it exists
    if os.path.exists(os.path.join(output_dir, 'gpu_utilization.png')):
        html_content += """
            <div class="image-container">
                <img src="gpu_utilization.png" alt="GPU Utilization">
            </div>
        """
    
    # Add process metrics
    html_content += """
        </div>
        
        <div class="section">
            <h2>Process Resource Usage</h2>
            <div class="image-container">
                <img src="process_usage.png" alt="Process Resource Usage">
            </div>
        </div>
        
        <div class="section">
            <h2>System Resource Usage</h2>
            <div class="image-container">
                <img src="system_metrics.png" alt="System Resource Usage">
            </div>
        </div>
        
        <h2>Recommendations</h2>
        <ul>
            <li>If GPU utilization is consistently below 50%, consider increasing batch size or model complexity</li>
            <li>If GPU memory usage is near maximum, consider reducing batch size or using gradient accumulation</li>
            <li>If CPU usage is very high but GPU utilization is low, data loading might be a bottleneck</li>
        </ul>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"Comprehensive report saved to {report_path}")

def analyze_system_metrics(df, output_dir):
    """Analyze and plot system-wide metrics"""
    # Filter for system records
    system_df = df[df['type'] == 'system'].copy()
    
    if system_df.empty:
        print("No system metrics found in log file.")
        return
    
    # Set timestamp as index
    system_df.set_index('timestamp', inplace=True)
    
    # Plot CPU, Memory, and Swap usage
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
    
    # Swap Usage if available
    if 'swap_percent' in system_df.columns:
        axes[2].plot(system_df.index, system_df['swap_percent'], 'g-', linewidth=2)
        axes[2].set_title('Swap Usage Over Time')
        axes[2].set_ylabel('Swap Usage (%)')
    else:
        axes[2].text(0.5, 0.5, 'No Swap Usage Data Available', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=axes[2].transAxes)
    
    axes[2].set_xlabel('Time')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'system_metrics.png'))
    
    # Calculate summary statistics
    summary = {
        'max_system_cpu': system_df['cpu_percent'].max(),
        'avg_system_cpu': system_df['cpu_percent'].mean(),
        'max_system_memory': system_df['memory_percent'].max(),
        'avg_system_memory': system_df['memory_percent'].mean(),
    }
    
    if 'swap_percent' in system_df.columns:
        summary['max_swap'] = system_df['swap_percent'].max()
        summary['avg_swap'] = system_df['swap_percent'].mean()
    
    # Save summary to file
    with open(os.path.join(output_dir, 'system_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nSystem Usage Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.2f}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='Analyze resource monitoring logs')
    parser.add_argument('--log', required=True, help='Path to the monitoring log file')
    parser.add_argument('--output', default='analysis_output', help='Output directory for analysis')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading log file: {args.log}")
    df = load_log(args.log)
    
    if df is None:
        return
    
    print(f"Analyzing {len(df)} log entries...")
    
    # Run analyses
    system_summary = analyze_system_metrics(df, output_dir)
    process_summary = analyze_process_metrics(df, output_dir)
    gpu_summary = analyze_gpu_metrics(df, output_dir)
    
    # Generate comprehensive report
    generate_report(system_summary, process_summary, gpu_summary, output_dir)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}/")

if __name__ == "__main__":
    main()
