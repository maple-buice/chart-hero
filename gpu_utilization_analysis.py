#!/usr/bin/env python3
"""
GPU Utilization Analysis Script for Chart-Hero.
This script runs multiple training scenarios with different configurations
and analyzes GPU utilization across scenarios.
"""

import argparse
import subprocess
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import json

# Define the scenarios to test
DEFAULT_SCENARIOS = [
    {"name": "small_batch", "batch_size": 4, "hidden_size": 256, "time_mins": 5},
    {"name": "medium_batch", "batch_size": 8, "hidden_size": 384, "time_mins": 5},
    {"name": "large_batch", "batch_size": 16, "hidden_size": 384, "time_mins": 5},
    {"name": "small_model", "batch_size": 8, "hidden_size": 256, "time_mins": 5},
    {"name": "medium_model", "batch_size": 8, "hidden_size": 384, "time_mins": 5},
    {"name": "large_model", "batch_size": 8, "hidden_size": 512, "time_mins": 5},
]

def run_scenario(scenario, args):
    """Run a single training scenario with the given parameters"""
    print(f"\n{'='*80}")
    print(f"RUNNING SCENARIO: {scenario['name']}")
    print(f"Batch Size: {scenario['batch_size']}, Hidden Size: {scenario['hidden_size']}")
    print(f"Duration: {scenario['time_mins']} minutes")
    print(f"{'='*80}\n")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_dir = os.path.join(args.output_dir, f"{scenario['name']}_{timestamp}")
    os.makedirs(scenario_dir, exist_ok=True)
    
    # Save scenario parameters
    scenario_params = scenario.copy()
    scenario_params['timestamp'] = timestamp
    with open(os.path.join(scenario_dir, 'scenario_params.json'), 'w') as f:
        json.dump(scenario_params, f, indent=2)
    
    # Run the timed training script with parameters
    cmd = [
        "./run_timed_training.sh",
        str(scenario['time_mins']),
        str(scenario['batch_size']),
        str(scenario['hidden_size']),
        scenario_dir
    ]
    
    try:
        # Run the process and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
        
        # Wait for completion
        process.wait()
        
        if process.returncode != 0:
            print(f"Error running scenario {scenario['name']}")
            stderr = process.stderr.read()
            print(f"Error output: {stderr}")
            return None
        
        # Find the monitoring log
        log_files = glob.glob(os.path.join(scenario_dir, "*.csv"))
        if not log_files:
            print(f"No log files found for scenario {scenario['name']}")
            return None
        
        return scenario_dir
    
    except Exception as e:
        print(f"Error executing scenario {scenario['name']}: {e}")
        return None

def analyze_results(scenarios_results, args):
    """Analyze results across all scenarios"""
    print("\n\n")
    print(f"{'='*40}")
    print("ANALYSIS RESULTS")
    print(f"{'='*40}")
    
    # Prepare data for analysis
    scenario_data = []
    
    for scenario_dir in scenarios_results:
        if not scenario_dir:
            continue
        
        # Load scenario parameters
        with open(os.path.join(scenario_dir, 'scenario_params.json'), 'r') as f:
            params = json.load(f)
        
        # Find the monitor log file
        log_files = glob.glob(os.path.join(scenario_dir, "*.csv"))
        if not log_files:
            print(f"No log files found in {scenario_dir}")
            continue
        
        monitor_log = log_files[0]
        
        # Load monitoring data
        try:
            df = pd.read_csv(monitor_log)
            
            # Filter for system records to get GPU data
            system_df = df[df['type'] == 'system'].copy()
            
            if 'gpu_percent' not in system_df.columns or system_df['gpu_percent'].isna().all():
                print(f"No GPU data found in {monitor_log}")
                continue
            
            # Calculate metrics
            avg_gpu = system_df['gpu_percent'].mean()
            max_gpu = system_df['gpu_percent'].max()
            std_gpu = system_df['gpu_percent'].std()
            
            # Process records if available
            process_df = df[df['type'] == 'process'].copy()
            avg_cpu_per_process = 0
            avg_memory_per_process = 0
            
            if not process_df.empty:
                # Calculate average CPU and memory per process
                avg_cpu_per_process = process_df['cpu_percent'].mean()
                if 'memory' in process_df.columns:
                    avg_memory_per_process = process_df['memory'].mean() / (1024*1024)  # Convert to MB
            
            # GPU Efficiency - higher is better
            # This is a simple metric: GPU_efficiency = avg_GPU_utilization / avg_CPU_utilization
            # Higher values suggest we're making better use of the GPU relative to CPU
            gpu_efficiency = avg_gpu / max(0.1, avg_cpu_per_process) if avg_cpu_per_process > 0 else 0
            
            # Add to the scenario data
            scenario_data.append({
                'name': params['name'],
                'batch_size': params['batch_size'],
                'hidden_size': params['hidden_size'],
                'avg_gpu': avg_gpu,
                'max_gpu': max_gpu,
                'std_gpu': std_gpu,
                'avg_cpu': avg_cpu_per_process,
                'avg_memory_mb': avg_memory_per_process,
                'gpu_efficiency': gpu_efficiency
            })
            
        except Exception as e:
            print(f"Error analyzing {monitor_log}: {e}")
    
    if not scenario_data:
        print("No valid scenario data found for analysis")
        return
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(scenario_data)
    
    # Display results
    print("\nGPU Utilization Summary:")
    print(results_df[['name', 'batch_size', 'hidden_size', 'avg_gpu', 'max_gpu', 'std_gpu', 'gpu_efficiency']])
    
    # Create output directory for analysis results
    analysis_dir = os.path.join(args.output_dir, "analysis_results")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Save results to CSV
    results_csv = os.path.join(analysis_dir, f"gpu_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nDetailed results saved to {results_csv}")
    
    # Create visualizations
    create_visualizations(results_df, analysis_dir)
    
    # Provide recommendations
    print("\nRECOMMENDATIONS:")
    
    # Find the most efficient configuration
    if not results_df.empty:
        best_efficiency = results_df.loc[results_df['gpu_efficiency'].idxmax()]
        print(f"Most efficient configuration: {best_efficiency['name']}")
        print(f"  - Batch size: {best_efficiency['batch_size']}")
        print(f"  - Hidden size: {best_efficiency['hidden_size']}")
        print(f"  - GPU efficiency score: {best_efficiency['gpu_efficiency']:.2f}")
        print(f"  - Average GPU utilization: {best_efficiency['avg_gpu']:.2f}%")
    
    # Check for underutilization
    if results_df['avg_gpu'].mean() < 50:
        print("\nGPU appears to be underutilized across scenarios (avg < 50%).")
        print("Consider increasing batch size or model complexity to better utilize the GPU.")
    
    # Check for potential bottlenecks
    batch_effect = results_df.groupby('batch_size')['avg_gpu'].mean()
    if batch_effect.max() - batch_effect.min() > 20:  # significant difference
        print("\nBatch size has a significant effect on GPU utilization.")
        optimal_batch = batch_effect.idxmax()
        print(f"The optimal batch size appears to be around {optimal_batch}.")
    
    print("\nAnalysis complete!")

def create_visualizations(results_df, output_dir):
    """Create visualizations for the results"""
    if results_df.empty:
        return
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Compare GPU utilization across scenarios
    plt.figure(figsize=(12, 6))
    ax = results_df.plot(x='name', y=['avg_gpu', 'max_gpu'], kind='bar', 
                         ylabel='GPU Utilization (%)', 
                         title='GPU Utilization by Scenario')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'gpu_utilization_by_scenario_{timestamp}.png'))
    
    # 2. Batch size vs. GPU utilization
    if len(results_df['batch_size'].unique()) > 1:
        plt.figure(figsize=(10, 6))
        batch_groups = results_df.groupby('batch_size')[['avg_gpu', 'max_gpu']].mean().reset_index()
        batch_groups.plot(x='batch_size', y=['avg_gpu', 'max_gpu'], marker='o', 
                          ylabel='GPU Utilization (%)', 
                          title='GPU Utilization by Batch Size')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'gpu_by_batch_size_{timestamp}.png'))
    
    # 3. Model size vs. GPU utilization
    if len(results_df['hidden_size'].unique()) > 1:
        plt.figure(figsize=(10, 6))
        model_groups = results_df.groupby('hidden_size')[['avg_gpu', 'max_gpu']].mean().reset_index()
        model_groups.plot(x='hidden_size', y=['avg_gpu', 'max_gpu'], marker='o', 
                         ylabel='GPU Utilization (%)', 
                         title='GPU Utilization by Model Size (Hidden Size)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'gpu_by_model_size_{timestamp}.png'))
    
    # 4. GPU Efficiency comparison
    plt.figure(figsize=(12, 6))
    ax = results_df.plot(x='name', y='gpu_efficiency', kind='bar', 
                         color='green', 
                         ylabel='GPU Efficiency Score', 
                         title='GPU Efficiency by Scenario (higher is better)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'gpu_efficiency_{timestamp}.png'))
    
    print(f"Visualizations saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='GPU Utilization Analysis for Chart-Hero')
    parser.add_argument('--scenarios', type=str, default=None,
                        help='JSON file containing scenarios to test')
    parser.add_argument('--output-dir', type=str, default='gpu_analysis',
                        help='Directory to store analysis results')
    parser.add_argument('--run-single', type=str, default=None,
                        help='Run a single scenario by name')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load scenarios
    scenarios = DEFAULT_SCENARIOS
    if args.scenarios:
        try:
            with open(args.scenarios, 'r') as f:
                scenarios = json.load(f)
        except Exception as e:
            print(f"Error loading scenarios file: {e}")
            print("Using default scenarios instead.")
    
    # Filter to single scenario if specified
    if args.run_single:
        scenarios = [s for s in scenarios if s['name'] == args.run_single]
        if not scenarios:
            print(f"No scenario found with name '{args.run_single}'")
            return
    
    # Run all scenarios
    scenario_results = []
    for scenario in scenarios:
        result_dir = run_scenario(scenario, args)
        scenario_results.append(result_dir)
        
        # Wait a bit between scenarios to let system stabilize
        if result_dir:
            print(f"Completed scenario: {scenario['name']}")
            print("Waiting 30 seconds before next scenario...\n")
            time.sleep(30)
    
    # Analyze the results
    analyze_results(scenario_results, args)

if __name__ == "__main__":
    main()
