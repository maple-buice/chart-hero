#!/usr/bin/env python3
"""
GPU Performance Analysis for PyTorch on M1 Mac
This script analyzes training logs to report GPU performance metrics
"""

import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def parse_pytorch_lightning_logs(log_dir):
    """Parse PyTorch Lightning logs to extract GPU usage and performance metrics"""
    metrics = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_f1': [],
        'val_f1': [],
        'gpu_mem_allocated': [],
        'batch_idx': [],
        'timestamp': []
    }
    
    # Find the most recent version directory
    versions = [d for d in os.listdir(log_dir) if d.startswith('version_')]
    if not versions:
        print(f"No version directories found in {log_dir}")
        return None
    
    # Sort versions numerically
    versions.sort(key=lambda x: int(x.split('_')[1]))
    latest_version = versions[-1]
    
    # Read the events file
    events_file = os.path.join(log_dir, latest_version, [f for f in os.listdir(os.path.join(log_dir, latest_version)) 
                                                        if f.startswith('events.out.tfevents')][0])
    
    print(f"Analyzing log file: {events_file}")
    
    # Use tensorboard's event parser to extract metrics
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        event_acc = EventAccumulator(events_file)
        event_acc.Reload()
        
        # Extract available tags
        tags = event_acc.Tags()['scalars']
        print(f"Available metrics: {tags}")
        
        # Process each tag
        for tag in tags:
            events = event_acc.Scalars(tag)
            for event in events:
                # Only process metrics we're interested in
                tag_parts = tag.split('/')
                metric = tag_parts[-1]
                
                if metric in metrics:
                    metrics[metric].append(event.value)
                    
                    # Add timestamp for this event
                    if 'timestamp' not in metrics:
                        metrics['timestamp'] = []
                    metrics['timestamp'].append(event.wall_time)
        
        # Create DataFrame
        df = pd.DataFrame(metrics)
        return df
    
    except ImportError:
        print("TensorBoard not installed. Please install with: pip install tensorboard")
        return None
    except Exception as e:
        print(f"Error parsing log file: {e}")
        return None

def analyze_gpu_performance(df):
    """Analyze GPU performance metrics from the dataframe"""
    if df is None or df.empty:
        print("No data available for analysis")
        return
    
    # Basic statistics
    print("\n===== Performance Analysis =====")
    
    if 'gpu_mem_allocated' in df.columns and not df['gpu_mem_allocated'].empty:
        avg_gpu_mem = df['gpu_mem_allocated'].mean()
        max_gpu_mem = df['gpu_mem_allocated'].max()
        print(f"Average GPU Memory Usage: {avg_gpu_mem:.2f} GB")
        print(f"Peak GPU Memory Usage: {max_gpu_mem:.2f} GB")
    
    if 'train_loss' in df.columns and not df['train_loss'].empty:
        initial_loss = df['train_loss'].iloc[0] if not df['train_loss'].empty else None
        final_loss = df['train_loss'].iloc[-1] if not df['train_loss'].empty else None
        
        if initial_loss and final_loss:
            loss_improvement = ((initial_loss - final_loss) / initial_loss) * 100
            print(f"Training Loss Improvement: {loss_improvement:.2f}%")
    
    if 'val_f1' in df.columns and not df['val_f1'].empty:
        best_f1 = df['val_f1'].max() if not df['val_f1'].empty else None
        if best_f1:
            print(f"Best Validation F1 Score: {best_f1:.4f}")
    
    # Calculate training speed
    if 'timestamp' in df.columns and len(df['timestamp']) > 1:
        training_time = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
        batches_processed = len(df['timestamp'])
        avg_batch_time = training_time / batches_processed
        print(f"Total Training Time: {training_time:.2f} seconds")
        print(f"Average Time per Batch: {avg_batch_time:.4f} seconds")
    
    # Plot metrics if available
    try:
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Training and Validation Loss
        if 'train_loss' in df.columns and not df['train_loss'].empty:
            plt.subplot(2, 2, 1)
            plt.plot(df['train_loss'], label='Training Loss')
            if 'val_loss' in df.columns and not df['val_loss'].empty:
                plt.plot(df['val_loss'], label='Validation Loss')
            plt.title('Loss over Time')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        
        # Plot 2: F1 Score
        if 'train_f1' in df.columns and not df['train_f1'].empty:
            plt.subplot(2, 2, 2)
            plt.plot(df['train_f1'], label='Training F1')
            if 'val_f1' in df.columns and not df['val_f1'].empty:
                plt.plot(df['val_f1'], label='Validation F1')
            plt.title('F1 Score over Time')
            plt.xlabel('Step')
            plt.ylabel('F1 Score')
            plt.legend()
            plt.grid(True)
        
        # Plot 3: GPU Memory Usage
        if 'gpu_mem_allocated' in df.columns and not df['gpu_mem_allocated'].empty:
            plt.subplot(2, 2, 3)
            plt.plot(df['gpu_mem_allocated'], label='GPU Memory')
            plt.title('GPU Memory Usage over Time')
            plt.xlabel('Step')
            plt.ylabel('Memory (GB)')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"gpu_performance_analysis_{timestamp}.png"
        plt.savefig(output_file)
        print(f"\nPerformance chart saved to: {output_file}")
    
    except Exception as e:
        print(f"Error creating performance chart: {e}")

def main():
    parser = argparse.ArgumentParser(description='Analyze GPU performance for PyTorch on M1 Mac')
    parser.add_argument('--log-dir', type=str, default='lightning_logs', 
                        help='Directory containing PyTorch Lightning logs')
    args = parser.parse_args()
    
    # Parse logs
    df = parse_pytorch_lightning_logs(args.log_dir)
    
    # Analyze performance
    analyze_gpu_performance(df)

if __name__ == "__main__":
    main()
