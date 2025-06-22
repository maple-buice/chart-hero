#!/usr/bin/env python3
"""
Script to prepare E-GMD dataset for transformer training.
Processes the raw E-GMD data into transformer-compatible format.
"""

import os
import sys
import logging
from pathlib import Path
import argparse

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_training.data_preparation import data_preparation
from model_training.transformer_data import convert_legacy_data

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Prepare E-GMD dataset for transformer training')
    parser.add_argument('--dataset-dir', type=str, 
                       default='datasets/e-gmd-v1.0.0',
                       help='Path to E-GMD dataset directory')
    parser.add_argument('--output-dir', type=str,
                       default='datasets/processed',
                       help='Output directory for processed data')
    parser.add_argument('--sample-ratio', type=float, default=1.0,
                       help='Fraction of dataset to use (default: 1.0 = all data)')
    parser.add_argument('--diff-threshold', type=float, default=1.0,
                       help='Maximum audio/MIDI duration difference in seconds')
    parser.add_argument('--num-batches', type=int, default=50,
                       help='Number of batches to create')
    parser.add_argument('--fix-length', type=float, default=10.0,
                       help='Fixed audio segment length in seconds')
    parser.add_argument('--memory-limit-gb', type=float, default=48.0,
                       help='Memory limit in GB (default: 48.0 for high-memory systems)')
    parser.add_argument('--conservative', action='store_true',
                       help='Use ultra-conservative settings for low-memory systems')
    parser.add_argument('--high-performance', action='store_true',
                       help='Use optimized settings for high-memory systems (64GB+ RAM)')
    parser.add_argument('--ultra-performance', action='store_true',
                       help='Use high-memory settings for very large batches (128GB+ RAM)')
    parser.add_argument('--extreme-performance', action='store_true',
                       help='Use high-memory settings for larger batches (requires 64GB+ RAM)')
    parser.add_argument('--batch-size-multiplier', type=float, default=2.0,
                       help='Multiply batch sizes by this factor (default: 2.0 for better performance)')
    parser.add_argument('--disable-parallel', action='store_true',
                       help='Force sequential processing (disables smart parallelization)')
    
    args = parser.parse_args()
    
    # Validate paths
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_path}")
        sys.exit(1)
    
    csv_file = dataset_path / 'e-gmd-v1.0.0.csv'
    if not csv_file.exists():
        logger.error(f"CSV file not found: {csv_file}")
        sys.exit(1)
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing E-GMD dataset from: {dataset_path}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Sample ratio: {args.sample_ratio}")
    logger.info(f"Diff threshold: {args.diff_threshold} seconds")
    logger.info(f"Number of batches: {args.num_batches}")
    logger.info(f"Fixed length: {args.fix_length} seconds")
    logger.info(f"Memory limit: {args.memory_limit_gb} GB")
    logger.info(f"Conservative mode: {args.conservative}")
    logger.info(f"High-performance mode: {args.high_performance}")
    logger.info(f"Ultra-performance mode: {args.ultra_performance}")
    logger.info(f"Batch size multiplier: {args.batch_size_multiplier}")
    logger.info(f"Disable parallel: {args.disable_parallel}")
    
    # Detect system memory for automatic optimization
    import psutil
    system_memory_gb = psutil.virtual_memory().total / (1024**3)
    logger.info(f"Detected system memory: {system_memory_gb:.1f} GB")
    
    # Apply mode-specific settings
    if sum([args.conservative, args.high_performance, args.ultra_performance, args.extreme_performance]) > 1:
        logger.error("Cannot use multiple performance modes simultaneously")
        sys.exit(1)
    elif args.conservative:
        args.memory_limit_gb = min(args.memory_limit_gb, 4.0)
        args.num_batches = max(args.num_batches, 200)  # Many small batches
        logger.info(f"Applied conservative settings: memory_limit=4GB, num_batches={args.num_batches}")
    elif args.extreme_performance:
        # Extreme mode: Use more memory for larger batches (moderate performance benefit)
        args.memory_limit_gb = max(args.memory_limit_gb, 56.0)  
        args.num_batches = max(3, args.num_batches // 20)  # Fewer, larger batches
        args.batch_size_multiplier = max(args.batch_size_multiplier, 10.0)  
        logger.info(f"🚀 EXTREME PERFORMANCE MODE: memory_limit={args.memory_limit_gb}GB, num_batches={args.num_batches}")
        logger.info(f"🚀 Batch size multiplier: {args.batch_size_multiplier}x (larger batches, moderate speed improvement)")
        logger.info("Note: Audio I/O is the main bottleneck, not batch size")
    elif args.ultra_performance:
        # For systems with 128GB+ RAM - use high memory for larger batches
        args.memory_limit_gb = max(args.memory_limit_gb, 80.0)  # Use lots of memory
        args.num_batches = max(5, args.num_batches // 10)  # Very few, very large batches
        args.batch_size_multiplier = max(args.batch_size_multiplier, 8.0)  # 8x larger batches
        logger.info(f"Applied ULTRA-PERFORMANCE settings: memory_limit={args.memory_limit_gb}GB, num_batches={args.num_batches}")
        logger.info(f"Batch size multiplier: {args.batch_size_multiplier}x - Using maximum system resources!")
    elif args.high_performance:
        # For systems with 64GB+ RAM - aggressive but safe
        args.memory_limit_gb = max(args.memory_limit_gb, 48.0)  # Use most available memory
        args.num_batches = max(10, args.num_batches // 5)  # Fewer, larger batches
        args.batch_size_multiplier = max(args.batch_size_multiplier, 5.0)  # 5x larger batches
        logger.info(f"Applied HIGH-PERFORMANCE settings: memory_limit={args.memory_limit_gb}GB, num_batches={args.num_batches}")
        logger.info(f"Batch size multiplier: {args.batch_size_multiplier}x - Optimized for 64GB+ systems!")
    elif system_memory_gb >= 32:
        # Auto-optimize for high-memory systems - MORE AGGRESSIVE defaults
        if system_memory_gb >= 64:
            args.memory_limit_gb = max(args.memory_limit_gb, 48.0)  # Use 75% of 64GB
            args.num_batches = max(8, args.num_batches // 6)  # Much fewer batches
            args.batch_size_multiplier = max(args.batch_size_multiplier, 4.0)  # 4x larger batches
            logger.info(f"AUTO-OPTIMIZED for {system_memory_gb:.0f}GB system: memory_limit={args.memory_limit_gb}GB, num_batches={args.num_batches}")
            logger.info(f"Batch size multiplier: {args.batch_size_multiplier}x - using more memory for larger batches")
        else:
            args.memory_limit_gb = max(args.memory_limit_gb, 24.0)
            args.num_batches = max(15, args.num_batches // 3)
            args.batch_size_multiplier = max(args.batch_size_multiplier, 2.5)
            logger.info(f"AUTO-OPTIMIZED for {system_memory_gb:.0f}GB system: memory_limit={args.memory_limit_gb}GB, num_batches={args.num_batches}")
    
    # Show final optimization summary
    if args.batch_size_multiplier > 1.0:
        estimated_records_per_batch = int(3000 * args.batch_size_multiplier)  # Updated estimate
        logger.info(f"Batch optimization: Processing ~{estimated_records_per_batch} records per batch (vs ~3000 default)")
        logger.info(f"Note: Batch size mainly affects memory usage, not speed (audio I/O is the bottleneck)")
        logger.info(f"Memory efficiency: Using {args.memory_limit_gb}GB limit on {system_memory_gb:.0f}GB system ({args.memory_limit_gb/system_memory_gb*100:.0f}% utilization)")
    
    # Provide easy command for more memory usage
    if system_memory_gb >= 64 and not any([args.conservative, args.high_performance, args.ultra_performance, args.extreme_performance]):
        logger.info("\n" + "🚀 " + "="*60)
        logger.info("MEMORY TIP: Your system has excellent resources!")
        logger.info("For larger batches, try:")
        logger.info("  python prepare_egmd_data.py --extreme-performance")
        logger.info("This will use 56GB RAM for larger batches (moderate speed improvement)")
        logger.info("="*60 + " 🚀\n")
    
    try:
        # Create data preparation instance
        logger.info("Initializing data preparation...")
        data_prep = data_preparation(
            directory_path=str(dataset_path),
            dataset='egmd',
            sample_ratio=args.sample_ratio,
            diff_threshold=args.diff_threshold
        )
        
        logger.info(f"Found {len(data_prep.midi_wav_map)} valid audio/MIDI pairs")
        
        # Process and create batched data
        logger.info("Creating audio set with transformer-compatible processing...")
        if args.disable_parallel:
            logger.info("Parallel processing disabled by user - using sequential mode")
            enable_parallelization = False
        else:
            logger.info("Using smart parallelization logic - will auto-select optimal processing mode")
            enable_parallelization = True
            
        data_prep.create_audio_set(
            pad_before=0.1,  # 100ms padding before note
            pad_after=0.1,   # 100ms padding after note
            fix_length=args.fix_length,
            batching=True,
            dir_path=str(output_path),
            num_batches=args.num_batches,
            memory_limit_gb=args.memory_limit_gb,
            batch_size_multiplier=args.batch_size_multiplier,
            enable_process_parallelization=enable_parallelization
        )
        
        logger.info("Data preparation completed successfully!")
        
        # List created files
        processed_files = list(output_path.glob("*.pkl"))
        logger.info(f"Created {len(processed_files)} processed data files:")
        
        train_files = [f for f in processed_files if 'train' in f.name]
        val_files = [f for f in processed_files if 'val' in f.name]
        test_files = [f for f in processed_files if 'test' in f.name]
        
        logger.info(f"  - Training files: {len(train_files)}")
        logger.info(f"  - Validation files: {len(val_files)}")
        logger.info(f"  - Test files: {len(test_files)}")
        
        # Show sample file sizes
        if train_files:
            sample_file = train_files[0]
            import pandas as pd
            sample_df = pd.read_pickle(sample_file)
            logger.info(f"Sample file {sample_file.name} contains {len(sample_df)} training examples")
        
        logger.info("\n" + "="*50)
        logger.info("NEXT STEPS:")
        logger.info("1. Test the processed data:")
        logger.info("   python model_training/test_transformer_setup.py")
        logger.info("2. Start training:")
        logger.info(f"   python model_training/train_transformer.py --config auto --data-dir {output_path} --audio-dir {dataset_path}")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Error during data preparation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()