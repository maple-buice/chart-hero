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
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 = all cores)')
    parser.add_argument('--memory-limit-gb', type=float, default=4.0,
                       help='Memory limit in GB (default: 4.0)')
    parser.add_argument('--conservative', action='store_true',
                       help='Use ultra-conservative settings for low-memory systems')
    
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
    logger.info(f"Parallel jobs: {args.n_jobs}")
    logger.info(f"Memory limit: {args.memory_limit_gb} GB")
    logger.info(f"Conservative mode: {args.conservative}")
    
    # Apply conservative settings if requested
    if args.conservative:
        args.n_jobs = 1
        args.memory_limit_gb = min(args.memory_limit_gb, 2.0)
        args.num_batches = max(args.num_batches, 200)  # Many more, much smaller batches
        logger.info(f"Applied conservative settings: n_jobs=1, memory_limit=2GB, num_batches={args.num_batches}")
    
    try:
        # Create data preparation instance
        logger.info("Initializing data preparation...")
        data_prep = data_preparation(
            directory_path=str(dataset_path),
            dataset='egmd',
            sample_ratio=args.sample_ratio,
            diff_threshold=args.diff_threshold,
            n_jobs=args.n_jobs
        )
        
        logger.info(f"Found {len(data_prep.midi_wav_map)} valid audio/MIDI pairs")
        
        # Process and create batched data
        logger.info("Creating audio set with transformer-compatible processing...")
        data_prep.create_audio_set(
            pad_before=0.1,  # 100ms padding before note
            pad_after=0.1,   # 100ms padding after note
            fix_length=args.fix_length,
            batching=True,
            dir_path=str(output_path),
            num_batches=args.num_batches,
            memory_limit_gb=args.memory_limit_gb
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