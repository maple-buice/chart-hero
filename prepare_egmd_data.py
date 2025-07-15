#!/usr/bin/env python3
"""
Main script to prepare E-GMD dataset for transformer training.

This script now acts as a simple wrapper around the new, streamlined
data preparation pipeline in `model_training.prepare_transformer_data`.
"""

import sys
import os
import argparse
import logging

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chart_hero.model_training.data_preparation import data_preparation

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Prepare E-GMD dataset for transformer training using the new pipeline.')
    parser.add_argument('--dataset-dir', type=str, 
                       default='datasets/e-gmd-v1.0.0',
                       help='Path to E-GMD dataset directory')
    parser.add_argument('--output-dir', type=str,
                       default='datasets/processed_transformer',
                       help='Output directory for processed .npy data')
    parser.add_argument('--sample-ratio', type=float, default=1.0,
                       help='Fraction of dataset to use (default: 1.0 = all data)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of parallel workers for data loading')
    
    args = parser.parse_args()

    logger.info("Starting data preparation with the new transformer pipeline...")
    logger.info(f"Dataset directory: {args.dataset_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Sample ratio: {args.sample_ratio}")
    logger.info(f"Number of workers: {args.num_workers}")

    try:
        data_prep = data_preparation(
            directory_path=args.dataset_dir,
            dataset='egmd',
            sample_ratio=args.sample_ratio
        )
        
        data_prep.create_audio_set(
            dir_path=args.output_dir,
            num_workers=args.num_workers
        )

        logger.info("Data preparation completed successfully!")
        logger.info("\n" + "="*50)
        logger.info("NEXT STEPS:")
        logger.info("1. Test the processed data:")
        logger.info("   python model_training/test_transformer_setup.py")
        logger.info(f"2. Start training:\n   python model_training/train_transformer.py --config auto --data-dir {args.output_dir} --audio-dir {args.dataset_dir}")
        logger.info("="*50)
    except Exception as e:
        logger.error(f"Error during data preparation: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
