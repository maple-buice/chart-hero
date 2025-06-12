import sys
import os
import logging
import time

# Ensure the model_training package is discoverable
# This assumes the script is run from the root of the chart-hero repository
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from model_training.data_preparation import data_preparation

# --- Configuration ---
# !!! PLEASE REPLACE THESE PATHS !!!
EGMD_DATASET_DIR = "/Users/maple/Repos/chart-hero/datasets/e-gmd-v1.0.0"  # Path to your E-GMD dataset
PROCESSED_OUTPUT_DIR = "/Users/maple/Repos/chart-hero/datasets/processed_test_parallel" # Path for processed output

SAMPLE_RATIO = 0.01  # Use a small fraction of the dataset for quick testing (e.g., 1%)
# SAMPLE_RATIO = 0.001 # Even smaller for a very quick test
NUM_BATCHES_TO_CREATE = 2 # Create only a few batches for testing

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting local test for data_preparation parallel processing...")

    if not os.path.exists(EGMD_DATASET_DIR):
        logger.error(f"E-GMD dataset directory not found: {EGMD_DATASET_DIR}")
        logger.error("Please update the EGMD_DATASET_DIR variable in this script.")
        return

    if not os.path.exists(PROCESSED_OUTPUT_DIR):
        logger.info(f"Creating processed output directory: {PROCESSED_OUTPUT_DIR}")
        os.makedirs(PROCESSED_OUTPUT_DIR)

    logger.info(f"Using E-GMD directory: {EGMD_DATASET_DIR}")
    logger.info(f"Using processed output directory: {PROCESSED_OUTPUT_DIR}")
    logger.info(f"Sample ratio: {SAMPLE_RATIO}")
    logger.info(f"Number of batches to create: {NUM_BATCHES_TO_CREATE}")

    # 1. Initialize data_preparation
    # This will trigger the parallel duration calculation in __init__
    try:
        logger.info("Initializing data_preparation...")
        init_start_time = time.perf_counter()
        data_prep_instance = data_preparation(
            directory_path=EGMD_DATASET_DIR,
            dataset='egmd',
            sample_ratio=SAMPLE_RATIO,
            diff_threshold=1.0,
            n_jobs=-1  # Use all available cores for __init__
        )
        init_end_time = time.perf_counter()
        logger.info(f"data_preparation initialization took {init_end_time - init_start_time:.2f} seconds.")
        logger.info(f"Number of MIDI-WAV pairs after init: {len(data_prep_instance.midi_wav_map)}")

        if not data_prep_instance.midi_wav_map.empty:
            # 2. Call create_audio_set with parameters to trigger parallel processing
            logger.info("Calling create_audio_set with memory_limit_gb=50 (to trigger parallel mode)...")
            create_set_start_time = time.perf_counter()
            data_prep_instance.create_audio_set(
                pad_before=0.1,
                pad_after=0.1,
                fix_length=10.0,
                batching=True,
                dir_path=PROCESSED_OUTPUT_DIR,
                num_batches=NUM_BATCHES_TO_CREATE, # Limit number of batches for testing
                memory_limit_gb=50 # This should trigger the parallel path in create_audio_set
            )
            create_set_end_time = time.perf_counter()
            logger.info(f"create_audio_set took {create_set_end_time - create_set_start_time:.2f} seconds.")
            logger.info(f"Test finished. Check {PROCESSED_OUTPUT_DIR} for output files and logs for errors.")
        else:
            logger.warning("midi_wav_map is empty after initialization. Cannot proceed with create_audio_set.")
            logger.warning("This might be due to a very small sample_ratio or issues with the dataset CSV.")

    except Exception as e:
        logger.error(f"An error occurred during the test: {e}", exc_info=True)

if __name__ == "__main__":
    main()
