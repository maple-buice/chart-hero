# %%
# Import packages and declare variables

import logging
import argparse
import time
import os # Import os if needed for path checks

# Configure logging FIRST
logging.basicConfig(level=logging.INFO, # Or logging.DEBUG for more detail
                    format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s", # Added module/line number
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__) # Get logger for this module (train.py)

# Import functions AFTER basicConfig is set
from model_training.create_audio_set import create_audio_set # Assuming this function exists now
from model_training.train_model import train_model
from model_training.label_data import label_data
# Import utils if needed for path setup
# from utils.file_utils import setup_output_directory # Example

def main(dataset_path: str, output_dir: str, skip_audio_set: bool, skip_label_data: bool):
    overall_start_time = time.perf_counter()
    logger.info("--- Starting Main Training Pipeline ---")
    logger.info(f"Dataset Path: {dataset_path}")
    logger.info(f"Output Directory: {output_dir}") # Ensure output_dir is used or passed correctly
    logger.info(f"Skip Audio Set Creation: {skip_audio_set}")
    logger.info(f"Skip Data Labeling: {skip_label_data}")

    # Optional: Setup output directory structure here if needed
    # setup_output_directory(output_dir)

    if not skip_audio_set:
        try:
            logger.info("=== Starting Step: Dataset Creation (create_audio_set) ===")
            step_start_time = time.perf_counter()
            # Pass relevant paths if needed, e.g., output_dir for batches
            # Assuming create_audio_set uses utils or is adapted
            create_audio_set(dataset_path) # Make sure this function exists and is imported
            step_end_time = time.perf_counter()
            logger.info(f"=== Finished Step: Dataset Creation completed in {step_end_time - step_start_time:.2f} seconds. ===")
        except NameError:
             logger.error("Function 'create_audio_set' not found. Please ensure it's defined and imported correctly.")
             return # Stop pipeline if a step function is missing
        except Exception as e:
            logger.error(f"!!! Error during dataset creation: {e}", exc_info=True)
            return # Stop pipeline on error
    else:
        logger.info("--- Skipping Step: Dataset Creation ---")

    if not skip_label_data:
        try:
            logger.info("=== Starting Step: Data Labeling (label_data) ===")
            step_start_time = time.perf_counter()
            label_data() # Assuming label_data uses utils for paths
            step_end_time = time.perf_counter()
            logger.info(f"=== Finished Step: Data Labeling completed in {step_end_time - step_start_time:.2f} seconds. ===")
        except Exception as e:
            logger.error(f"!!! Error during data labeling: {e}", exc_info=True)
            return # Stop pipeline on error
    else:
        logger.info("--- Skipping Step: Data Labeling ---")

    try:
        logger.info("=== Starting Step: Model Training (train_model) ===")
        step_start_time = time.perf_counter()
        # Pass output_dir if train_model needs it directly, otherwise it uses utils
        train_model() # Pass output_dir if needed: train_model(output_dir)
        step_end_time = time.perf_counter()
        logger.info(f"=== Finished Step: Model Training completed in {step_end_time - step_start_time:.2f} seconds. ===")
    except Exception as e:
        logger.error(f"!!! Error during model training: {e}", exc_info=True)
        return # Stop pipeline on error

    overall_end_time = time.perf_counter()
    logger.info(f"--- Total Pipeline Execution Time: {overall_end_time - overall_start_time:.2f} seconds. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full drum recognition model training pipeline.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the root dataset directory (e.g., e-gmd-v1.0.0).")
    parser.add_argument("--output_dir", type=str, required=True, help="Root directory for all generated outputs (audio sets, labeled data, model files, backups).")
    parser.add_argument("--skip_audio_set", action="store_true", help="Skip the initial audio set creation step.")
    parser.add_argument("--skip_label_data", action="store_true", help="Skip the data labeling (spectrogram generation) step.")
    # Add other arguments as needed (e.g., batch_size, epochs, sample_ratio)
    args = parser.parse_args()

    # Optional: Validate paths early
    if not os.path.isdir(args.dataset_path):
        logger.error(f"Dataset path not found or not a directory: {args.dataset_path}")
        exit(1)
    # No need to check output_dir existence here, utils or steps should create it

    # Pass arguments to main
    main(args.dataset_path, args.output_dir, args.skip_audio_set, args.skip_label_data)
