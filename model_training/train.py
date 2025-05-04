# %%
# Import packages and declare variables

from model_training.create_audio_set import create_audio_set
from model_training.train_model import train_model
from model_training.label_data import label_data
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main(dataset_path: str, output_dir: str, skip_audio_set: bool, skip_label_data: bool):
    if not skip_audio_set:
        try:
            logging.info("Starting dataset creation...")
            create_audio_set(dataset_path)
            logging.info("Dataset creation completed.")
        except Exception as e:
            logging.error(f"Error during dataset creation: {e}")
            return
    else:
        logging.info("Skipping dataset creation as requested.")

    if not skip_label_data:
        try:
            logging.info("Starting data labeling...")
            label_data()
            logging.info("Data labeling completed.")
        except Exception as e:
            logging.error(f"Error during data labeling: {e}")
            return
    else:
        logging.info("Skipping data labeling as requested.")

    try:
        logging.info("Starting model training...")
        train_model(output_dir)
        logging.info("Model training completed.")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train drum recognition model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model.")
    parser.add_argument("--skip_audio_set", action="store_true", help="Skip creating the audio set.")
    parser.add_argument("--skip_label_data", action="store_true", help="Skip labeling the data.")
    args = parser.parse_args()

    main(args.dataset_path, args.output_dir, args.skip_audio_set, args.skip_label_data)
