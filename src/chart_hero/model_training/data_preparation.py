import logging
import shutil
import sys
from pathlib import Path

import mido
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, random_split
from tqdm import tqdm

from chart_hero.model_training.transformer_config import (
    DRUM_HIT_MAP,
    DRUM_HIT_TO_INDEX,
    TARGET_CLASSES,
    get_config,
)
from chart_hero.model_training.transformer_data import SpectrogramProcessor

logger = logging.getLogger(__name__)


class EGMDRawDataset(Dataset):
    """
    PyTorch Dataset for the E-GMD dataset.
    Processes raw audio/MIDI pairs into full-length spectrograms and frame-by-frame label matrices.
    """

    def __init__(self, data_map: pd.DataFrame, dataset_dir: str, config):
        self.data_map = data_map
        self.dataset_dir = Path(dataset_dir)
        self.config = config
        self.processor = SpectrogramProcessor(config)

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx: int):
        row = self.data_map.iloc[idx]
        audio_filename = self.dataset_dir / row["audio_filename"]
        midi_filename = self.dataset_dir / row["midi_filename"]

        # Load and process audio to full spectrogram
        try:
            audio, sr = torchaudio.load(audio_filename)
            if sr != self.config.sample_rate:
                audio = torchaudio.functional.resample(
                    audio, sr, self.config.sample_rate
                )
            spectrogram = self.processor.audio_to_spectrogram(audio)
        except Exception as e:
            logger.warning(f"Could not load audio file {audio_filename}: {e}")
            return None, None

        # Create frame-by-frame label matrix from MIDI
        try:
            label_matrix = self._create_label_matrix(
                midi_filename, spectrogram.shape[1]
            )
        except Exception as e:
            logger.warning(f"Could not process MIDI file {midi_filename}: {e}")
            return None, None

        return spectrogram, label_matrix

    def _create_label_matrix(self, midi_path: Path, num_time_frames: int):
        """Creates a frame-by-frame label matrix from a MIDI file."""
        try:
            midi_file = mido.MidiFile(midi_path)
        except Exception as e:
            logger.warning(f"Could not read MIDI file {midi_path}: {e}")
            return None

        ticks_per_beat = midi_file.ticks_per_beat
        tempo = 500000  # Default tempo

        for msg in midi_file.tracks[0]:
            if msg.type == "set_tempo":
                tempo = msg.tempo
                break

        label_matrix = torch.zeros(
            (num_time_frames, len(TARGET_CLASSES)), dtype=torch.float32
        )

        time_log = 0

        for msg in midi_file.tracks[-1]:
            time_log += msg.time
            time_sec = mido.tick2second(time_log, ticks_per_beat, tempo)
            frame_index = int(
                time_sec * self.config.sample_rate / self.config.hop_length
            )

            if frame_index >= num_time_frames:
                continue

            if msg.type == "note_on" and msg.velocity > 0:
                target_hit = DRUM_HIT_MAP.get(msg.note)
                if target_hit:
                    target_index = DRUM_HIT_TO_INDEX.get(target_hit)
                    if target_index is not None:
                        label_matrix[frame_index, target_index] = 1

        return label_matrix


def _save_segments(
    spectrogram: torch.Tensor,
    label_matrix: torch.Tensor,
    segment_length: int,
    output_dir: Path,
    base_filename: str,
):
    """Segments spectrogram and labels and saves them to disk."""
    num_frames = spectrogram.shape[1]
    for i in range(0, num_frames, segment_length):
        end_frame = i + segment_length
        if end_frame > num_frames:
            continue  # Skip incomplete segments

        spec_segment = spectrogram[:, i:end_frame, :]
        label_segment = label_matrix[i:end_frame, :]

        # Ensure correct dimensions before saving
        if spec_segment.dim() == 2:
            spec_segment = spec_segment.unsqueeze(0)  # Add channel dim: [C, T, F]
        if spec_segment.dim() != 3:
            logger.warning(
                f"Skipping segment {base_filename}_{i}: incorrect spec dimensions: {spec_segment.dim()}"
            )
            continue

        if label_segment.dim() != 2:
            logger.warning(
                f"Skipping segment {base_filename}_{i}: incorrect label dimensions: {label_segment.dim()}"
            )
            continue

        # Save to .npy files
        spec_filename = output_dir / f"{base_filename}_{i}_mel.npy"
        label_filename = output_dir / f"{base_filename}_{i}_label.npy"
        np.save(spec_filename, spec_segment.numpy())
        np.save(label_filename, label_segment.numpy())


def main(
    dataset_dir: str,
    output_dir: str,
    config,
    limit: int = None,
    clear_output: bool = False,
    no_progress: bool = False,
):
    """
    Main function to process the E-GMD dataset.
    """
    output_path = Path(output_dir)
    if clear_output and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data map
    data_map_file = Path(dataset_dir) / "e-gmd-v1.0.0.csv"
    if not data_map_file.exists():
        raise FileNotFoundError(f"Data map not found at {data_map_file}")
    data_map = pd.read_csv(data_map_file)

    if limit:
        data_map = data_map.head(limit)

    # Create dataset
    dataset = EGMDRawDataset(data_map, dataset_dir, config)

    # Split dataset
    generator = torch.Generator().manual_seed(config.seed)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_indices, val_indices, test_indices = random_split(
        range(len(dataset)), [train_size, val_size, test_size], generator=generator
    )

    split_map = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }

    segment_length_frames = int(
        config.max_audio_length * config.sample_rate / config.hop_length
    )

    # Disable progress bar if not in a TTY or if --no-progress is set
    progress_disabled = no_progress or not sys.stdout.isatty()

    # Process and save data
    for split, indices in split_map.items():
        split_dir = output_path / split
        split_dir.mkdir(exist_ok=True)
        for i in tqdm(
            indices, desc=f"Processing {split} set", disable=progress_disabled
        ):
            spectrogram, label_matrix = dataset[i]
            if spectrogram is None or label_matrix is None:
                continue

            base_filename = f"{split}_{i}"
            _save_segments(
                spectrogram,
                label_matrix,
                segment_length_frames,
                split_dir,
                base_filename,
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process E-GMD dataset.")
    parser.add_argument(
        "--dataset-dir", type=str, required=True, help="Path to the E-GMD dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to save the processed data.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="local",
        help="Configuration to use (local, cloud, etc.)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit the number of files to process."
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Clear the output directory before processing.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = get_config(args.config)
    main(
        args.dataset_dir,
        args.output_dir,
        config,
        args.limit,
        args.clear_output,
        args.no_progress,
    )
