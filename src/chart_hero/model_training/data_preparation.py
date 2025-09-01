import logging
import shutil
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, random_split
from tqdm import tqdm

from chart_hero.model_training.augment_audio import (
    augment_dynamic_eq,
    augment_pitch_jitter,
    augment_time_stretch,
)
from chart_hero.model_training.transformer_config import get_config
from chart_hero.utils.midi_utils import MidiProcessor

logger = logging.getLogger(__name__)


def create_transient_enhanced_spectrogram(y, sr, n_fft, hop_length, n_mels):
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    min_len = min(log_mel_spec.shape[1], len(onset_env))
    log_mel_spec = log_mel_spec[:, :min_len]
    onset_env = onset_env[:min_len]
    if np.max(onset_env) > 0:
        onset_env = onset_env / np.max(onset_env)
    return log_mel_spec * onset_env


class EGMDRawDataset(Dataset[tuple[torch.Tensor | None, torch.Tensor | None]]):
    def __init__(self, data_map: pd.DataFrame, dataset_dir: str, config):
        self.data_map = data_map
        self.dataset_dir = Path(dataset_dir)
        self.config = config
        self.midi_processor = MidiProcessor(config)

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx: int):
        row = self.data_map.iloc[idx]
        audio_filename = self.dataset_dir / row["audio_filename"]
        midi_filename = self.dataset_dir / row["midi_filename"]

        try:
            audio_np, sr = librosa.load(audio_filename, sr=self.config.sample_rate)
            spec_np = create_transient_enhanced_spectrogram(
                y=audio_np,
                sr=sr,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                n_mels=self.config.n_mels,
            )
            spectrogram = torch.from_numpy(spec_np).float().unsqueeze(0)
        except Exception as e:
            logger.warning(f"Could not process audio file {audio_filename}: {e}")
            return None, None

        label_matrix = self.midi_processor.create_label_matrix(
            midi_filename, spectrogram.shape[2]
        )
        if label_matrix is None:
            return None, None

        return spectrogram, label_matrix


def _save_segments(
    spectrogram: torch.Tensor,
    label_matrix: torch.Tensor,
    segment_length: int,
    output_dir: Path,
    base_filename: str,
):
    spec_np = spectrogram.squeeze(0).numpy()
    num_frames = spec_np.shape[1]
    min_spec_val = np.min(spec_np)

    total = 0
    zero_label = 0
    for i in range(0, num_frames, segment_length):
        end_frame = i + segment_length
        spec_segment = spec_np[:, i:end_frame]
        label_segment = label_matrix[i:end_frame, :]

        if spec_segment.shape[1] < segment_length:
            pad_width = segment_length - spec_segment.shape[1]
            spec_segment = np.pad(
                spec_segment,
                ((0, 0), (0, pad_width)),
                mode="constant",
                constant_values=min_spec_val,
            )
            label_segment = np.pad(
                label_segment,
                ((0, pad_width), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        spec_filename = output_dir / f"{base_filename}_{i}_mel.npy"
        label_filename = output_dir / f"{base_filename}_{i}_label.npy"
        np.save(spec_filename, spec_segment)
        np.save(label_filename, label_segment)

        total += 1
        if np.sum(label_segment) <= 0:
            zero_label += 1

    return total, zero_label


def main(
    dataset_dir: str,
    output_dir: str,
    config,
    limit: int | None = None,
    clear_output: bool = False,
    no_progress: bool = False,
    validate_labels: bool = False,
):
    output_path = Path(output_dir)
    if clear_output and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    data_map_file = Path(dataset_dir) / "e-gmd-v1.0.0.csv"
    if not data_map_file.exists():
        raise FileNotFoundError(f"Data map not found at {data_map_file}")
    data_map = pd.read_csv(data_map_file)
    if limit:
        data_map = data_map.head(limit)

    dataset = EGMDRawDataset(data_map, dataset_dir, config)
    generator = torch.Generator().manual_seed(config.seed)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_indices, val_indices, test_indices = random_split(
        TensorDataset(torch.arange(len(dataset))),
        [train_size, val_size, test_size],
        generator=generator,
    )
    split_map = {"train": train_indices, "val": val_indices, "test": test_indices}
    segment_length_frames = int(
        config.max_audio_length * config.sample_rate / config.hop_length
    )
    progress_disabled = no_progress or not sys.stdout.isatty()

    summary = {}
    for split, indices in split_map.items():
        split_dir = output_path / split
        split_dir.mkdir(exist_ok=True)
        # random_split returns torch.utils.data.Subset; iterate integer indices
        subset_indices = getattr(indices, "indices", list(range(len(indices))))
        split_total_segments = 0
        split_zero_segments = 0
        for idx in tqdm(
            subset_indices, desc=f"Processing {split} set", disable=progress_disabled
        ):
            original_spectrogram, label_matrix = dataset[int(idx)]
            if original_spectrogram is None:
                continue

            base_filename = f"{split}_{int(idx)}_original"
            t, z = _save_segments(
                original_spectrogram,
                label_matrix,
                segment_length_frames,
                split_dir,
                base_filename,
            )
            split_total_segments += t
            split_zero_segments += z

            if split == "train" and config.enable_timbre_augmentation:
                audio_path = (
                    dataset.dataset_dir
                    / dataset.data_map.iloc[int(idx)]["audio_filename"]
                )
                if not audio_path.exists():
                    continue
                audio_np, sr = librosa.load(audio_path, sr=config.sample_rate)
                augmentations = {
                    "pitch": augment_pitch_jitter(audio_np, sr),
                    "stretch": augment_time_stretch(audio_np),
                    "eq": augment_dynamic_eq(audio_np, sr),
                }
                for aug_name, aug_audio in augmentations.items():
                    spec = create_transient_enhanced_spectrogram(
                        aug_audio, sr, config.n_fft, config.hop_length, config.n_mels
                    )
                    t_a, z_a = _save_segments(
                        torch.from_numpy(spec).float().unsqueeze(0),
                        label_matrix,
                        segment_length_frames,
                        split_dir,
                        f"{split}_{int(idx)}_{aug_name}",
                    )
                    split_total_segments += t_a
                    split_zero_segments += z_a

        summary[split] = {
            "segments": split_total_segments,
            "zero_label_segments": split_zero_segments,
        }
        if validate_labels and split_total_segments > 0:
            ratio = split_zero_segments / max(1, split_total_segments)
            logger.info(
                f"[{split}] segments={split_total_segments} zero_label_segments={split_zero_segments} ({ratio:.2%})"
            )

    if validate_labels:
        total_segments = sum(v["segments"] for v in summary.values())
        total_zero = sum(v["zero_label_segments"] for v in summary.values())
        logger.info(
            f"[summary] segments={total_segments} zero_label_segments={total_zero} ({(total_zero / max(1, total_segments)):.2%})"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process E-GMD dataset.")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="local")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
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
        args.validate_labels,
    )
