"""
Data preparation and loading for transformer-based drum transcription.
Handles patch-based spectrogram processing and efficient data loading.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

from .transformer_config import BaseConfig

logger = logging.getLogger(__name__)


class SpectrogramProcessor:
    """Handles audio to spectrogram conversion with patch-based tokenization."""

    def __init__(self, config: BaseConfig):
        self.config = config
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            power=2.0,
        )

    def audio_to_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio waveform to log-mel spectrogram."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Ensure audio is the right length
        target_length = int(self.config.max_audio_length * self.config.sample_rate)

        if audio.shape[-1] > target_length:
            # Random crop for training data augmentation
            start_idx = torch.randint(
                0, audio.shape[-1] - target_length + 1, (1,)
            ).item()
            audio = audio[..., start_idx : start_idx + target_length]
        elif audio.shape[-1] < target_length:
            # Pad with zeros
            padding = target_length - audio.shape[-1]
            audio = F.pad(audio, (0, padding))

        # Ensure exactly the target length
        audio = audio[..., :target_length]

        # Convert to mel spectrogram
        mel_spec = self.mel_transform(audio)

        # Convert to log scale
        # log_mel_spec = torch.log(mel_spec + 1e-8) # Original
        log_mel_spec = torch.log(
            torch.clamp(mel_spec, min=1e-8)
        )  # Clamp to avoid log(0)

        # Ensure consistent shape: [channels, freq, time] -> [channels, time, freq]
        # MelSpectrogram outputs [channels, n_mels, time_frames]
        # We want [channels, time_frames, n_mels] for consistency
        if log_mel_spec.shape[1] == self.config.n_mels:
            # Current shape is [channels, n_mels, time_frames], transpose to [channels, time_frames, n_mels]
            log_mel_spec = log_mel_spec.transpose(1, 2)

        # Normalize to [-1, 1] range
        # log_mel_spec = 2 * (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min()) - 1 # Original
        min_val = log_mel_spec.min()
        max_val = log_mel_spec.max()
        if (
            max_val - min_val
        ) > 1e-8:  # Avoid division by zero if all values are the same
            log_mel_spec = 2 * (log_mel_spec - min_val) / (max_val - min_val) - 1
        else:
            log_mel_spec = torch.zeros_like(
                log_mel_spec
            )  # Set to zero if range is too small

        # Check for NaNs or Infs after normalization and replace them
        if torch.isnan(log_mel_spec).any() or torch.isinf(log_mel_spec).any():
            logger.warning(
                "NaN or Inf detected in spectrogram after normalization. Replacing with zeros."
            )
            log_mel_spec = torch.nan_to_num(
                log_mel_spec, nan=0.0, posinf=0.0, neginf=0.0
            )

        return log_mel_spec

    def prepare_patches(
        self, spectrogram: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Prepare spectrogram for patch-based processing.

        Args:
            spectrogram: Log-mel spectrogram [channels, time, freq]

        Returns:
            Padded spectrogram and patch shape (time_patches, freq_patches)
        """
        channels, time_frames, freq_bins = spectrogram.shape
        patch_time, patch_freq = self.config.patch_size

        # Calculate required padding
        time_padding = (patch_time - (time_frames % patch_time)) % patch_time
        freq_padding = (patch_freq - (freq_bins % patch_freq)) % patch_freq

        # Pad spectrogram - padding order is (left, right, top, bottom) for last two dims
        # For tensor [channels, time, freq], we pad freq (last dim) then time (second-to-last)
        if time_padding > 0 or freq_padding > 0:
            spectrogram = F.pad(spectrogram, (0, freq_padding, 0, time_padding))

        # Calculate patch dimensions after padding
        final_time_frames = time_frames + time_padding
        final_freq_bins = freq_bins + freq_padding

        time_patches = final_time_frames // patch_time
        freq_patches = final_freq_bins // patch_freq

        return spectrogram, (time_patches, freq_patches)


class NpyDrumDataset(Dataset):
    """Dataset for loading pre-computed full-length spectrograms and labels."""

    def __init__(
        self,
        data_files: List[Tuple[str, str]],
        config: BaseConfig,
        mode: str = "train",
        augment: bool = True,
    ):
        self.data_files = data_files
        self.config = config
        self.mode = mode
        self.augment = augment and (mode == "train")
        self.segment_length_frames = int(
            self.config.max_audio_length
            * self.config.sample_rate
            / self.config.hop_length
        )

        logger.info(f"Created {mode} dataset with {len(self.data_files)} files.")

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        spec_file, label_file = self.data_files[idx]
        spectrogram = torch.from_numpy(np.load(spec_file))
        label_matrix = torch.from_numpy(np.load(label_file))

        # Get a random segment
        num_frames = spectrogram.shape[1]
        if num_frames > self.segment_length_frames:
            start_frame = torch.randint(
                0, num_frames - self.segment_length_frames, (1,)
            ).item()
            end_frame = start_frame + self.segment_length_frames
            spectrogram_segment = spectrogram[:, start_frame:end_frame, :]
            label_matrix_segment = label_matrix[start_frame:end_frame, :]
        else:
            # Pad if the spectrogram is shorter than the segment length
            padding = self.segment_length_frames - num_frames
            spectrogram_segment = F.pad(spectrogram, (0, 0, 0, padding))
            label_matrix_segment = F.pad(label_matrix, (0, 0, 0, padding))

        return {
            "spectrogram": spectrogram_segment,
            "labels": label_matrix_segment,
        }


def create_data_loaders(
    config: BaseConfig, data_dir: str, batch_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for train, validation, and test sets from .npy files."""

    if batch_size is None:
        batch_size = config.train_batch_size

    # Find all .npy files and group them by mode (train, val, test)
    data_files = {"train": [], "val": [], "test": []}

    for file in Path(data_dir).glob("*_mel.npy"):
        label_file = file.parent / file.name.replace("_mel.npy", "_label.npy")
        if label_file.exists():
            if "train" in file.name:
                data_files["train"].append((str(file), str(label_file)))
            elif "val" in file.name:
                data_files["val"].append((str(file), str(label_file)))
            elif "test" in file.name:
                data_files["test"].append((str(file), str(label_file)))

    if not data_files["train"]:
        raise FileNotFoundError(f"No training data found in {data_dir}")

    # Create datasets
    train_dataset = NpyDrumDataset(
        data_files["train"], config, mode="train", augment=True
    )
    val_dataset = NpyDrumDataset(data_files["val"], config, mode="val", augment=False)
    test_dataset = NpyDrumDataset(
        data_files["test"], config, mode="test", augment=False
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=getattr(config, "persistent_workers", False)
        and config.num_workers > 0,
        prefetch_factor=(
            getattr(config, "prefetch_factor", 2) if config.num_workers > 0 else None
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=getattr(config, "persistent_workers", False)
        and config.num_workers > 0,
        prefetch_factor=(
            getattr(config, "prefetch_factor", 2) if config.num_workers > 0 else None
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=getattr(config, "persistent_workers", False)
        and config.num_workers > 0,
        prefetch_factor=(
            getattr(config, "prefetch_factor", 2) if config.num_workers > 0 else None
        ),
    )

    return train_loader, val_loader, test_loader
