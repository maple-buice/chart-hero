"""
Data preparation and loading for transformer-based drum transcription.
Handles patch-based spectrogram processing and efficient data loading.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

from .augment_audio import (
    augment_spectrogram_frequency_masking,
    augment_spectrogram_time_masking,
)
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

        # Convert to mel spectrogram
        mel_spec = self.mel_transform(audio)

        # Convert to log scale
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-8))

        # Transpose to [channels, time, freq]
        if log_mel_spec.shape[1] == self.config.n_mels:
            log_mel_spec = log_mel_spec.transpose(1, 2)

        # Normalize
        min_val, max_val = log_mel_spec.min(), log_mel_spec.max()
        if (max_val - min_val) > 1e-8:
            log_mel_spec = 2 * (log_mel_spec - min_val) / (max_val - min_val) - 1
        else:
            log_mel_spec = torch.zeros_like(log_mel_spec)

        return torch.nan_to_num(log_mel_spec)

    def prepare_patches(
        self, spectrogram: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Prepare spectrogram for patch-based processing.
        """
        channels, time_frames, freq_bins = spectrogram.shape
        patch_time, patch_freq = self.config.patch_size

        time_padding = (patch_time - (time_frames % patch_time)) % patch_time
        freq_padding = (patch_freq - (freq_bins % patch_freq)) % patch_freq

        if time_padding > 0 or freq_padding > 0:
            spectrogram = F.pad(spectrogram, (0, freq_padding, 0, time_padding))

        final_time_frames = time_frames + time_padding
        final_freq_bins = freq_bins + freq_padding

        time_patches = final_time_frames // patch_time
        freq_patches = final_freq_bins // patch_freq

        return spectrogram, (time_patches, freq_patches)


class NpyDrumDataset(Dataset):
    """Dataset for loading pre-computed spectrogram and label segments."""

    def __init__(
        self,
        data_files: List[Tuple[str, str]],
        config: BaseConfig,
        mode: str = "train",
    ):
        self.data_files = data_files
        self.config = config
        self.mode = mode
        logger.info(f"Created {mode} dataset with {len(self.data_files)} files.")

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        spec_file, label_file = self.data_files[idx]

        # Load data and ensure correct types
        spectrogram = torch.from_numpy(np.load(spec_file)).float()
        label_matrix = torch.from_numpy(np.load(label_file)).float()

        # Apply SpecAugment only during training
        if self.mode == "train" and self.config.enable_spec_augmentation:
            spec_np = spectrogram.clone().squeeze(0).numpy()
            spec_np = augment_spectrogram_time_masking(
                spec_np,
                num_masks=self.config.spec_aug_num_time_masks,
                max_mask_percentage=self.config.spec_aug_max_time_mask_percentage,
            )
            spec_np = augment_spectrogram_frequency_masking(
                spec_np,
                num_masks=self.config.spec_aug_num_freq_masks,
                max_mask_percentage=self.config.spec_aug_max_freq_mask_percentage,
            )
            spectrogram = torch.from_numpy(spec_np).unsqueeze(0)

        # Ensure tensor is (1, freq, time) and then transpose to (1, time, freq) for the model
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)

        return spectrogram.transpose(1, 2), label_matrix


def create_data_loaders(
    config: BaseConfig, data_dir: str, batch_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for train, validation, and test sets from .npy files."""

    if batch_size is None:
        batch_size = config.train_batch_size

    data_path = Path(data_dir)
    data_files = {"train": [], "val": [], "test": []}

    for split in ["train", "val", "test"]:
        split_dir = data_path / split
        if not split_dir.exists():
            continue
        for file in split_dir.glob("*_mel.npy"):
            label_file = file.parent / file.name.replace("_mel.npy", "_label.npy")
            if label_file.exists():
                data_files[split].append((str(file), str(label_file)))

    if not data_files["train"]:
        raise FileNotFoundError(f"No training data found in {data_dir}")

    # Create datasets
    train_dataset = NpyDrumDataset(data_files["train"], config, mode="train")
    val_dataset = NpyDrumDataset(data_files["val"], config, mode="val")
    test_dataset = NpyDrumDataset(data_files["test"], config, mode="test")

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
