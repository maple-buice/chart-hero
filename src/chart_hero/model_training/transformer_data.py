"""
Data preparation and loading for transformer-based drum transcription.
Handles patch-based spectrogram processing and efficient data loading.
"""

import logging
import math
import os
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from .augment_spec import (
    augment_spectrogram_frequency_masking,
    augment_spectrogram_time_masking,
)
from .data_utils import collate_with_lengths, custom_collate_fn
from .transformer_config import BaseConfig

logger = logging.getLogger(__name__)


class SpectrogramProcessor:
    """Handles audio to spectrogram conversion with patch-based tokenization."""

    def __init__(self, config: BaseConfig):
        self.config = config
        # Lazy import torchaudio so precomputed-numpy pipelines don't require it
        try:
            import torchaudio  # type: ignore

            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                n_mels=config.n_mels,
                power=2.0,
            )
        except (
            Exception
        ) as e:  # pragma: no cover - only hit in environments without torchaudio
            raise RuntimeError(
                "torchaudio is required only when generating spectrograms from audio. "
                "For precomputed .npy spectrograms, this class is unused."
            ) from e

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


class NpyDrumDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
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

        # Load data via numpy memmap for lower I/O overhead
        spectrogram_np = np.load(spec_file, allow_pickle=False, mmap_mode="r")
        label_np = np.load(label_file, allow_pickle=False, mmap_mode="r")
        spectrogram = torch.from_numpy(
            np.array(spectrogram_np, dtype=np.float32, copy=self.mode == "train")
        )
        label_matrix = torch.from_numpy(
            np.array(label_np, dtype=np.float32, copy=self.mode == "train")
        )

        # Robustness: ensure labels have shape (T, C) with T>0 and correct C
        num_classes = self.config.num_drum_classes
        if (
            label_matrix.ndim != 2
            or label_matrix.shape[-1] != num_classes
            or label_matrix.shape[0] == 0
        ):
            T = int(spectrogram.shape[-1]) if spectrogram.ndim >= 3 else 0
            if T <= 0:
                # Fallback minimal length if spectrogram is also unexpected; avoid zero-length
                T = int(getattr(self.config, "max_seq_len", 1))
            label_matrix = torch.zeros((T, num_classes), dtype=torch.float32)

        # Optional windowing to enforce maximum sequence length
        max_seq_len = int(getattr(self.config, "max_seq_len", 0) or 0)
        max_audio_frames = int(
            round(
                (getattr(self.config, "max_audio_length", 0.0) or 0.0)
                * self.config.sample_rate
                / max(1, int(self.config.hop_length))
            )
        )
        seg_frames = max(max_seq_len, max_audio_frames)

        if seg_frames > 0 and spectrogram.dim() >= 2:
            T = int(spectrogram.shape[-1])
            if T > seg_frames:
                if self.mode == "train":
                    start = int(torch.randint(0, T - seg_frames + 1, (1,)).item())
                else:
                    # Center crop for val/test
                    start = max(0, (T - seg_frames) // 2)
                end = start + seg_frames
                spectrogram = spectrogram[..., start:end]
                label_matrix = label_matrix[start:end, :]

        # Ensure tensor is (1, freq, time)
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)

        # Normalize per-sample to stabilize training (z-score across freq x time)
        if getattr(self.config, "normalize_spectrograms", False):
            # shape (1, F, T)
            mean = spectrogram.mean(dim=(1, 2), keepdim=True)
            std = spectrogram.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
            spectrogram = (spectrogram - mean) / std

        # Apply SpecAugment only during training
        if self.mode == "train" and self.config.enable_spec_augmentation:
            spec = spectrogram.squeeze(0)
            spec = augment_spectrogram_time_masking(
                spec,
                num_masks=self.config.spec_aug_num_time_masks,
                max_mask_percentage=self.config.spec_aug_max_time_mask_percentage,
            )
            spec = augment_spectrogram_frequency_masking(
                spec,
                num_masks=self.config.spec_aug_num_freq_masks,
                max_mask_percentage=self.config.spec_aug_max_freq_mask_percentage,
            )
            spectrogram = spec.unsqueeze(0)

        # Optional time-shift augmentation (circular) on training only
        if self.mode == "train" and getattr(
            self.config, "enable_time_shift_augmentation", False
        ):
            if torch.rand(1).item() < getattr(self.config, "time_shift_prob", 0.0):
                max_pct = getattr(self.config, "time_shift_max_percentage", 0.1)
                T = spectrogram.shape[-1]
                max_shift = max(1, int(max_pct * T))
                shift = int(
                    torch.randint(low=-max_shift, high=max_shift + 1, size=(1,)).item()
                )
                if shift != 0:
                    # Roll spectrogram along time dimension
                    spectrogram = torch.roll(spectrogram, shifts=shift, dims=-1)
                    # Roll labels along time dimension to keep alignment (labels: [T, C])
                    label_matrix = torch.roll(label_matrix, shifts=shift, dims=0)

        # The model's PatchEmbedding expects (Batch, Channels, Freq, Time),
        # so we do NOT transpose here. The loaded npy is already (Freq, Time).
        return spectrogram, label_matrix


class BucketBatchSampler(Sampler[List[int]]):
    """Group samples of similar lengths to minimize padding overhead.

    Sorts within medium-sized buckets to preserve randomness while reducing
    variance in sequence lengths per batch.
    """

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        shuffle: bool = True,
        bucket_size_mult: int = 50,
        drop_last: bool = False,
    ) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = max(batch_size * bucket_size_mult, batch_size)
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        import numpy as _np

        n = len(self.lengths)
        indices = _np.arange(n)
        if self.shuffle:
            _np.random.shuffle(indices)
        # Form buckets
        for start in range(0, n, self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            # Sort bucket by length (descending) to reduce padding
            bucket_sorted = bucket[_np.argsort([self.lengths[i] for i in bucket])[::-1]]
            # Yield batches
            for i in range(0, len(bucket_sorted), self.batch_size):
                batch = bucket_sorted[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield [int(x) for x in batch]

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)


def create_data_loaders(
    config: BaseConfig,
    data_dir: str,
    batch_size: Optional[int] = None,
    with_lengths: bool = False,
) -> Tuple[
    DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    DataLoader[Tuple[torch.Tensor, torch.Tensor]],
]:
    """Create data loaders for train, validation, and test sets from .npy files."""

    if batch_size is None:
        batch_size = config.train_batch_size

    data_path = Path(data_dir)
    data_files: dict[str, list[tuple[str, str]]] = {"train": [], "val": [], "test": []}

    for split in ["train", "val", "test"]:
        split_dir = data_path / split
        if not split_dir.exists():
            continue
        for file in sorted(split_dir.glob("*_mel.npy")):
            label_file = file.parent / file.name.replace("_mel.npy", "_label.npy")
            if label_file.exists():
                data_files[split].append((str(file), str(label_file)))

    if not data_files["train"]:
        raise FileNotFoundError(f"No training data found in {data_dir}")

    # Optional subsampling per split for faster iteration
    def _subset_pairs(
        pairs: List[Tuple[str, str]], split: str
    ) -> List[Tuple[str, str]]:
        if not pairs:
            return pairs
        frac = float(getattr(config, "dataset_fraction", 1.0) or 1.0)
        cap = getattr(config, "max_files_per_split", None)
        n = len(pairs)
        keep = n
        if 0.0 < frac < 1.0:
            keep = max(1, int(round(n * frac)))
        if isinstance(cap, int) and cap > 0:
            keep = min(keep, cap)
        if keep < n:
            rng = np.random.default_rng(seed=getattr(config, "seed", 42))
            idx = np.sort(rng.choice(n, size=keep, replace=False))
            pairs = [pairs[int(i)] for i in idx]
            logger.info("Subsampled %s split: %d/%d files", split, keep, n)
        return pairs

    data_files["train"] = _subset_pairs(data_files["train"], "train")
    data_files["val"] = _subset_pairs(data_files["val"], "val")
    data_files["test"] = _subset_pairs(data_files["test"], "test")

    # Create datasets
    train_dataset = NpyDrumDataset(data_files["train"], config, mode="train")
    val_dataset = NpyDrumDataset(data_files["val"], config, mode="val")
    test_dataset = NpyDrumDataset(data_files["test"], config, mode="test")

    # Efficient: estimate T from file size without opening arrays: T = bytes / (C * 4)
    def _file_lengths_from_stat(
        pairs: List[Tuple[str, str]], num_classes: int
    ) -> List[int]:
        lengths: List[int] = []
        bytes_per_timestep = num_classes * 4  # float32 per class
        for _, lbl in pairs:
            size = os.stat(lbl).st_size
            t = max(1, int(size // bytes_per_timestep))
            lengths.append(t)
        return lengths

    # Cache lengths to avoid re-computing across runs. Use model_dir to avoid read-only data_dir.
    cache_root = Path(getattr(config, "model_dir", "."))
    cache_dir = cache_root / ".length_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        cache_dir = None

    def _cache_paths(split: str) -> tuple[Path, Path]:
        return (
            cache_dir / f"lengths_{split}.npy",
            cache_dir / f"lengths_{split}.meta.json",
        )

    import json as _json

    def _load_or_compute_lengths(split: str, pairs: List[Tuple[str, str]]) -> List[int]:
        if cache_dir is not None:
            npy_path, meta_path = _cache_paths(split)
            try:
                if npy_path.exists() and meta_path.exists():
                    meta = _json.loads(meta_path.read_text())
                    if meta.get("num_files") == len(pairs):
                        arr = np.load(npy_path)
                        if arr.ndim == 1 and arr.shape[0] == len(pairs):
                            logger.info(
                                f"Loaded cached lengths for split '{split}' from {npy_path}"
                            )
                            return arr.astype(int).tolist()
            except Exception as e:
                logger.warning(
                    f"Failed to load cached lengths for split '{split}': {e}"
                )

        lengths = (
            _file_lengths_from_stat(pairs, config.num_drum_classes) if pairs else []
        )
        if cache_dir is not None:
            try:
                npy_path, meta_path = _cache_paths(split)
                np.save(npy_path, np.array(lengths, dtype=np.int32))
                meta = {"num_files": len(pairs)}
                meta_path.write_text(_json.dumps(meta))
                logger.info(f"Saved lengths cache for split '{split}' to {npy_path}")
            except Exception as e:
                logger.warning(f"Failed to save lengths cache for split '{split}': {e}")
        return lengths

    train_lengths = (
        _load_or_compute_lengths("train", data_files["train"])
        if data_files["train"]
        else []
    )
    val_lengths = (
        _load_or_compute_lengths("val", data_files["val"]) if data_files["val"] else []
    )
    test_lengths = (
        _load_or_compute_lengths("test", data_files["test"])
        if data_files["test"]
        else []
    )

    # Auto-tune workers/prefetch for better CPU utilization
    cpu_count = os.cpu_count() or 4
    tuned_num_workers = config.num_workers
    if tuned_num_workers <= 2 and cpu_count >= 4:
        tuned_num_workers = min(8, max(2, cpu_count - 1))
    tuned_prefetch = getattr(config, "prefetch_factor", 2)
    if tuned_num_workers > 0:
        tuned_prefetch = max(2, min(8, tuned_prefetch if tuned_prefetch else 2))

    # Create data loaders
    collate_fn = collate_with_lengths if with_lengths else custom_collate_fn

    # Do not drop the last partial batch; on small datasets or heavy subsampling,
    # dropping can yield zero train batches, which prevents any checkpoints.
    train_sampler = BucketBatchSampler(
        train_lengths,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        bucket_size_mult=50,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=tuned_num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
        persistent_workers=getattr(config, "persistent_workers", False)
        and tuned_num_workers > 0,
        prefetch_factor=(tuned_prefetch if tuned_num_workers > 0 else None),
    )

    val_sampler = BucketBatchSampler(
        val_lengths,
        batch_size=config.val_batch_size,
        shuffle=False,
        drop_last=False,
        bucket_size_mult=50,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=tuned_num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
        persistent_workers=getattr(config, "persistent_workers", False)
        and tuned_num_workers > 0,
        prefetch_factor=(tuned_prefetch if tuned_num_workers > 0 else None),
    )

    test_sampler = BucketBatchSampler(
        test_lengths,
        batch_size=config.val_batch_size,
        shuffle=False,
        drop_last=False,
        bucket_size_mult=50,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        num_workers=tuned_num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
        persistent_workers=getattr(config, "persistent_workers", False)
        and tuned_num_workers > 0,
        prefetch_factor=(tuned_prefetch if tuned_num_workers > 0 else None),
    )

    return train_loader, val_loader, test_loader


def compute_class_pos_weights(
    data_dir: str,
    num_classes: int,
    split: str = "train",
    max_files: Optional[int] = None,
    cap: float | None = None,
) -> torch.Tensor:
    """Compute BCEWithLogits pos_weight per class from label .npy files.

    pos_weight = (N - P) / (P + eps) computed per class over all frames.
    """
    path = Path(data_dir) / split
    # Accumulate in NumPy to avoid PyTorch warnings with read-only memmaps
    pos_np = np.zeros((num_classes,), dtype=np.float64)
    total_frames = 0
    eps = 1e-6
    if not path.exists():
        return torch.ones(num_classes, dtype=torch.float32)

    label_files = list(path.glob("*_label.npy"))
    if max_files is not None and len(label_files) > max_files:
        # Random but deterministic subset selection for reproducibility
        rng = np.random.default_rng(seed=12345)
        label_files = list(rng.choice(label_files, size=max_files, replace=False))

    for label_file in label_files:
        arr = np.load(label_file, allow_pickle=False, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[1] != num_classes:
            continue
        # Sum positives per class in float64 for numerical stability
        pos_np += np.sum(arr, axis=0, dtype=np.float64)
        total_frames += int(arr.shape[0])

    neg_np = np.maximum(0.0, total_frames - pos_np)
    pw_np = (neg_np / (pos_np + eps)).astype(np.float32)
    # Bound pos_weight to reasonable range to avoid extreme gradients
    upper = float(cap) if (cap is not None and cap > 0) else 50.0
    pw_np = np.clip(pw_np, 0.5, upper)
    return torch.from_numpy(pw_np)
