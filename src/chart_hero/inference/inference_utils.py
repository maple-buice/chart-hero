from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def map_patch_to_sample(
    start_frame: int,
    patch_idx: int,
    stride_frames: int,
    patch_size: int,
    hop_length: int,
    offset_samples: int,
    add_ms: float,
    sample_rate: int,
) -> int:
    """Convert a patch index into an absolute sample position."""
    frame = start_frame + patch_idx * stride_frames + patch_size // 2
    sample = frame * hop_length  # offset_samples removed to prevent double application
    if add_ms:
        sample += int(add_ms * sample_rate / 1000.0)
    return int(sample)


def estimate_global_shift(
    energy: np.ndarray,
    pred_frames: Sequence[int],
    hop_length: int,
    sample_rate: int,
    max_shift_ms: float = 200.0,
) -> float:
    """Estimate global time shift via cross-correlation (ms)."""
    if energy.size == 0 or not pred_frames:
        return 0.0
    N = energy.size
    pred = np.zeros(N, dtype=float)
    for f in pred_frames:
        if 0 <= f < N:
            pred[f] += 1.0
    energy_z = energy - energy.mean()
    pred_z = pred - pred.mean()
    corr = np.correlate(energy_z, pred_z, mode="full")
    lags = np.arange(-N + 1, N)
    max_shift_frames = int(round(max_shift_ms * sample_rate / (hop_length * 1000.0)))
    center = N - 1
    start = max(0, center - max_shift_frames)
    end = min(corr.size, center + max_shift_frames + 1)
    window = corr[start:end]
    lwindow = lags[start:end]
    best_lag = int(lwindow[np.argmax(window)])
    return best_lag * hop_length * 1000.0 / sample_rate
