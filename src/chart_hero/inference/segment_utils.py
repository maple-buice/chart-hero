from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .types import Segment


def frame_energies_from_segments(segments: Sequence[Segment]) -> np.ndarray:
    """Return concatenated mean energies for each frame across segments."""
    energies: list[np.ndarray] = []
    for seg in segments:
        spec = seg["spec"]
        if isinstance(spec, torch.Tensor):
            spec_arr = spec.detach().cpu().numpy()
        else:
            spec_arr = np.asarray(spec)
        if spec_arr.ndim != 2:
            continue
        if np.any(spec_arr < 0):
            spec_arr = np.power(10.0, spec_arr / 20.0)
        energies.append(spec_arr.mean(axis=0))
    if not energies:
        return np.array([], dtype=float)
    return np.concatenate(energies)


def detect_leading_silence_from_segments(
    segments: Sequence[Segment],
    threshold_db: float = -60.0,
    min_silence_frames: int = 0,
) -> int:
    """Return number of silent frames before the first audible frame.

    ``threshold_db`` is measured relative to the maximum frame energy.
    """
    energies = frame_energies_from_segments(segments)
    if energies.size == 0:
        return 0
    max_e = float(energies.max())
    if max_e <= 0:
        return 0
    thr = max_e * (10.0 ** (threshold_db / 20.0))
    idx = np.nonzero(energies >= thr)[0]
    if idx.size == 0:
        return int(energies.size)
    first = int(idx[0])
    if first < int(min_silence_frames):
        return 0
    return first
