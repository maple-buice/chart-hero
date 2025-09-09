from __future__ import annotations

from typing import Any


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
    sample = offset_samples + frame * hop_length
    if add_ms:
        sample += int(add_ms * sample_rate / 1000.0)
    return int(sample)
