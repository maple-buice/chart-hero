from __future__ import annotations

from typing import Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray


class Segment(TypedDict):
    spec: NDArray[np.floating]
    start_frame: int
    end_frame: int
    total_frames: int


PredictionRow = dict[str, int]


class TransformerConfig(Protocol):
    sample_rate: int
    n_mels: int
    n_fft: int
    hop_length: int
    max_audio_length: float
    patch_size: tuple[int, int]
    prediction_threshold: float
    class_thresholds: list[float] | None
    device: str
    inference_batch_size: int
    leading_silence_db: float
    leading_silence_min_ms: float
    global_shift_max_ms: float
