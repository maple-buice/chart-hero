from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import librosa
import numpy as np
from numpy.typing import NDArray
from scipy.signal import medfilt


@dataclass
class TempoSegment:
    """Represents a tempo segment starting at a given time."""

    time: float
    bpm: float


def estimate_tempo_map(
    y: NDArray[np.floating],
    sr: int,
    hop_length: int = 512,
    change_threshold: float = 10.0,
) -> Tuple[List[TempoSegment], float, float]:
    """Estimate a tempo curve and detect tempo change points.

    Parameters
    ----------
    y : np.ndarray
        Audio time series.
    sr : int
        Sampling rate of ``y``.
    hop_length : int, optional
        Hop length for tempo estimation frames.
    change_threshold : float, optional
        Minimum BPM difference between successive segments to trigger a new
        tempo segment.

    Returns
    -------
    segments : list[TempoSegment]
        Detected tempo segments ordered by time.
    global_bpm : float
        Median BPM across the entire tempo curve.
    confidence : float
        Simple confidence score ``1 / len(segments)`` (0 if no segments).
    """

    tempo_curve = librosa.feature.tempo(
        y=y, sr=sr, hop_length=hop_length, aggregate=None
    )
    if tempo_curve is None or tempo_curve.size == 0:
        return [], 0.0, 0.0

    # Smooth the tempo curve to reduce jitter
    tempo_curve = medfilt(tempo_curve, kernel_size=5)
    tempo_curve = tempo_curve.clip(30, 240)

    # Simple change detection based on BPM differences
    change_points: List[int] = []
    last_bpm = tempo_curve[0]
    for i in range(1, len(tempo_curve)):
        if abs(tempo_curve[i] - last_bpm) > change_threshold:
            change_points.append(i)
            last_bpm = tempo_curve[i]
    change_points.append(len(tempo_curve))

    segments: List[TempoSegment] = []
    start = 0
    for end in change_points:
        segment = tempo_curve[start:end]
        if segment.size == 0:
            continue
        bpm = float(np.median(segment))
        t = float(librosa.frames_to_time(start, sr=sr, hop_length=hop_length))
        segments.append(TempoSegment(time=t, bpm=bpm))
        start = end

    global_bpm = float(np.median(tempo_curve))
    confidence = 1.0 / len(segments) if segments else 0.0
    return segments, global_bpm, confidence
