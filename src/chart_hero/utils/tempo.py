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

    The previous implementation relied on ``librosa.feature.tempo`` which
    operates on fixed–width frames.  While simple, frame based estimation
    caused the detected change points to drift by hundreds of milliseconds on
    real audio.  This version tracks the beats first and derives instantaneous
    tempo between consecutive beats.  Segment boundaries are then detected on
    the beat grid which greatly improves accuracy.

    Parameters
    ----------
    y : np.ndarray
        Audio time series.
    sr : int
        Sampling rate of ``y``.
    hop_length : int, optional
        Hop length for onset envelope and beat tracking.
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

    # Compute onset envelope and track beats.  ``trim=False`` keeps leading
    # beats so that the first detected beat is close to time 0.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length, trim=False
    )

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    if beat_times.size < 2:
        return [], float(tempo) if tempo else 0.0, 0.0

    # Instantaneous tempo for each interval between beats.
    bpms = 60.0 / np.diff(beat_times)
    bpms = medfilt(bpms, kernel_size=3)
    bpms = bpms.clip(30, 240)

    # Detect change points when BPM jumps by more than ``change_threshold``.
    change_points: List[int] = [0]
    last_bpm = bpms[0]
    for i in range(1, len(bpms)):
        if abs(bpms[i] - last_bpm) >= change_threshold:
            change_points.append(i)
            last_bpm = bpms[i]
    change_points.append(len(bpms))

    segments: List[TempoSegment] = []
    for start, end in zip(change_points[:-1], change_points[1:]):
        segment_bpms = bpms[start:end]
        if segment_bpms.size == 0:
            continue
        bpm = float(np.median(segment_bpms))
        # The segment begins at the first beat of this region.  Force the very
        # first segment to start at exactly 0.0 seconds for stability.
        t = 0.0 if start == 0 else float(beat_times[start])
        segments.append(TempoSegment(time=t, bpm=bpm))

    global_bpm = float(np.median(bpms))
    confidence = 1.0 / len(segments) if segments else 0.0
    return segments, global_bpm, confidence
