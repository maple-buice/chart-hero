from __future__ import annotations

import librosa
import numpy as np

from chart_hero.utils.tempo import estimate_tempo_map


def _click_track(
    bpms: list[float], durations: list[float], sr: int = 22050
) -> np.ndarray:
    """Generate a concatenated click track for the given BPM segments."""
    assert len(bpms) == len(durations)
    pieces = []
    for bpm, dur in zip(bpms, durations):
        clicks = librosa.clicks(
            times=np.arange(0, dur, 60.0 / bpm), sr=sr, length=int(dur * sr)
        )
        pieces.append(clicks)
    return np.concatenate(pieces)


def test_estimate_tempo_map_detects_changes() -> None:
    sr = 22050
    y = _click_track([100.0, 150.0], [4.0, 4.0], sr)
    segments, global_bpm, conf = estimate_tempo_map(y, sr)
    assert len(segments) >= 2
    assert abs(segments[0].bpm - 100.0) < 5.0
    assert abs(segments[1].bpm - 150.0) < 5.0
    assert conf > 0
