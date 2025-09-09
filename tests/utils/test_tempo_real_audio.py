from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest
import soundfile as sf

from chart_hero.utils.tempo import estimate_tempo_map

ASSETS = pathlib.Path(__file__).resolve().parent.parent / "assets"


@pytest.mark.parametrize(
    "audio,bpms,times,tol",
    [
        ("tempo_2tempo_test_track.wav", [100, 150], [0.0, 3.37], 0.05),
        ("tempo_3tempo_test_track.wav", [90, 150, 80], [0.0, 4.0, 6.4], 0.6),
        (
            "tempo_4tempo_test_track.wav",
            [90, 150, 80, 140],
            [0.0, 4.0, 6.4, 10.9],
            0.6,
        ),
        (
            "tempo_5tempo_test_track.wav",
            [90, 150, 80, 140, 110],
            [0.0, 4.0, 6.4, 10.9, 13.47],
            0.7,
        ),
    ],
)
def test_estimate_tempo_map_with_real_audio(
    audio: str, bpms: list[float], times: list[float], tol: float
) -> None:
    path = ASSETS / audio
    assert path.exists(), f"missing test audio {path}"
    y, sr = sf.read(os.fspath(path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    segments, global_bpm, conf = estimate_tempo_map(y.astype(np.float32), sr)
    assert len(segments) >= len(bpms)
    print(segments)
    for seg, bpm in zip(segments, bpms):
        assert abs(seg.bpm - bpm) < 5.0
    for seg, time in zip(segments, times):
        assert abs(seg.time - time) < tol
    assert conf > 0
