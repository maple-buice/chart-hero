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
        ("tempo_3tempo_test_track.wav", [90, 150, 80], [0.0, 4.0, 6.4], 0.05),
        (
            "tempo_4tempo_test_track.wav",
            [90, 150, 80, 140],
            [0.0, 4.0, 6.4, 10.9],
            0.05,
        ),
        (
            "tempo_5tempo_test_track.wav",
            [90, 150, 80, 140, 110],
            [0.0, 4.0, 6.4, 10.9, 13.47],
            0.05,
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


@pytest.mark.xfail(reason="local tempo mapping currently exceeds 250ms tolerance")
def test_estimate_tempo_map_on_5tempo_audio() -> None:
    path = ASSETS / "tempo_5tempo_test_track.wav"
    assert path.exists(), f"missing test audio {path}"
    y, sr = sf.read(os.fspath(path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    segments, global_bpm, conf = estimate_tempo_map(y.astype(np.float32), sr)
    expected_bpms = [90, 150, 80, 140, 110]
    beats_per_segment = 6
    expected_times = [0.0]
    t = 0.0
    for bpm in expected_bpms[:-1]:
        t += beats_per_segment * 60.0 / bpm
        expected_times.append(t)
    assert len(segments) >= len(expected_bpms)
    for seg, bpm in zip(segments, expected_bpms):
        assert abs(seg.bpm - bpm) < 5.0
    for seg, time in zip(segments, expected_times):
        assert abs(seg.time - time) < 0.05
    assert conf > 0
