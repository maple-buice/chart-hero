#!/usr/bin/env python3
"""Evaluate tempo mapping accuracy against Clone Hero charts.

Given a root directory of Clone Hero songs, this script scans each song folder
for a ``notes.mid`` file and an accompanying audio stem (e.g., ``song.ogg``).
It extracts the ground truth tempo map from the MIDI, runs the repository's
``estimate_tempo_map`` routine on the audio, and reports simple accuracy metrics
for real-world use cases.

Usage
-----
    python scripts/eval_tempo_map.py /path/to/CloneHero/Songs --output results.json

The generated JSON contains, for each song, the chart tempo segments, the
audio-estimated segments, and per-segment BPM and boundary errors. This makes
it easy to compare different tempo-mapping implementations (e.g., PR #60 vs
PR #61) by running the script on each branch.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import librosa
import mido

from chart_hero.utils.tempo import TempoSegment, estimate_tempo_map


@dataclass
class ChartTempo:
    """Tempo segment extracted from the MIDI chart."""

    time: float  # start time in seconds
    bpm: float


def _collect_tempo_changes(mid: mido.MidiFile) -> List[Tuple[int, int]]:
    """Return sorted list of (tick, microseconds per beat) tempo events."""
    changes: dict[int, int] = {}
    for tr in mid.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.is_meta and msg.type == "set_tempo" and t not in changes:
                changes[t] = int(msg.tempo)
    if 0 not in changes:
        changes[0] = 500000  # default 120 BPM
    return sorted(changes.items(), key=lambda x: x[0])


def _tempo_segments(mid_path: str) -> List[ChartTempo]:
    """Extract tempo segments from a notes.mid file as absolute times."""
    mid = mido.MidiFile(mid_path)
    tpq = mid.ticks_per_beat or 480
    changes = _collect_tempo_changes(mid)
    segments: List[ChartTempo] = []
    sec = 0.0
    last_tick, last_tempo = changes[0]
    segments.append(ChartTempo(time=0.0, bpm=mido.tempo2bpm(last_tempo)))
    for tick, tempo in changes[1:]:
        sec += (tick - last_tick) * (last_tempo / 1_000_000.0) / tpq
        segments.append(ChartTempo(time=sec, bpm=mido.tempo2bpm(tempo)))
        last_tick, last_tempo = tick, tempo
    return segments


def _find_audio(song_dir: str) -> Optional[str]:
    """Heuristic to locate an audio stem for a Clone Hero song."""
    candidates = [
        "song.ogg",
        "guitar.ogg",
        "audio.ogg",
        "song.mp3",
        "guitar.mp3",
    ]
    for name in candidates:
        p = os.path.join(song_dir, name)
        if os.path.exists(p):
            return p
    return None


def _bpm_at(time_sec: float, segments: Iterable[TempoSegment]) -> Optional[float]:
    """Return the BPM from segments active at the given time."""
    last: Optional[TempoSegment] = None
    for seg in segments:
        if seg.time <= time_sec:
            last = seg
        else:
            break
    return last.bpm if last else None


def evaluate_song(song_dir: str) -> Optional[dict]:
    midi_path = os.path.join(song_dir, "notes.mid")
    audio_path = _find_audio(song_dir)
    if not os.path.exists(midi_path) or audio_path is None:
        return None

    chart_segs = _tempo_segments(midi_path)
    y, sr = librosa.load(audio_path, sr=None)
    est_segs, global_bpm, conf = estimate_tempo_map(y, sr)

    bpm_errs: List[float] = []
    boundary_errs: List[float] = []

    # Compare BPM at each chart segment start
    for idx, cseg in enumerate(chart_segs):
        est_bpm = _bpm_at(cseg.time, est_segs)
        if est_bpm is not None:
            bpm_errs.append(abs(est_bpm - cseg.bpm))
        # Compare boundary times where both have a subsequent segment
        if idx + 1 < len(chart_segs) and idx + 1 < len(est_segs):
            boundary_errs.append(abs(chart_segs[idx + 1].time - est_segs[idx + 1].time))

    return {
        "song": os.path.basename(song_dir),
        "chart_segments": [(s.time, s.bpm) for s in chart_segs],
        "estimated_segments": [(s.time, s.bpm) for s in est_segs],
        "avg_bpm_error": sum(bpm_errs) / len(bpm_errs) if bpm_errs else None,
        "avg_boundary_error": (
            sum(boundary_errs) / len(boundary_errs) if boundary_errs else None
        ),
        "global_bpm": global_bpm,
        "confidence": conf,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tempo mapping accuracy")
    parser.add_argument(
        "root", help="Root directory containing Clone Hero song folders"
    )
    parser.add_argument("--output", help="Optional JSON file to save detailed results")
    args = parser.parse_args()

    results = []
    for entry in os.scandir(args.root):
        if entry.is_dir():
            res = evaluate_song(entry.path)
            if res:
                results.append(res)
                bpm_err = (
                    f"{res['avg_bpm_error']:.2f}"
                    if res["avg_bpm_error"] is not None
                    else "N/A"
                )
                boundary_err = (
                    f"{res['avg_boundary_error']:.3f}s"
                    if res["avg_boundary_error"] is not None
                    else "N/A"
                )
                print(f"{res['song']}: BPM err={bpm_err} Boundary err={boundary_err}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
