#!/usr/bin/env python3
"""
Analyze timing offsets between predicted and ground-truth events for Clone Hero songs.

Runs inference on songs under provided roots (default CloneHero/KnownGoodSongs),
matches predictions to truth with a tolerance window, and reports per-class
statistics about timing offsets (in milliseconds). Optionally writes a CSV of all
matched offsets for deeper analysis.

Example:
  python scripts/analyze_offsets.py --model models/local_transformer_models/best_model.ckpt
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from chart_hero.eval.evaluate_chart import Event, load_truth_from_mid
from chart_hero.inference.charter import Charter
from chart_hero.inference.input_transform import audio_to_tensors
from chart_hero.model_training.transformer_config import get_config, get_drum_hits


def _discover_songs(roots: Sequence[str]) -> List[Tuple[Path, Path]]:
    """Return list of (midi, audio) pairs under provided roots."""
    pairs: List[Tuple[Path, Path]] = []
    audio_exts = {".ogg", ".mp3", ".wav", ".m4a"}
    for r in roots:
        base = Path(r)
        if not base.exists():
            continue
        for dp, _dn, fns in os.walk(base):
            if "notes.mid" not in fns:
                continue
            midi = Path(dp) / "notes.mid"
            audio = None
            if (Path(dp) / "song.ogg").exists():
                audio = Path(dp) / "song.ogg"
            else:
                for fn in fns:
                    if Path(fn).suffix.lower() in audio_exts:
                        audio = Path(dp) / fn
                        break
            if audio is not None:
                pairs.append((midi, audio))
    return pairs


def _df_to_events(df, sr: int) -> List[Event]:
    classes = get_drum_hits()
    ev: List[Event] = []
    for row in df.to_dict(orient="records"):
        sec = float(int(row["peak_sample"]) / float(sr))
        for cls in classes:
            if int(row.get(cls, 0)) == 1:
                ev.append(Event(t=sec, cls=cls))
    return ev


def _match_offsets(pred: List[Event], true: List[Event], tol_s: float) -> Dict[str, List[float]]:
    """Greedy match events and return per-class offsets (pred - true in sec)."""
    classes = get_drum_hits()
    offsets: Dict[str, List[float]] = {c: [] for c in classes}
    by_pred = {c: sorted([e.t for e in pred if e.cls == c]) for c in classes}
    by_true = {c: sorted([e.t for e in true if e.cls == c]) for c in classes}
    for c in classes:
        P = by_pred[c]
        T = by_true[c]
        i = j = 0
        while i < len(P) and j < len(T):
            dt = P[i] - T[j]
            if abs(dt) <= tol_s:
                offsets[c].append(dt)
                i += 1
                j += 1
            elif dt < 0:
                i += 1
            else:
                j += 1
    return offsets


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze prediction timing offsets")
    ap.add_argument(
        "--roots", type=str, nargs="+", default=["CloneHero/KnownGoodSongs"]
    )
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--config", type=str, default="local")
    ap.add_argument("--nms-k", type=int, default=9)
    ap.add_argument("--activity-gate", type=float, default=0.45)
    ap.add_argument("--cymbal-margin", type=float, default=0.30)
    ap.add_argument("--tom-over-cymbal-margin", type=float, default=0.35)
    ap.add_argument("--patch-stride", type=int, default=1)
    ap.add_argument("--tol-ms", type=float, default=45.0)
    ap.add_argument(
        "--out-csv", type=str, default=None, help="Optional CSV path to write offsets"
    )
    args = ap.parse_args()

    # Build config
    config = get_config(args.config)
    config.event_nms_kernel_patches = int(args.nms_k)
    config.activity_gate = float(args.activity_gate)
    config.cymbal_margin = float(args.cymbal_margin)
    config.tom_over_cymbal_margin = float(args.tom_over_cymbal_margin)
    try:
        config.patch_stride = int(args.patch_stride)
    except Exception:
        pass

    # Discover songs
    songs = _discover_songs(args.roots)
    if not songs:
        print("No songs with notes.mid + audio found under:", ", ".join(args.roots))
        return
    print(f"Analyzing {len(songs)} song(s)")

    # Load model once
    charter = Charter(config, args.model)
    sr = config.sample_rate
    classes = get_drum_hits()

    offsets_all: Dict[str, List[float]] = {c: [] for c in classes}

    for mid, aud in songs:
        segs = audio_to_tensors(str(aud), config)
        truth = load_truth_from_mid(mid)
        df = charter.predict(segs)
        preds = _df_to_events(df, sr)
        off = _match_offsets(preds, truth, float(args.tol_ms) / 1000.0)
        for c in classes:
            offsets_all[c].extend(off[c])

    # Optionally write CSV of offsets
    if args.out_csv:
        outp = Path(args.out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cls", "offset_ms"])
            for c, vals in offsets_all.items():
                for v in vals:
                    w.writerow([c, v * 1000.0])
        print(f"Wrote offsets to {outp}")

    # Print summary statistics
    print("\nPer-class offset statistics (ms):")
    for c in classes:
        arr = np.array([v * 1000.0 for v in offsets_all[c]], dtype=float)
        if arr.size == 0:
            print(f"  {c}: n=0")
            continue
        mean = float(arr.mean())
        med = float(np.median(arr))
        std = float(arr.std())
        print(f"  {c}: n={arr.size:4d} mean={mean:7.2f} med={med:7.2f} std={std:7.2f}")


if __name__ == "__main__":
    main()
