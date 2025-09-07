#!/usr/bin/env python3
"""
Calibrate per-class thresholds on a small dev set of Clone Hero songs.

Finds <notes.mid> + audio under one or more roots (default CloneHero/KnownGoodSongs),
runs inference repeatedly while sweeping each class threshold, and chooses the
value that maximizes aggregate F1 for that class. Writes class_thresholds.json
next to the provided checkpoint by default.

Example:
  python scripts/calibrate_thresholds.py \
    --roots CloneHero/KnownGoodSongs \
    --model models/local_transformer_models/best_model.ckpt \
    --grid 0.5,0.55,0.6,0.65,0.7,0.75 \
    --nms-k 9 --activity-gate 0.45 --patch-stride 1 --tol-ms 45
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from chart_hero.eval.evaluate_chart import Event, load_truth_from_mid, match_events
from chart_hero.inference.charter import Charter
from chart_hero.inference.input_transform import audio_to_tensors
from chart_hero.model_training.transformer_config import get_config, get_drum_hits


def _discover_songs(roots: Sequence[str]) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    audio_exts = {".ogg", ".mp3", ".wav", ".m4a"}
    for r in roots:
        base = Path(r)
        if not base.exists():
            continue
        for dp, _dn, fns in os.walk(base):
            if "notes.mid" in fns:
                midi = Path(dp) / "notes.mid"
                audio = None
                # prefer song.ogg else first audio file
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Calibrate per-class thresholds on dev set"
    )
    ap.add_argument(
        "--roots", type=str, nargs="+", default=["CloneHero/KnownGoodSongs"]
    )
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--config", type=str, default="local")
    ap.add_argument("--grid", type=str, default="0.5,0.55,0.6,0.65,0.7,0.75")
    ap.add_argument("--nms-k", type=int, default=9)
    ap.add_argument("--activity-gate", type=float, default=0.45)
    ap.add_argument("--cymbal-margin", type=float, default=0.30)
    ap.add_argument("--tom-over-cymbal-margin", type=float, default=0.35)
    ap.add_argument("--patch-stride", type=int, default=1)
    ap.add_argument("--tol-ms", type=float, default=45.0)
    ap.add_argument(
        "--out", type=str, default=None, help="Output class_thresholds.json path"
    )
    args = ap.parse_args()

    # Parse grid
    grid_vals = []
    for s in (x.strip() for x in args.grid.split(",")):
        try:
            grid_vals.append(float(s))
        except Exception:
            pass
    grid = sorted({v for v in grid_vals if 0.05 <= v <= 0.99}) or [0.5, 0.6, 0.7, 0.8]

    # Build config (balanced defaults, no preset)
    config = get_config(args.config)
    config.event_nms_kernel_patches = int(args.nms_k)
    config.activity_gate = float(args.activity_gate)
    config.cymbal_margin = float(args.cymbal_margin)
    config.tom_over_cymbal_margin = float(args.tom_over_cymbal_margin)
    try:
        config.patch_stride = int(args.patch_stride)
    except Exception:
        pass
    # Ensure HF gate off during calibration
    try:
        config.cymbal_highfreq_ratio_gate = None
    except Exception:
        pass
    # Mild min spacing (match main balanced defaults)
    try:
        config.min_spacing_ms_default = 22.0
        config.min_spacing_ms_map = {
            "0": 28.0,
            "1": 26.0,
            "2": 24.0,
            "3": 26.0,
            "4": 28.0,
            "66": 22.0,
            "67": 24.0,
            "68": 24.0,
        }
    except Exception:
        pass

    # Discover songs
    songs = _discover_songs(args.roots)
    if not songs:
        print("No songs with notes.mid + audio found under:", ", ".join(args.roots))
        return
    print(f"Calibrating on {len(songs)} song(s)")

    # Load model once
    charter = Charter(config, args.model)
    sr = config.sample_rate
    classes = get_drum_hits()
    base_thr = {c: 0.5 for c in classes}

    # Cache segments and truths per song
    cache: List[Tuple[str, List[dict], List[Event]]] = []
    for mid, aud in songs:
        segs = audio_to_tensors(str(aud), config)
        truth = load_truth_from_mid(mid)
        cache.append((str(aud), segs, truth))

    # Helper to score a thresholds map
    def score(thr_map: Dict[str, float]) -> Dict[str, Tuple[int, int, int]]:
        agg: Dict[str, Tuple[int, int, int]] = {c: (0, 0, 0) for c in classes}
        # Build ordered list in class order
        thr_list = [float(thr_map.get(c, 0.5)) for c in classes]
        charter.config.class_thresholds = thr_list
        for _aud, segs, truth in cache:
            df = charter.predict(segs)
            # Convert to events
            pred_events = _df_to_events(df, sr)
            # Match
            _, summary, _ = match_events(
                pred_events, truth, tol_s=float(args.tol_ms) / 1000.0
            )
            for c in classes:
                tp = int(summary[c]["tp"])  # type: ignore[index]
                fp = int(summary[c]["fp"])  # type: ignore[index]
                fn = int(summary[c]["fn"])  # type: ignore[index]
                old = agg[c]
                agg[c] = (old[0] + tp, old[1] + fp, old[2] + fn)
        return agg

    # Per-class sweep
    chosen: Dict[str, float] = {}
    for c in classes:
        best_thr = base_thr[c]
        best_f1 = -1.0
        for v in grid:
            thr_map = {**base_thr, **chosen, c: float(v)}
            agg = score(thr_map)
            tp, fp, fn = agg[c]
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(v)
        chosen[c] = float(best_thr)
        print(f"Class {c}: best_thr={best_thr:.3f} (F1={best_f1:.3f})")

    # Final pass with chosen set for report
    final_agg = score(chosen)
    print("\nAggregate PRF with chosen thresholds:")
    for c in classes:
        tp, fp, fn = final_agg[c]
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        print(f"  {c}: P={prec:.3f} R={rec:.3f} F1={f1:.3f} thr={chosen[c]:.3f}")

    # Write JSON next to checkpoint by default
    outp = (
        Path(args.out)
        if args.out
        else (Path(args.model).parent / "class_thresholds.json")
    )
    payload = {"classes": classes, "class_thresholds": [chosen[c] for c in classes]}
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved thresholds to {outp}")


if __name__ == "__main__":
    main()
