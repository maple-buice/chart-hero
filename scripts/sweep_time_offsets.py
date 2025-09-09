#!/usr/bin/env python3
"""
Sweep constant time offsets applied to model predictions when evaluating Clone Hero
songs.  Useful for diagnosing global alignment issues.

Example usage:
  python scripts/sweep_time_offsets.py \
      --model models/local_transformer_models/best_model.ckpt \
      --offsets -80,-60,-40,-20,0,20,40,60,80
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from chart_hero.eval.evaluate_chart import Event, load_truth_from_mid, match_events
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate with swept prediction offsets")
    ap.add_argument(
        "--roots", type=str, nargs="+", default=["CloneHero/KnownGoodSongs"]
    )
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--config", type=str, default="local")
    ap.add_argument(
        "--offsets",
        type=str,
        default="-80,-40,-20,-10,0,10,20,40,80",
        help="Comma-separated offsets in milliseconds",
    )
    ap.add_argument("--nms-k", type=int, default=9)
    ap.add_argument("--activity-gate", type=float, default=0.45)
    ap.add_argument("--cymbal-margin", type=float, default=0.30)
    ap.add_argument("--tom-over-cymbal-margin", type=float, default=0.35)
    ap.add_argument("--patch-stride", type=int, default=1)
    ap.add_argument("--tol-ms", type=float, default=45.0)
    args = ap.parse_args()

    # Parse offset grid
    off_vals = []
    for s in (x.strip() for x in args.offsets.split(",")):
        try:
            off_vals.append(float(s))
        except Exception:
            pass
    offsets = sorted(set(off_vals)) or [0.0]

    # Build config similar to calibrate_thresholds defaults
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
    print(f"Evaluating {len(songs)} song(s)")

    # Load model once
    charter = Charter(config, args.model)
    sr = config.sample_rate
    classes = get_drum_hits()

    # Cache predicted events and truths per song
    cache: List[Tuple[List[Event], List[Event]]] = []
    for mid, aud in songs:
        segs = audio_to_tensors(str(aud), config)
        truth = load_truth_from_mid(mid)
        df = charter.predict(segs)
        preds = _df_to_events(df, sr)
        cache.append((preds, truth))

    # Evaluate for each offset
    tol_s = float(args.tol_ms) / 1000.0
    for off in offsets:
        off_s = off / 1000.0
        agg: Dict[str, Tuple[int, int, int]] = {c: (0, 0, 0) for c in classes}
        for preds, truth in cache:
            shifted = [Event(t=e.t + off_s, cls=e.cls) for e in preds]
            _df, summary, _ = match_events(shifted, truth, tol_s)
            for c in classes:
                tp = int(summary[c]["tp"])
                fp = int(summary[c]["fp"])
                fn = int(summary[c]["fn"])
                old = agg[c]
                agg[c] = (old[0] + tp, old[1] + fp, old[2] + fn)
        # Aggregate metrics
        print(f"\nOffset {off:+.1f} ms")
        for c in classes:
            tp, fp, fn = agg[c]
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
            print(f"  {c}: P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
        # Overall
        tot_tp = sum(a[0] for a in agg.values())
        tot_fp = sum(a[1] for a in agg.values())
        tot_fn = sum(a[2] for a in agg.values())
        tot_prec = tot_tp / max(1, tot_tp + tot_fp)
        tot_rec = tot_tp / max(1, tot_tp + tot_fn)
        tot_f1 = 0.0 if (tot_prec + tot_rec) == 0 else 2 * tot_prec * tot_rec / (tot_prec + tot_rec)
        print(f"  ALL: P={tot_prec:.3f} R={tot_rec:.3f} F1={tot_f1:.3f}")


if __name__ == "__main__":
    main()
