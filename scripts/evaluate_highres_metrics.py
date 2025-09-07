#!/usr/bin/env python3
"""
High-resolution evaluator: IOI-binned and subdivision recall for Clone Hero RB drum MIDIs.

Usage:
  python scripts/evaluate_highres_metrics.py \
    --songs-dir CloneHero/Songs \
    --model-ckpt models/local_transformer_models/best_model.ckpt \
    --config local_highres \
    --tol-patches 3 \
    --out-csv artifacts/highres_eval.csv

Notes:
- Requires torchaudio for feature extraction via SpectrogramProcessor.
- Loads calibrated class thresholds from <ckpt_dir>/class_thresholds.json when present.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import torch

from chart_hero.model_training.transformer_config import (
    get_config,
    get_drum_hits,
)
from chart_hero.model_training.transformer_model import create_model
from chart_hero.model_training.transformer_data import SpectrogramProcessor
from chart_hero.utils.rb_midi_utils import RbMidiProcessor


def series_to_events(series: torch.Tensor) -> List[int]:
    s = series.detach().to(torch.int8)
    s = torch.cat([torch.tensor([0], dtype=torch.int8, device=s.device), s])
    diff = s[1:] - s[:-1]
    onsets = (diff > 0).nonzero(as_tuple=False).view(-1).tolist()
    return onsets


def greedy_match(
    pred: Sequence[int], true: Sequence[int], tol: int
) -> Tuple[int, int, int, Dict[int, int]]:
    """Greedy match predicted to true events within +/- tol indices.

    Returns (tp, fp, fn, mapping) where mapping maps true_idx->pred_idx for matched events.
    """
    tp = 0
    used_true: set[int] = set()
    mapping: Dict[int, int] = {}
    for pi, p in enumerate(pred):
        best = None
        best_d = tol + 1
        for ti, t in enumerate(true):
            if ti in used_true:
                continue
            d = abs(p - t)
            if d <= tol and d < best_d:
                best = ti
                best_d = d
        if best is not None:
            used_true.add(best)
            mapping[best] = pi
            tp += 1
    fp = len(pred) - tp
    fn = len(true) - tp
    return tp, fp, fn, mapping


def evaluate_song(
    midi_path: Path,
    audio_path: Path,
    config,
    model: Optional[torch.nn.Module],
    processor: SpectrogramProcessor,
    tol_patches: int,
) -> dict:
    """Evaluate a single song, returning detailed per-class metrics and offsets."""
    # Load audio and build spectrogram
    y, sr = librosa.load(str(audio_path), sr=config.sample_rate)
    y_t = torch.from_numpy(y).float().unsqueeze(0)
    spec = processor.audio_to_spectrogram(y_t)  # (1, T, F) or (1, F, T)
    if spec.shape[1] != config.n_mels:
        spec = spec.transpose(1, 2)
    spec = spec.unsqueeze(1)  # (1, 1, F, T)

    # Labels
    num_frames = spec.shape[-1]
    rb = RbMidiProcessor(config)
    labels = rb.create_label_matrix(midi_path, int(num_frames))  # (T, C)

    # Downsample labels to patch steps via adaptive max pool
    lab = torch.nn.functional.adaptive_max_pool1d(
        labels.unsqueeze(0).permute(0, 2, 1),  # (1, C, T)
        output_size=None,
    )
    # Infer number of patches from model output if available; else approximate from patching
    if model is not None:
        with torch.no_grad():
            out = model(spec)
            logits = out["logits"]  # (1, T_p, C)
            t_patches = logits.shape[1]
            lab_p = torch.nn.functional.adaptive_max_pool1d(
                labels.unsqueeze(0).permute(0, 2, 1), t_patches
            )
            lab_p = lab_p.permute(0, 2, 1).reshape(-1, config.num_drum_classes)
            probs = torch.sigmoid(logits.reshape(-1, config.num_drum_classes))
    else:
        raise RuntimeError("Model checkpoint is required for evaluation.")

    # Thresholds (global or calibrated per-class)
    thr = getattr(config, "prediction_threshold", 0.5)
    thr_vec = None
    if getattr(config, "class_thresholds", None):
        ct = config.class_thresholds
        if isinstance(ct, (list, tuple)) and len(ct) == config.num_drum_classes:
            thr_vec = torch.tensor(ct, dtype=probs.dtype).view(1, -1)

    if thr_vec is not None:
        preds = (probs >= thr_vec).to(torch.int8)
    else:
        preds = (probs >= thr).to(torch.int8)

    # Per-class event lists and IOI bins
    classes = get_drum_hits()
    # duration per patch in milliseconds
    # step is patch_stride frames at hop_length
    patch_step_frames = int(getattr(config, "patch_stride", config.patch_size[0]))
    ms_per_patch = (
        patch_step_frames * config.hop_length * 1000.0 / float(config.sample_rate)
    )

    # bins in ms for IOI analysis
    default_bins = [10, 20, 30, 40, 60, 80, 120, 160, 240]

    results: dict = {"per_class": {}, "offsets": {}}
    for c_idx, c in enumerate(classes):
        p_events = series_to_events(preds[:, c_idx])
        t_events = series_to_events(lab_p[:, c_idx])
        tp, fp, fn, mapping = greedy_match(p_events, t_events, tol_patches)
        # Compute offsets (pred - true) in ms for matched pairs
        offsets_ms: List[float] = []
        for ti, pi in mapping.items():
            d_patches = p_events[pi] - t_events[ti]
            offsets_ms.append(d_patches * ms_per_patch)
        # IOI for true events (previous true of same class)
        ioi_ms: List[float] = []
        for i in range(1, len(t_events)):
            ioi_ms.append((t_events[i] - t_events[i - 1]) * ms_per_patch)
        # Assign true events to IOI bins (index by the event index i>=1)
        bin_totals = {b: 0 for b in default_bins}
        bin_hits = {b: 0 for b in default_bins}
        for i in range(1, len(t_events)):
            ioi = ioi_ms[i - 1]
            # Find first bin >= ioi
            b_sel = None
            for b in default_bins:
                if ioi <= b:
                    b_sel = b
                    break
            if b_sel is None:
                b_sel = default_bins[-1]
            bin_totals[b_sel] += 1
            if i in mapping:  # the i-th true event is matched
                bin_hits[b_sel] += 1

        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        results["per_class"][c] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "offsets_ms": offsets_ms,
            "ioi_bins": {
                str(b): {
                    "total": bin_totals[b],
                    "hits": bin_hits[b],
                    "recall": (bin_hits[b] / bin_totals[b])
                    if bin_totals[b] > 0
                    else None,
                }
                for b in default_bins
            },
        }
        results["offsets"][c] = offsets_ms

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate IOI-binned metrics (high-res)")
    ap.add_argument("--songs-dir", type=str, nargs="+", default=["CloneHero/Songs"])
    ap.add_argument("--model-ckpt", type=str, required=True)
    ap.add_argument("--config", type=str, default="local_highres")
    ap.add_argument("--tol-patches", type=int, default=3)
    ap.add_argument(
        "--difficulty",
        type=str,
        default=None,
        help="Filter by difficulty name in path (e.g., Expert)",
    )
    ap.add_argument("--max-songs", type=int, default=20)
    ap.add_argument("--out-csv", type=str, default=None)
    args = ap.parse_args()

    config = get_config(args.config)
    # Try to load calibrated thresholds from the checkpoint directory
    try:
        thrs_path = Path(args.model_ckpt).parent / "class_thresholds.json"
        if thrs_path.exists():
            with thrs_path.open("r") as f:
                payload = json.load(f)
            ct = payload.get("class_thresholds")
            if isinstance(ct, list) and len(ct) == config.num_drum_classes:
                config.class_thresholds = [float(x) for x in ct]
    except Exception:
        pass

    processor = SpectrogramProcessor(config)
    model = create_model(config)
    sd = torch.load(args.model_ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        model.load_state_dict(sd["state_dict"], strict=False)
    elif isinstance(sd, dict):
        model.load_state_dict(sd, strict=False)
    model.eval()

    songs: List[Tuple[Path, Path]] = []
    audio_exts = (".ogg", ".mp3", ".wav")
    for root in args.songs_dir:
        r = Path(root)
        if not r.exists():
            continue
        for dirpath, _dirnames, filenames in os.walk(r):
            if "notes.mid" in filenames:
                midi = Path(dirpath) / "notes.mid"
                # pick an audio file
                audio = None
                for fn in filenames:
                    if Path(fn).suffix.lower() in audio_exts:
                        audio = Path(dirpath) / fn
                        break
                if audio is not None:
                    # optional difficulty filter: only consider folders with token in path
                    if (
                        args.difficulty
                        and args.difficulty.lower() not in dirpath.lower()
                    ):
                        pass
                    songs.append((midi, audio))

    if not songs:
        print("No songs with notes.mid and audio found.")
        return
    songs = songs[: max(1, int(args.max_songs))]
    print(f"Evaluating {len(songs)} song(s)")

    classes = get_drum_hits()
    agg_bins = [10, 20, 30, 40, 60, 80, 120, 160, 240]
    # Aggregates
    agg_counts: Dict[str, Dict[str, int]] = {
        c: {str(b): 0 for b in agg_bins} for c in classes
    }
    agg_hits: Dict[str, Dict[str, int]] = {
        c: {str(b): 0 for b in agg_bins} for c in classes
    }
    agg_tp: Dict[str, int] = {c: 0 for c in classes}
    agg_fp: Dict[str, int] = {c: 0 for c in classes}
    agg_fn: Dict[str, int] = {c: 0 for c in classes}

    rows: List[List[str | float | int]] = []
    for midi, audio in songs:
        try:
            res = evaluate_song(midi, audio, config, model, processor, args.tol_patches)
        except Exception as e:
            print(f"ERR evaluating {midi}: {e}")
            continue
        base = f"{midi.parent.name}"
        for c, d in res["per_class"].items():
            agg_tp[c] += int(d["tp"])  # type: ignore[arg-type]
            agg_fp[c] += int(d["fp"])  # type: ignore[arg-type]
            agg_fn[c] += int(d["fn"])  # type: ignore[arg-type]
            for b, vals in d["ioi_bins"].items():
                agg_counts[c][b] += int(vals["total"] or 0)
                agg_hits[c][b] += int(vals["hits"] or 0)
                rows.append(
                    [
                        base,
                        c,
                        int(b),
                        int(vals["total"] or 0),
                        int(vals["hits"] or 0),
                        float(vals["recall"]) if vals["total"] else None,
                        float(d["precision"]),
                        float(d["recall"]),
                        float(d["f1"]),
                    ]
                )

    # Print aggregate summary
    print("\nAggregate IOI-binned recall:")
    for c in classes:
        line = [c]
        for b in agg_bins:
            total = agg_counts[c][str(b)]
            hits = agg_hits[c][str(b)]
            rec = hits / total if total > 0 else None
            line.append(f"{b}ms:{rec:.3f}" if rec is not None else f"{b}ms:n/a")
        print("  ".join(line))
    print("\nAggregate event-level PRF:")
    for c in classes:
        tp, fp, fn = agg_tp[c], agg_fp[c], agg_fn[c]
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        print(f"  {c}: P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    # Optional CSV export
    if args.out_csv:
        outp = Path(args.out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "song",
                    "class",
                    "ioi_bin_ms",
                    "support",
                    "hits",
                    "bin_recall",
                    "precision",
                    "recall",
                    "f1",
                ]
            )
            for row in rows:
                w.writerow(row)
        print(f"CSV written: {outp}")


if __name__ == "__main__":
    main()
