#!/usr/bin/env python3
"""
High-resolution evaluator: IOI-binned and subdivision recall for Clone Hero RB drum MIDIs.

Usage:
  python scripts/evaluate_highres_metrics.py \
    --songs-dir CloneHero/Songs \
    --model-ckpt models/local_transformer_models/best_model.ckpt \
    --config local \
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
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import mido
import numpy as np
import torch

from chart_hero.model_training.transformer_config import get_config, get_drum_hits
from chart_hero.model_training.transformer_data import SpectrogramProcessor
from chart_hero.model_training.transformer_model import create_model
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
    apply_offsets_ms: Optional[Dict[str, float]] = None,
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

    # Build tempo map in seconds for subdivision recall
    def _tempo_map_sec(mp: Path) -> List[Tuple[float, int]]:
        try:
            try:
                mf = mido.MidiFile(mp, clip=True)  # type: ignore[call-arg]
            except TypeError:
                mf = mido.MidiFile(mp)
        except Exception:
            return [(0.0, 500000)]
        tpq = mf.ticks_per_beat or 480
        # Collect tempo changes as (tick, us_per_beat)
        changes: List[Tuple[int, int]] = []
        for tr in mf.tracks:
            t = 0
            for msg in tr:
                t += msg.time
                if msg.is_meta and msg.type == "set_tempo":
                    changes.append((int(t), int(msg.tempo)))
        if not changes or changes[0][0] != 0:
            changes.insert(0, (0, 500000))
        changes.sort(key=lambda x: x[0])
        # Convert ticks to absolute seconds
        out: List[Tuple[float, int]] = []
        sec = 0.0
        last_tick = 0
        last_us = 500000
        for tick, us in changes:
            if tick > last_tick:
                sec += (tick - last_tick) * (last_us / 1_000_000.0) / tpq
                last_tick = tick
            out.append((float(sec), int(us)))
            last_us = us
        return out or [(0.0, 500000)]

    tempo_sec = _tempo_map_sec(midi_path)

    def _spb_at(t_sec: float) -> float:
        # seconds per beat at given absolute time
        prev = 0.0
        us = 500000
        for ts, u in tempo_sec:
            if ts > t_sec:
                break
            prev = ts
            us = u
        return float(us) / 1_000_000.0

    for c_idx, c in enumerate(classes):
        p_events = series_to_events(preds[:, c_idx])
        # Apply per-class constant offset (ms) to predictions before matching
        if apply_offsets_ms and c in apply_offsets_ms and apply_offsets_ms[c] != 0:
            shift = int(round(float(apply_offsets_ms[c]) / ms_per_patch))
            if shift != 0:
                p_events = [max(0, p - shift) for p in p_events]
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
        # Subdivision bins (focus on 1/16, 1/32, 1/64, 1/128)
        subdiv_keys = ["1/16", "1/32", "1/64", "1/128"]
        subdiv_vals = [1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0, 1.0 / 128.0]
        sub_totals = {k: 0 for k in subdiv_keys}
        sub_hits = {k: 0 for k in subdiv_keys}
        sub_tol = 0.2  # +/-20% tolerance around ideal subdivision
        # event times in seconds for true events
        t_times = [te * ms_per_patch / 1000.0 for te in t_events]
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
            # Subdivision assignment (based on local tempo at event i)
            spb = _spb_at(t_times[i])  # seconds per beat
            beats = (ioi / 1000.0) / spb if spb > 0 else 0.0
            # choose nearest among target subdivisions
            best_k = None
            best_err = 1e9
            for k, v in zip(subdiv_keys, subdiv_vals):
                err = abs(beats - v) / max(v, 1e-6)
                if err < best_err:
                    best_err = err
                    best_k = k
            if best_k is not None and best_err <= sub_tol:
                sub_totals[best_k] += 1
                if i in mapping:
                    sub_hits[best_k] += 1

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
            "subdiv_bins": {
                k: {
                    "total": sub_totals[k],
                    "hits": sub_hits[k],
                    "recall": (sub_hits[k] / sub_totals[k])
                    if sub_totals[k] > 0
                    else None,
                }
                for k in subdiv_keys
            },
        }
        results["offsets"][c] = offsets_ms

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate IOI-binned metrics (high-res)")
    ap.add_argument("--songs-dir", type=str, nargs="+", default=["CloneHero/Songs"])
    ap.add_argument("--model-ckpt", type=str, required=True)
    ap.add_argument("--config", type=str, default="local")
    ap.add_argument("--tol-patches", type=int, default=3)
    ap.add_argument(
        "--difficulty",
        type=str,
        default=None,
        help="Filter by difficulty name in path (e.g., Expert)",
    )
    ap.add_argument("--max-songs", type=int, default=20)
    ap.add_argument("--out-csv", type=str, default=None)
    ap.add_argument(
        "--offsets-json",
        type=str,
        default=None,
        help="Path to per-class offsets (ms) JSON to apply during eval",
    )
    ap.add_argument(
        "--learn-offsets",
        action="store_true",
        help="Learn per-class constant offsets (median) from matched events",
    )
    ap.add_argument(
        "--save-offsets",
        type=str,
        default=None,
        help="Write learned offsets JSON to this path",
    )
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
    agg_sub_keys = ["1/16", "1/32", "1/64", "1/128"]
    # Aggregates
    agg_counts: Dict[str, Dict[str, int]] = {
        c: {str(b): 0 for b in agg_bins} for c in classes
    }
    agg_hits: Dict[str, Dict[str, int]] = {
        c: {str(b): 0 for b in agg_bins} for c in classes
    }
    agg_sub_counts: Dict[str, Dict[str, int]] = {
        c: {k: 0 for k in agg_sub_keys} for c in classes
    }
    agg_sub_hits: Dict[str, Dict[str, int]] = {
        c: {k: 0 for k in agg_sub_keys} for c in classes
    }
    agg_tp: Dict[str, int] = {c: 0 for c in classes}
    agg_fp: Dict[str, int] = {c: 0 for c in classes}
    agg_fn: Dict[str, int] = {c: 0 for c in classes}

    # CSV rows: song,class,metric,bin_label,support,hits,bin_recall,precision,recall,f1
    rows: List[List[str | float | int]] = []

    # Offsets to apply (if provided)
    apply_offsets: Optional[Dict[str, float]] = None
    if args.offsets_json:
        try:
            with open(args.offsets_json, "r") as f:
                data = json.load(f)
            # accept list or map
            classes = get_drum_hits()
            if isinstance(data, dict) and "offsets_ms" in data:
                m = data["offsets_ms"]
            else:
                m = data
            if isinstance(m, dict):
                apply_offsets = {str(k): float(v) for k, v in m.items()}
            elif isinstance(m, list):
                apply_offsets = {
                    classes[i]: float(v) for i, v in enumerate(m) if i < len(classes)
                }
        except Exception:
            apply_offsets = None

    # Accumulate offsets when learning
    offsets_samples: Dict[str, List[float]] = {c: [] for c in classes}
    for midi, audio in songs:
        try:
            res = evaluate_song(
                midi,
                audio,
                config,
                model,
                processor,
                args.tol_patches,
                apply_offsets_ms=apply_offsets,
            )
        except Exception as e:
            print(f"ERR evaluating {midi}: {e}")
            continue
        base = f"{midi.parent.name}"
        for c, d in res["per_class"].items():
            # Collect offsets if learning
            if args.learn_offsets:
                try:
                    offsets_samples[c].extend(
                        [float(x) for x in d.get("offsets_ms", [])]
                    )
                except Exception:
                    pass
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
                        "ioi",
                        str(int(b)),
                        int(vals["total"] or 0),
                        int(vals["hits"] or 0),
                        float(vals["recall"]) if vals["total"] else None,
                        float(d["precision"]),
                        float(d["recall"]),
                        float(d["f1"]),
                    ]
                )
            # Subdivision rows
            for k, vals in d.get("subdiv_bins", {}).items():
                agg_sub_counts[c][k] += int(vals["total"] or 0)
                agg_sub_hits[c][k] += int(vals["hits"] or 0)
                rows.append(
                    [
                        base,
                        c,
                        "subdiv",
                        k,
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

    print("\nAggregate subdivision recall:")
    for c in classes:
        parts = [c]
        for k in agg_sub_keys:
            total = agg_sub_counts[c][k]
            hits = agg_sub_hits[c][k]
            rec = hits / total if total > 0 else None
            parts.append(f"{k}:{rec:.3f}" if rec is not None else f"{k}:n/a")
        print("  ".join(parts))

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
                    "metric",
                    "bin_label",
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

    # Learn and optionally save offsets
    if args.learn_offsets:
        learned: Dict[str, float] = {}
        for c in classes:
            vals = offsets_samples[c]
            if vals:
                learned[c] = float(np.median(np.array(vals, dtype=np.float32)))
        print("\nLearned per-class offsets (ms):", json.dumps(learned, indent=2))
        if args.save_offsets:
            payload = {"offsets_ms": learned, "classes": classes}
            outp = Path(args.save_offsets)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with outp.open("w") as f:
                json.dump(payload, f, indent=2)
            print(f"Offsets JSON written to {outp}")


if __name__ == "__main__":
    main()
