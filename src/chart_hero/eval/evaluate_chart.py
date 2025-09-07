from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import mido
import numpy as np
import pandas as pd

from chart_hero.inference.charter import Charter
from chart_hero.inference.input_transform import audio_to_tensors
from chart_hero.model_training.transformer_config import get_config, get_drum_hits
from chart_hero.utils.audio_io import load_audio

"""
Evaluate inference against a known-good PART DRUMS MIDI.

Usage:
  python -m chart_hero.eval.evaluate_chart \
    --audio path/to/song.ogg \
    --mid path/to/notes.mid \
    --model models/local_transformer_models/best_model.ckpt \
    [--patch-stride 8] [--tol-ms 45]

Outputs a concise summary of per-class precision/recall/F1 and mean/median
timing offsets (ms) for matched events.

Additions:
- IOI-binned recall and subdivision recall per class.
- Optional per-class constant offset application from JSON.
- Optional metrics CSV export (IOI and subdivision bins).
"""

# Mapping for PART DRUMS Expert gems
EXPERT_BASE = 96
GEM_TO_PAD = {
    EXPERT_BASE + 0: 0,  # Kick
    EXPERT_BASE + 1: 1,  # Snare
    EXPERT_BASE + 2: 2,  # Yellow
    EXPERT_BASE + 3: 3,  # Blue
    EXPERT_BASE + 4: 4,  # Green
    EXPERT_BASE + 5: 4,  # 5-lane orange -> map to green for our classes
}

# Cymbal toggles
TOGGLE_TO_PAD = {110: 2, 111: 3, 112: 4}

# Our class labels per pad when cym_on False/True
PAD_TO_CLASS_TOM = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"}
PAD_TO_CLASS_CYM = {2: "67", 3: "68", 4: "66"}


@dataclass
class Event:
    t: float  # seconds
    cls: str  # class label string matching get_drum_hits()


def load_truth_from_mid(mid_path: Path) -> List[Event]:
    mid = mido.MidiFile(mid_path, clip=True)
    # find PART DRUMS track
    tr = None
    for track in mid.tracks:
        for msg in track:
            if (
                msg.is_meta
                and msg.type == "track_name"
                and "PART DRUMS" in str(msg.name).upper()
            ):
                tr = track
                break
        if tr is not None:
            break
    if tr is None:
        # fallback: use last non-tempo track
        tr = mid.tracks[-1]

    # Gather tempo from track 0 (first set_tempo), default 120bpm
    tempo = 500000
    for msg in mid.tracks[0]:
        if msg.is_meta and msg.type == "set_tempo":
            tempo = int(msg.tempo)
            break

    events: List[Event] = []
    cym_on = {2: True, 3: True, 4: True}  # default ON like Clone Hero
    time_ticks = 0
    for msg in tr:
        time_ticks += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            if msg.note in TOGGLE_TO_PAD:
                pad = TOGGLE_TO_PAD[msg.note]
                cym_on[pad] = not cym_on[pad]
                continue
            pad = GEM_TO_PAD.get(msg.note)
            if pad is None:
                continue
            sec = mido.tick2second(time_ticks, mid.ticks_per_beat, tempo)
            if pad in (2, 3, 4) and cym_on.get(pad, True):
                cls = PAD_TO_CLASS_CYM[pad]
            else:
                cls = PAD_TO_CLASS_TOM.get(pad, None)
            if cls is not None:
                events.append(Event(t=float(sec), cls=str(cls)))
    return events


def _maybe_mix_audio(paths: Sequence[str], target_sr: int) -> Tuple[str, float]:
    """Load one or more audio files and mix down to mono float32, write temp WAV.

    Returns a filesystem path and sample rate. Writes under output/_eval_mix.wav.
    """
    import numpy as np
    import soundfile as sf

    if len(paths) == 1:
        return paths[0], float(target_sr)
    ys: List[np.ndarray] = []
    for p in paths:
        y, sr_f = load_audio(p, sr=target_sr)
        ys.append(y.astype(np.float32))
    # align by shortest
    m = min(map(len, ys))
    ys = [y[:m] for y in ys]
    mix = np.clip(np.sum(ys, axis=0) / max(1, len(ys)), -1.0, 1.0)
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "_eval_mix.wav"
    sf.write(str(out_path), mix, target_sr)
    return str(out_path), float(target_sr)


def predict_events(
    audio_paths: Sequence[str],
    model_path: str,
    *,
    patch_stride: Optional[int],
    threshold: Optional[float] = None,
    disable_calibrated: bool = False,
    activity_gate: Optional[float] = None,
    cymbal_margin: Optional[float] = None,
    nms_k: Optional[int] = None,
    class_thresholds: Optional[List[float]] = None,
    class_gains: Optional[List[float]] = None,
    cymbal_hf_gate: Optional[float] = None,
    cymbal_hf_cut: Optional[float] = None,
) -> List[Event]:
    config = get_config("local")
    # Apply balanced inference defaults if not provided via CLI
    try:
        # Global gates/margins/NMS
        if getattr(config, "activity_gate", None) is None:
            config.activity_gate = 0.50
        # Prefer slightly stronger NMS for stability
        config.event_nms_kernel_patches = int(
            max(1, getattr(config, "event_nms_kernel_patches", 3))
        )
        if config.event_nms_kernel_patches < 9:
            config.event_nms_kernel_patches = 9
        if getattr(config, "cymbal_margin", None) is None:
            config.cymbal_margin = 0.30
        # Make tom win require clearly higher prob than cymbal
        if float(getattr(config, "tom_over_cymbal_margin", 0.35)) < 0.45:
            config.tom_over_cymbal_margin = 0.45
        # Mild per-class min spacing by default (unless provided)
        try:
            base_map = getattr(config, "min_spacing_ms_map", None) or {}
            mild_map = {
                "0": 28.0,
                "1": 26.0,
                "2": 24.0,
                "3": 26.0,
                "4": 28.0,
                "66": 22.0,
                "67": 24.0,
                "68": 24.0,
            }
            for k, v in mild_map.items():
                if k not in base_map:
                    base_map[k] = v
            config.min_spacing_ms_map = base_map
            if getattr(config, "min_spacing_ms_default", None) is None:
                config.min_spacing_ms_default = 22.0
        except Exception:
            pass
        # Per-class thresholds default only if no checkpoint calibration file
        if (
            getattr(config, "class_thresholds", None) is None
            and class_thresholds is None
        ):
            try:
                from pathlib import Path as _Path

                thr_path = _Path(model_path).parent / "class_thresholds.json"
                has_calibrated = thr_path.exists()
            except Exception:
                has_calibrated = False
            if not has_calibrated:
                thr_map = {
                    "0": 0.55,
                    "1": 0.62,
                    "2": 0.60,
                    "3": 0.60,
                    "4": 0.65,
                    "66": 0.86,
                    "67": 0.88,
                    "68": 0.92,
                }
                classes = get_drum_hits()
                config.class_thresholds = [
                    float(thr_map.get(c, config.prediction_threshold)) for c in classes
                ]
        # Do not force class_gains by default; keep None unless provided
    except Exception:
        # If any defaults application fails, continue with existing config
        pass
    if patch_stride is not None and patch_stride > 0:
        config.patch_stride = int(patch_stride)
    # Optional tuning hooks
    if activity_gate is not None:
        config.activity_gate = float(activity_gate)
    if cymbal_margin is not None:
        config.cymbal_margin = float(cymbal_margin)
    if nms_k is not None and nms_k > 0:
        config.event_nms_kernel_patches = int(nms_k)
    # Optional HF cymbal gate
    if cymbal_hf_gate is not None:
        config.cymbal_highfreq_ratio_gate = float(cymbal_hf_gate)
    if cymbal_hf_cut is not None:
        config.cymbal_highfreq_cutoff_mel = float(cymbal_hf_cut)

    # Mix audio if multiple paths provided
    mixed_path, _ = _maybe_mix_audio(list(audio_paths), config.sample_rate)
    segments = audio_to_tensors(mixed_path, config)
    ch = Charter(config, model_path)
    # Optional threshold overrides after Charter init (which may load calibrated)
    if disable_calibrated:
        ch.config.class_thresholds = None
    if threshold is not None:
        ch.config.prediction_threshold = float(threshold)
    if class_thresholds is not None:
        # Only apply if same length as classes
        classes = get_drum_hits()
        if len(class_thresholds) == len(classes):
            ch.config.class_thresholds = [
                float(x) if x == x else ch.config.prediction_threshold
                for x in class_thresholds
            ]
    if class_gains is not None:
        classes = get_drum_hits()
        if len(class_gains) == len(classes):
            ch.config.class_gains = [float(max(0.0, g)) for g in class_gains]
    # Per-class thresholds via parsed mapping handled by caller by setting config.class_thresholds
    df = ch.predict(segments)
    sr = config.sample_rate
    ev: List[Event] = []
    for row in df.to_dict(orient="records"):
        sec = float(int(row["peak_sample"]) / float(sr))
        for cls in get_drum_hits():
            if int(row.get(cls, 0)) == 1:
                ev.append(Event(t=sec, cls=cls))
    return ev


def _greedy_match_indices(
    pred_idx: List[int], true_idx: List[int], tol: int
) -> Dict[int, int]:
    """Greedy match true indices to pred indices within +/- tol; returns mapping true->pred."""
    used_true: set[int] = set()
    mapping: Dict[int, int] = {}
    for pi, p in enumerate(pred_idx):
        best = None
        best_d = tol + 1
        for ti, t in enumerate(true_idx):
            if ti in used_true:
                continue
            d = abs(p - t)
            if d <= tol and d < best_d:
                best = ti
                best_d = d
        if best is not None:
            used_true.add(best)
            mapping[best] = pi
    return mapping


def match_events(
    pred: List[Event],
    true: List[Event],
    tol_s: float,
) -> Tuple[
    pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, Dict[str, Dict[str, float]]]
]:
    classes = get_drum_hits()
    # Build per-class lists
    pred_by = {c: [e for e in pred if e.cls == c] for c in classes}
    true_by = {c: [e for e in true if e.cls == c] for c in classes}

    rows: List[Dict[str, object]] = []
    summary: Dict[str, Dict[str, float]] = {}
    # IOI/subdivision stats per class
    ioi_bins_ms = [10, 20, 30, 40, 60, 80, 120, 160, 240]
    subdiv_keys = ["1/16", "1/32", "1/64", "1/128"]
    subdiv_vals = [1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0, 1.0 / 128.0]
    ioi_summary: Dict[str, Dict[str, Dict[str, float]]] = {}

    for c in classes:
        P = pred_by[c]
        T = true_by[c]
        used = set()
        tp = 0
        fp = 0
        fn = 0
        offsets: List[float] = []
        for p in P:
            best = None
            best_d = tol_s + 1
            for i, t in enumerate(T):
                if i in used:
                    continue
                d = abs(p.t - t.t)
                if d <= tol_s and d < best_d:
                    best = i
                    best_d = d
            if best is not None:
                used.add(best)
                tp += 1
                offsets.append((p.t - T[best].t))
            else:
                fp += 1
                rows.append({"cls": c, "type": "FP", "t": p.t})
        # Unmatched truths
        for i, t in enumerate(T):
            if i not in used:
                fn += 1
                rows.append({"cls": c, "type": "FN", "t": t.t})
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        off_ms = [o * 1000.0 for o in offsets]
        summary[c] = {
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "offset_mean_ms": float(np.mean(off_ms)) if off_ms else 0.0,
            "offset_median_ms": float(np.median(off_ms)) if off_ms else 0.0,
        }

        # Build IOI and subdivision stats for this class
        # True event indices and times
        true_idx = list(range(len(T)))
        # Build greedy mapping using integer indices based on times converted to pseudo-frames at 1 ms resolution
        # For this evaluator, use tol in samples -> convert times to integer milliseconds for robustness
        pred_ms_idx = [int(round(p.t * 1000.0)) for p in P]
        true_ms_idx = [int(round(t.t * 1000.0)) for t in T]
        tol_ms = int(round(tol_s * 1000.0))
        mapping = _greedy_match_indices(pred_ms_idx, true_ms_idx, tol_ms)

        # IOI bins
        bin_totals = {str(b): 0 for b in ioi_bins_ms}
        bin_hits = {str(b): 0 for b in ioi_bins_ms}
        # tempo estimate from this file (single tempo approximation)
        # compute seconds per beat
        # Get a naive tempo from adjacent true events if available
        spb_guess = None
        if len(T) > 1:
            intervals = [T[i].t - T[i - 1].t for i in range(1, len(T))]
            # median interval used as rough guide for subdivisions when no MIDI tempo provided downstream
            med = float(np.median(np.array(intervals))) if intervals else None
            if med and med > 0:
                spb_guess = med * 2.0  # crude: assume many are 8ths -> 0.5 beat spacing
        # Default to 0.5s per beat (~120 bpm) if unknown
        spb = float(spb_guess or 0.5)
        sub_totals = {k: 0 for k in subdiv_keys}
        sub_hits = {k: 0 for k in subdiv_keys}
        sub_tol = 0.2  # +/-20%

        for i in range(1, len(T)):
            ioi = (T[i].t - T[i - 1].t) * 1000.0  # ms
            # IOI bin select
            b_sel = None
            for b in ioi_bins_ms:
                if ioi <= b:
                    b_sel = str(b)
                    break
            if b_sel is None:
                b_sel = str(ioi_bins_ms[-1])
            bin_totals[b_sel] += 1
            if i in mapping:
                bin_hits[b_sel] += 1
            # Subdivision: nearest among target fractions of beat
            beats = (ioi / 1000.0) / spb if spb > 0 else 0.0
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

        ioi_summary[c] = {
            "ioi_bins": {
                b: {
                    "total": bin_totals[b],
                    "hits": bin_hits[b],
                    "recall": (bin_hits[b] / bin_totals[b])
                    if bin_totals[b] > 0
                    else None,
                }
                for b in map(str, ioi_bins_ms)
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

    return pd.DataFrame(rows), summary, ioi_summary


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate model vs known-good notes.mid")
    ap.add_argument(
        "--audio",
        required=True,
        nargs="+",
        help="One or more audio paths; if multiple are given (e.g., drums_1 drums_2 drums_3), they are mixed",
    )
    ap.add_argument("--mid", required=True, help="notes.mid path (ground truth)")
    ap.add_argument("--model", required=True, help="Model checkpoint path")
    ap.add_argument("--patch-stride", type=int, default=None)
    ap.add_argument("--tol-ms", type=float, default=45.0)
    ap.add_argument(
        "--csv", type=str, default=None, help="Optional path to write FP/FN CSV"
    )
    ap.add_argument(
        "--metrics-csv",
        type=str,
        default=None,
        help="Optional path to write IOI/subdivision metrics CSV",
    )
    ap.add_argument(
        "--offsets-json",
        type=str,
        default=None,
        help="Apply per-class constant offsets (ms) to predictions before matching",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override global prediction threshold (disables per-class if --disable-calibrated also set)",
    )
    ap.add_argument(
        "--disable-calibrated",
        action="store_true",
        help="Ignore calibrated per-class thresholds from checkpoint during eval",
    )
    ap.add_argument(
        "--activity-gate",
        type=float,
        default=None,
        help="Drop ticks whose max prob < gate (0..1)",
    )
    ap.add_argument(
        "--cymbal-margin",
        type=float,
        default=None,
        help="Require cymbal prob >= tom + margin for Y/B/G (default 0.1)",
    )
    ap.add_argument(
        "--nms-k",
        type=int,
        default=None,
        help="NMS window (patches) per class; odd numbers like 3 or 5 recommended",
    )
    ap.add_argument(
        "--class-thresholds",
        type=str,
        default=None,
        help="Override per-class thresholds as '0=0.55,1=0.55,2=0.6,3=0.6,4=0.6,66=0.7,67=0.75,68=0.8'",
    )
    ap.add_argument(
        "--class-gains",
        type=str,
        default=None,
        help="Per-class probability multipliers '0=1.0,1=1.0,2=0.4,3=0.4,4=0.4,66=1.0,67=1.1,68=1.1' (values clipped to >=0)",
    )
    ap.add_argument(
        "--cymbal-hf-gate",
        type=float,
        default=None,
        help="Require this fraction of mel energy in high bands for cymbal classes (e.g., 0.30–0.40)",
    )
    ap.add_argument(
        "--cymbal-hf-cut",
        type=float,
        default=None,
        help="Fraction of mel bins considered 'high' for the cymbal HF gate (default 0.70)",
    )
    args = ap.parse_args()

    truth = load_truth_from_mid(Path(args.mid))

    # Optional per-class thresholds
    if args.class_thresholds:
        thr_map = {}
        for kv in args.class_thresholds.split(","):
            kv = kv.strip()
            if not kv or "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            try:
                thr_map[str(k.strip())] = float(v)
            except Exception:
                pass
        if thr_map:
            # Put into order of get_drum_hits
            classes = get_drum_hits()
            config = get_config("local")
            class_list = [float(thr_map.get(c, np.nan)) for c in classes]
            # pass via activity gate hook by setting environment? Instead, we'll set after Charter init below; store in locals
            per_class_overrides = class_list
        else:
            per_class_overrides = None
    else:
        per_class_overrides = None

    # Parse class gains if provided
    if args.class_gains:
        gn_map = {}
        for kv in args.class_gains.split(","):
            kv = kv.strip()
            if not kv or "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            try:
                gn_map[str(k.strip())] = float(v)
            except Exception:
                pass
        if gn_map:
            classes = get_drum_hits()
            gains_list = [float(gn_map.get(c, 1.0)) for c in classes]
        else:
            gains_list = None
    else:
        gains_list = None

    # Optional per-class offsets (ms)
    offsets_map: Optional[Dict[str, float]] = None
    if args.offsets_json:
        try:
            import json as _json

            with open(args.offsets_json, "r") as f:
                data = _json.load(f)
            if isinstance(data, dict) and "offsets_ms" in data:
                m = data["offsets_ms"]
            else:
                m = data
            classes = get_drum_hits()
            if isinstance(m, dict):
                offsets_map = {str(k): float(v) for k, v in m.items()}
            elif isinstance(m, list) and len(m) == len(classes):
                offsets_map = {classes[i]: float(v) for i, v in enumerate(m)}
        except Exception:
            offsets_map = None

    pred = predict_events(
        args.audio,
        args.model,
        patch_stride=args.patch_stride,
        threshold=args.threshold,
        disable_calibrated=bool(args.disable_calibrated),
        activity_gate=args.activity_gate,
        cymbal_margin=args.cymbal_margin,
        nms_k=args.nms_k,
        class_thresholds=per_class_overrides,
        class_gains=gains_list,
        cymbal_hf_gate=args.cymbal_hf_gate,
        cymbal_hf_cut=args.cymbal_hf_cut,
    )
    # Apply per-class time offsets to predictions
    if offsets_map:
        shifted: List[Event] = []
        for e in pred:
            off_ms = float(offsets_map.get(e.cls, 0.0))
            shifted.append(Event(t=e.t - off_ms / 1000.0, cls=e.cls))
        pred = shifted
    df, summary, ioi = match_events(pred, truth, tol_s=args.tol_ms / 1000.0)

    # Print concise summary
    order = get_drum_hits()
    print("class tp fp fn precision recall f1 off_mean_ms off_med_ms")
    for c in order:
        s = summary[c]
        print(
            f"{c:>3} {int(s['tp']):4d} {int(s['fp']):4d} {int(s['fn']):4d} "
            f"{s['precision']:.3f} {s['recall']:.3f} {s['f1']:.3f} "
            f"{s['offset_mean_ms']:.1f} {s['offset_median_ms']:.1f}"
        )

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)

    # Print IOI and subdivision metrics
    print("\nIOI-binned recall (per class):")
    order = get_drum_hits()
    for c in order:
        bins = ioi[c]["ioi_bins"]
        parts = [c]
        for b in ["10", "20", "30", "40", "60", "80", "120", "160", "240"]:
            val = bins[b]
            if val["total"] > 0 and val["recall"] is not None:
                parts.append(f"{b}ms:{val['recall']:.3f}")
            else:
                parts.append(f"{b}ms:n/a")
        print("  ".join(parts))
    print("\nSubdivision recall (per class):")
    for c in order:
        bins = ioi[c]["subdiv_bins"]
        parts = [c]
        for k in ["1/16", "1/32", "1/64", "1/128"]:
            val = bins[k]
            if val["total"] > 0 and val["recall"] is not None:
                parts.append(f"{k}:{val['recall']:.3f}")
            else:
                parts.append(f"{k}:n/a")
        print("  ".join(parts))

    # Optional metrics CSV export
    if args.metrics_csv:
        outp = Path(args.metrics_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        import csv as _csv

        with outp.open("w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["class", "metric", "bin_label", "support", "hits", "recall"])
            for c in order:
                for b, vals in ioi[c]["ioi_bins"].items():
                    w.writerow(
                        [
                            c,
                            "ioi",
                            b,
                            int(vals["total"]),
                            int(vals["hits"]),
                            vals["recall"] if vals["total"] > 0 else None,
                        ]
                    )
                for k, vals in ioi[c]["subdiv_bins"].items():
                    w.writerow(
                        [
                            c,
                            "subdiv",
                            k,
                            int(vals["total"]),
                            int(vals["hits"]),
                            vals["recall"] if vals["total"] > 0 else None,
                        ]
                    )


if __name__ == "__main__":
    main()
