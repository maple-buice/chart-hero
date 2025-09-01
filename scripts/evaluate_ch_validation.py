#!/usr/bin/env python3
"""
Evaluate a trained model against Clone Hero songs that include Rock Band-style drum MIDIs.

For each song directory under --songs-dir, attempts to find a MIDI (notes.mid) and audio
(song.ogg/mp3/wav). Builds frame-level labels using RbMidiProcessor, runs the model to
produce predictions, downsamples labels to patch steps, and computes event-level metrics
with tolerance.

Usage:
  python scripts/evaluate_ch_validation.py --songs-dir CloneHero/Songs --model-ckpt path/to.ckpt --config local

If --model-ckpt is omitted, prints label coverage only.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import torch

from chart_hero.model_training.transformer_config import get_config
from chart_hero.model_training.transformer_model import create_model
from chart_hero.model_training.transformer_data import SpectrogramProcessor
from chart_hero.utils.rb_midi_utils import RbMidiProcessor


def series_to_events(series: torch.Tensor) -> List[int]:
    s = series.detach().to(torch.int8)
    s = torch.cat([torch.tensor([0], dtype=torch.int8, device=s.device), s])
    diff = s[1:] - s[:-1]
    onsets = (diff > 0).nonzero(as_tuple=False).view(-1).tolist()
    return onsets


def match_events(pred: List[int], true: List[int], tol: int) -> Tuple[int, int, int]:
    tp = 0
    used = set()
    for p in pred:
        best = None
        best_d = tol + 1
        for i, t in enumerate(true):
            if i in used:
                continue
            d = abs(p - t)
            if d <= tol and d < best_d:
                best = i
                best_d = d
        if best is not None:
            used.add(best)
            tp += 1
    fp = len(pred) - tp
    fn = len(true) - tp
    return tp, fp, fn


def evaluate_song(
    midi_path: Path,
    audio_path: Path,
    config,
    model: Optional[torch.nn.Module],
    processor: SpectrogramProcessor,
    tol_patches: int = 1,
) -> Tuple[float, float, float]:
    # Load audio and build spectrogram
    y, sr = librosa.load(str(audio_path), sr=config.sample_rate)
    y_t = torch.from_numpy(y).float().unsqueeze(0)
    spec = processor.audio_to_spectrogram(
        y_t
    )  # (1, T, F) or (1, F, T) normalized inside
    # Ensure (1, 1, F, T)
    if spec.shape[1] != config.n_mels:
        # spec is (1, T, F) -> transpose to (1, F, T)
        spec = spec.transpose(1, 2)
    spec = spec.unsqueeze(1)

    # Labels
    num_frames = spec.shape[-1]
    rb = RbMidiProcessor(config)
    labels = rb.create_label_matrix(midi_path, num_frames)

    if model is None:
        total = int(labels.sum().item())
        print(f"Labels only: total positives {total}")
        return (0.0, 0.0, 0.0)

    model.eval()
    with torch.no_grad():
        out = model(spec)
        logits = out["logits"]  # (1, T_patches, C)
        t_patches = logits.shape[1]
        # Downsample labels to patches via adaptive max pool
        lab = torch.nn.functional.adaptive_max_pool1d(
            labels.unsqueeze(0).permute(0, 2, 1), t_patches
        )
        lab = lab.permute(0, 2, 1).reshape(-1, config.num_drum_classes)
        probs = torch.sigmoid(logits.reshape(-1, config.num_drum_classes))
        thr = getattr(config, "prediction_threshold", 0.5)
        preds = (probs > thr).to(torch.int8)

    # Event-level metrics across all classes
    tp_total = fp_total = fn_total = 0
    for c in range(config.num_drum_classes):
        p_events = series_to_events(preds[:, c])
        t_events = series_to_events(lab[:, c])
        tp, fp, fn = match_events(p_events, t_events, tol_patches)
        tp_total += tp
        fp_total += fp
        fn_total += fn
    prec = tp_total / max(1, tp_total + fp_total)
    rec = tp_total / max(1, tp_total + fn_total)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return (prec, rec, f1)


def main():
    ap = argparse.ArgumentParser(description="Evaluate model on CH RB MIDIs")
    ap.add_argument("--songs-dir", type=str, default="CloneHero/Songs")
    ap.add_argument("--model-ckpt", type=str, default=None)
    ap.add_argument("--config", type=str, default="local")
    ap.add_argument("--tol-patches", type=int, default=1)
    args = ap.parse_args()

    config = get_config(args.config)
    processor = SpectrogramProcessor(config)
    model = None
    if args.model_ckpt:
        model = create_model(config)
        # Load checkpoint state_dict if available
        sd = torch.load(args.model_ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            model.load_state_dict(sd["state_dict"], strict=False)
        elif isinstance(sd, dict):
            model.load_state_dict(sd, strict=False)

    songs_path = Path(args.songs_dir)
    mids: List[Tuple[Path, Path]] = []
    audio_exts = (".ogg", ".mp3", ".wav")
    for root, _dirs, files in os.walk(songs_path):
        if "notes.mid" in files:
            midi = Path(root) / "notes.mid"
            # Pick an audio file if present
            audio = None
            for fn in files:
                if Path(fn).suffix.lower() in audio_exts:
                    audio = Path(root) / fn
                    break
            if audio is not None:
                mids.append((midi, audio))

    print(f"Found {len(mids)} song(s) with notes.mid and audio")
    agg_prec = agg_rec = agg_f1 = 0.0
    n_eval = 0
    for midi, audio in mids[:10]:  # limit for quick run
        print(f"Evaluating: {midi}")
        prec, rec, f1 = evaluate_song(
            midi, audio, config, model, processor, args.tol_patches
        )
        if model is not None:
            print(f"  P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
            agg_prec += prec
            agg_rec += rec
            agg_f1 += f1
            n_eval += 1
    if model is not None and n_eval > 0:
        print(
            f"Aggregate over {n_eval} songs: P={agg_prec / n_eval:.3f} R={agg_rec / n_eval:.3f} F1={agg_f1 / n_eval:.3f}"
        )


if __name__ == "__main__":
    main()
