#!/usr/bin/env python3
"""
Dataset builder for high-resolution drum transcription using Clone Hero song folders.

Scans one or more Clone Hero roots, parses Rock Band-style drum charts from
`notes.mid` (preferred) or `.chart/.txt`, selects audio, aligns globally,
writes spectrogram + frame label shards suitable for transformer training.

Initial prototype focuses on MIDI + single full mix selection. Stems mixing and
advanced alignment are left as TODOs.

Usage:
  python -m chart_hero.train.build_dataset \
    --roots CloneHero/Songs \
    --out-dir datasets/processed_highres \
    --config local_highres \
    --max-files 500
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import torch

from chart_hero.model_training.transformer_config import (
    BaseConfig,
    get_config,
    get_drum_hits,
)
from chart_hero.model_training.transformer_data import SpectrogramProcessor
from chart_hero.utils.rb_midi_utils import RbMidiProcessor


AUDIO_EXTS = (".ogg", ".mp3", ".wav", ".flac", ".m4a")


@dataclass
class Record:
    song_dir: Path
    notes_path: Path
    source: str  # "midi" | "chart"


def find_song_folders(roots: Sequence[str | os.PathLike[str]]) -> List[Path]:
    out: List[Path] = []
    for raw in roots:
        root = Path(raw)
        if not root.exists():
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            names = set(filenames)
            if "song.ini" in names and (
                "notes.mid" in names or "notes.chart" in names or "notes.txt" in names
            ):
                out.append(Path(dirpath))
    return out


def choose_audio(song_dir: Path) -> Optional[Path]:
    # Priority: song.ogg if present; else any audio file in dir
    p = song_dir / "song.ogg"
    if p.exists():
        return p
    for fn in os.listdir(song_dir):
        if Path(fn).suffix.lower() in AUDIO_EXTS:
            return song_dir / fn
    return None


def build_labels_from_midi(
    midi_path: Path, num_time_frames: int, config: BaseConfig
) -> Optional[torch.Tensor]:
    rb = RbMidiProcessor(config)
    return rb.create_label_matrix(midi_path, num_time_frames)


def dilate_labels_time(labels: torch.Tensor, frames: int) -> torch.Tensor:
    """Dilate binary labels along time to add tolerance."""
    if frames <= 0:
        return labels
    k = frames * 2 + 1
    pad = frames
    # labels: (T, C) -> (1, C, T)
    lab = labels.unsqueeze(0).permute(0, 2, 1)
    lab = torch.nn.functional.max_pool1d(lab, kernel_size=k, stride=1, padding=pad)
    lab = lab.permute(0, 2, 1).squeeze(0)
    return lab


def save_pair(
    out_dir: Path, base: str, spectrogram: torch.Tensor, labels: torch.Tensor
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Ensure shapes are (F, T) and (T, C)
    if spectrogram.dim() == 3:
        spec_np = spectrogram.squeeze(0).cpu().numpy()
    else:
        spec_np = spectrogram.cpu().numpy()
    lab_np = labels.cpu().numpy()
    spec_path = out_dir / f"{base}_mel.npy"
    lab_path = out_dir / f"{base}_label.npy"
    np.save(spec_path, spec_np)
    np.save(lab_path, lab_np)
    return spec_path, lab_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build high-res dataset from Clone Hero songs"
    )
    ap.add_argument("--roots", type=str, nargs="+", default=["CloneHero/Songs"])
    ap.add_argument("--out-dir", type=str, default="datasets/processed_highres")
    ap.add_argument("--config", type=str, default="local_highres")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--splits", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    args = ap.parse_args()

    config = get_config(args.config)
    processor = SpectrogramProcessor(config)

    folders = find_song_folders(args.roots)
    if not folders:
        print("No Clone Hero songs found.")
        return
    if args.max_files is not None and len(folders) > args.max_files:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(folders), size=args.max_files, replace=False)
        folders = [folders[int(i)] for i in sorted(idx)]
    print(f"Found {len(folders)} candidate song folders")

    # Collect records (MIDI-only for the initial version)
    recs: List[Record] = []
    for d in folders:
        midi = d / "notes.mid"
        if midi.exists():
            recs.append(Record(song_dir=d, notes_path=midi, source="midi"))
    if not recs:
        print("No notes.mid found; .chart parsing not yet implemented in builder.")
        return

    # Simple split by folder (deterministic shuffle)
    rng = np.random.default_rng(seed=1234)
    order = list(range(len(recs)))
    rng.shuffle(order)
    n = len(order)
    n_train = int(args.splits[0] * n)
    n_val = int(args.splits[1] * n)
    idx_train = set(order[:n_train])
    idx_val = set(order[n_train : n_train + n_val])

    out_root = Path(args.out_dir)
    out_train = out_root / "train"
    out_val = out_root / "val"
    out_test = out_root / "test"
    for p in (out_train, out_val, out_test):
        p.mkdir(parents=True, exist_ok=True)

    classes = get_drum_hits()
    written = 0
    skipped = 0
    for i, rec in enumerate(recs):
        audio = choose_audio(rec.song_dir)
        if audio is None:
            skipped += 1
            continue
        try:
            # Load audio and convert to spectrogram
            y, _ = librosa.load(str(audio), sr=config.sample_rate)
            y_t = torch.from_numpy(y).float().unsqueeze(0)
            spec = processor.audio_to_spectrogram(y_t)
            if spec.shape[1] != config.n_mels:
                spec = spec.transpose(1, 2)  # (1,T,F) -> (1,F,T)
            # Labels from MIDI at frame-level (T, C)
            T = int(spec.shape[-1])
            labels = build_labels_from_midi(rec.notes_path, T, config)
            if labels is None:
                skipped += 1
                continue
            # Optional label dilation for training robustness
            dil = int(getattr(config, "label_dilation_frames", 0) or 0)
            if dil > 0:
                labels = dilate_labels_time(labels, dil)

            # Save pair
            base = f"{rec.song_dir.parent.name}_{rec.song_dir.name}_{i:06d}"
            split_dir = (
                out_train if i in idx_train else (out_val if i in idx_val else out_test)
            )
            save_pair(split_dir, base, spec.squeeze(0), labels)
            written += 1
        except Exception as e:
            print(f"ERR processing {rec.song_dir}: {e}")
            skipped += 1
            continue

    print(
        f"Done. Wrote {written} examples ({out_root}). Skipped {skipped} due to missing audio/labels."
    )


if __name__ == "__main__":
    main()
