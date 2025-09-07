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


def choose_audio_path(song_dir: Path) -> Optional[Path]:
    # Priority: song.ogg if present; else any audio file in dir
    p = song_dir / "song.ogg"
    if p.exists():
        return p
    for fn in os.listdir(song_dir):
        if Path(fn).suffix.lower() in AUDIO_EXTS:
            return song_dir / fn
    return None


def list_stems(song_dir: Path) -> dict[str, Path]:
    stems: dict[str, Path] = {}
    for fn in os.listdir(song_dir):
        low = fn.lower()
        if not any(low.endswith(ext) for ext in AUDIO_EXTS):
            continue
        p = song_dir / fn
        key = low.split(".")[0]
        stems[key] = p
    return stems


def load_audio_for_training(
    song_dir: Path, config: BaseConfig
) -> tuple[np.ndarray, str]:
    """Select or synthesize training audio and return (waveform, domain).

    Domains: 'stems_full_mix', 'fullmix', 'drums_only'
    """
    stems = list_stems(song_dir)
    sr = int(config.sample_rate)
    # Prefer stems full-mix when drums and any other stem(s) exist
    drum_keys = [k for k in stems.keys() if "drum" in k]
    other_keys = [
        k
        for k in stems.keys()
        if any(
            w in k for w in ("guitar", "bass", "vocals", "rhythm", "keys", "backing")
        )
    ]
    if drum_keys and other_keys:
        tracks: list[np.ndarray] = []
        max_len = 0
        for k in drum_keys + other_keys:
            try:
                y, _ = librosa.load(str(stems[k]), sr=sr)
                tracks.append(y.astype(np.float32))
                max_len = max(max_len, len(y))
            except Exception:
                continue
        if tracks:
            mix = np.zeros(max_len, dtype=np.float32)
            for t in tracks:
                if len(t) < max_len:
                    t = np.pad(t, (0, max_len - len(t)))
                mix += t
            peak = float(np.max(np.abs(mix))) if mix.size > 0 else 1.0
            if peak > 0:
                mix = 0.5 * mix / peak  # -6 dB headroom
            return mix, "stems_full_mix"
    # Fall back to song.ogg or any audio file
    p = choose_audio_path(song_dir)
    if p is not None:
        y, _ = librosa.load(str(p), sr=sr)
        return y.astype(np.float32), "fullmix"
    # Last resort: drums only
    if drum_keys:
        try:
            y, _ = librosa.load(str(stems[drum_keys[0]]), sr=sr)
            return y.astype(np.float32), "drums_only"
        except Exception:
            pass
    return np.zeros(0, dtype=np.float32), "unknown"


def normalize_loudness_rms(y: np.ndarray, target_dbfs: float = -14.0) -> np.ndarray:
    """Approximate loudness normalization via RMS to a target dBFS.

    Not EBU R128 LUFS, but stabilizes dynamics across charts for training.
    Applies conservative gain with clipping prevention.
    """
    if y.size == 0:
        return y
    rms = float(np.sqrt(np.mean(np.square(y), dtype=np.float64)))
    if not np.isfinite(rms) or rms <= 1e-8:
        return y
    target = 10.0 ** (target_dbfs / 20.0)
    gain = target / rms
    # Prevent clipping
    peak = float(np.max(np.abs(y))) if y.size else 1.0
    if peak * gain > 0.99:
        gain = 0.99 / max(peak, 1e-6)
    return (y * gain).astype(np.float32)


def build_labels_from_midi(
    midi_path: Path, num_time_frames: int, config: BaseConfig
) -> Optional[torch.Tensor]:
    rb = RbMidiProcessor(config)
    return rb.create_label_matrix(midi_path, num_time_frames)


# --- .chart/.txt support ---
ALLOWED_DRUM_NOTE_BASE = {0, 1, 2, 3, 4, 5}
PRO_DRUMS_OFFSET = 64


def _chart_tick_to_seconds(
    tick: int, tempo_changes: list[tuple[int, int]], resolution: int
) -> float:
    sec = 0.0
    last_tick = 0
    last_us = 500000
    for t_tick, us_per_beat in tempo_changes:
        if t_tick > tick:
            break
        if t_tick > last_tick:
            sec += (t_tick - last_tick) * (last_us / 1_000_000.0) / max(1, resolution)
            last_tick = t_tick
        last_us = us_per_beat
    if tick > last_tick:
        sec += (tick - last_tick) * (last_us / 1_000_000.0) / max(1, resolution)
    return float(sec)


def _chart_tempos_us_per_beat(sync_track: list[dict]) -> list[tuple[int, int]]:
    changes: list[tuple[int, int]] = []
    for x in sync_track:
        if x.get("type") != "B":
            continue
        tick = int(x.get("tick", 0))
        bpm_x1000 = int(x.get("value", 120000))
        bpm = max(1.0, bpm_x1000 / 1000.0)
        us = int(round(60_000_000.0 / bpm))
        changes.append((tick, us))
    if not changes or changes[0][0] != 0:
        changes.append((0, 500000))
    changes.sort(key=lambda x: x[0])
    return changes


def _load_chart_obj(path: Path) -> dict | None:
    """Dynamically import and call scripts/discover_clonehero.parse_chart."""
    import importlib.util

    # scripts/ is 3 levels up from this file's directory
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    src_path = scripts_dir / "discover_clonehero.py"
    if not src_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("discover_clonehero", str(src_path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    try:
        return module.parse_chart(str(path))  # type: ignore[attr-defined]
    except Exception:
        return None


def build_labels_from_chart(
    chart_path: Path, num_time_frames: int, config: BaseConfig
) -> Optional[torch.Tensor]:
    obj = _load_chart_obj(chart_path)
    if not obj:
        return None
    # Choose difficulty track (prefer Expert)
    order = ["ExpertDrums", "HardDrums", "MediumDrums", "EasyDrums"]
    events: list[dict] = []
    for dname in order:
        evs = obj.get("tracks", {}).get(dname, [])
        if evs:
            events = evs
            break
    if not events:
        return None
    resolution = int(obj.get("song", {}).get("Resolution") or 192)
    tempos = _chart_tempos_us_per_beat(obj.get("sync_track", []))

    # Group by tick and determine cymbal flags for 2/3/4
    by_tick: dict[int, list[dict]] = {}
    for e in events:
        if e.get("kind") not in ("N", "S"):
            continue
        t = int(e.get("tick", 0))
        by_tick.setdefault(t, []).append(e)

    from chart_hero.model_training.transformer_config import DRUM_HIT_TO_INDEX

    label = torch.zeros((num_time_frames, len(DRUM_HIT_TO_INDEX)), dtype=torch.float32)
    for tick in sorted(by_tick.keys()):
        bucket = by_tick[tick]
        base_lanes = {
            int(x["code"])
            for x in bucket
            if x.get("kind") == "N"
            and int(x.get("code", -999)) in ALLOWED_DRUM_NOTE_BASE
        }
        flags_cym = {
            int(x["code"]) - PRO_DRUMS_OFFSET
            for x in bucket
            if x.get("kind") == "N"
            and int(x.get("code", 0)) >= PRO_DRUMS_OFFSET
            and (int(x.get("code")) - PRO_DRUMS_OFFSET) in ALLOWED_DRUM_NOTE_BASE
        }
        for lane in sorted(base_lanes):
            if lane == 0:
                hit = "0"
            elif lane == 1:
                hit = "1"
            elif lane in (2, 3, 4):
                if lane in flags_cym:
                    hit = {2: "66", 3: "67", 4: "68"}[lane]
                else:
                    hit = {2: "2", 3: "3", 4: "4"}[lane]
            else:
                hit = "4"
            sec = _chart_tick_to_seconds(tick, tempos, int(resolution))
            frame = int(sec * config.sample_rate / config.hop_length)
            if 0 <= frame < num_time_frames:
                idx = DRUM_HIT_TO_INDEX.get(hit)
                if idx is not None:
                    label[frame, idx] = 1.0
    return label


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


def estimate_and_apply_global_offset(
    y: np.ndarray, labels: torch.Tensor, config: BaseConfig
) -> torch.Tensor:
    """Estimate a small global offset (in frames) aligning labels to audio onset envelope.

    Uses librosa onset strength at the model's hop_length, searches lags within ±250 ms.
    Returns labels shifted accordingly (zeros filled).
    """
    try:
        hop = int(config.hop_length)
        sr = int(config.sample_rate)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        # labels: (T, C) -> sum across classes -> (T,)
        lab_sum = labels.sum(dim=1).detach().cpu().numpy().astype(np.float32)
        T = min(len(onset_env), lab_sum.shape[0])
        if T <= 1:
            return labels
        a = onset_env[:T]
        b = lab_sum[:T]

        # Normalize to unit energy to avoid scale bias
        def _norm(x: np.ndarray) -> np.ndarray:
            m = float(np.max(np.abs(x))) if x.size > 0 else 1.0
            return x / (m + 1e-8)

        a = _norm(a)
        b = _norm(b)
        # Search lags in frames within ±250 ms
        max_shift = max(1, int(round(0.250 * sr / hop)))
        best_lag = 0
        best_score = -1e9
        for lag in range(-max_shift, max_shift + 1):
            if lag < 0:
                x = a[-lag:]
                yv = b[: T + lag]
            elif lag > 0:
                x = a[: T - lag]
                yv = b[lag:]
            else:
                x = a
                yv = b
            if len(x) <= 1 or len(yv) <= 1:
                continue
            score = float(np.dot(x, yv) / max(1, len(x)))
            if score > best_score:
                best_score = score
                best_lag = lag
        if best_lag == 0:
            return labels
        # Positive lag means labels trail audio -> shift labels left by lag
        shifted = torch.zeros_like(labels)
        if best_lag > 0:
            shifted[:-best_lag, :] = labels[best_lag:, :]
        else:
            lag = -best_lag
            shifted[lag:, :] = labels[:-lag, :]
        return shifted
    except Exception:
        return labels


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

    # Collect records (.mid preferred; else .chart/.txt)
    recs: List[Record] = []
    for d in folders:
        midi = d / "notes.mid"
        if midi.exists():
            recs.append(Record(song_dir=d, notes_path=midi, source="midi"))
            continue
        for base in ("notes.chart", "notes.txt"):
            p = d / base
            if p.exists():
                recs.append(Record(song_dir=d, notes_path=p, source="chart"))
                break
    if not recs:
        print("No usable charts found (notes.mid / notes.chart / notes.txt).")
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
        try:
            # Load/mix audio and convert to spectrogram
            y, domain = load_audio_for_training(rec.song_dir, config)
            if y.size == 0:
                skipped += 1
                continue
            # Loudness normalize to target
            y_norm = normalize_loudness_rms(y, target_dbfs=-14.0)
            y_t = torch.from_numpy(y_norm).float().unsqueeze(0)
            spec = processor.audio_to_spectrogram(y_t)
            if spec.shape[1] != config.n_mels:
                spec = spec.transpose(1, 2)  # (1,T,F) -> (1,F,T)
            # Labels at frame-level (T, C)
            T = int(spec.shape[-1])
            if rec.source == "midi":
                labels = build_labels_from_midi(rec.notes_path, T, config)
            else:
                labels = build_labels_from_chart(rec.notes_path, T, config)
            if labels is None:
                skipped += 1
                continue
            # Global offset correction via onset alignment (±250 ms)
            labels = estimate_and_apply_global_offset(y, labels, config)
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
