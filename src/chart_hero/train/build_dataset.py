#!/usr/bin/env python3
"""
Dataset builder for high-resolution drum transcription using Clone Hero song folders.

Scans one or more Clone Hero roots, parses Rock Band-style drum charts from
`notes.mid` (preferred) or `.chart/.txt`, selects audio, aligns globally,
writes spectrogram + frame label shards suitable for transformer training.

Initial prototype focuses on MIDI + single full mix selection. Stems mixing and
advanced alignment are left as TODOs. Uses Expert-only drum charts and prefers
pro_drums-capable charts/tracks when available.

Usage:
  python -m chart_hero.train.build_dataset \
    --roots CloneHero/Songs \
    --out-dir datasets/processed_highres \
    --config local_highres \
    --max-files 500
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import torch

from chart_hero.model_training.transformer_config import BaseConfig, get_config
from chart_hero.model_training.transformer_data import SpectrogramProcessor
from chart_hero.utils.rb_midi_utils import RbMidiProcessor
from chart_hero.utils.song_utils import (
    choose_audio_path,
    list_stems,
    mix_stems_to_waveform,
)

MAX_HASHES = 100000


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


def _parse_bool_like(v: str) -> Optional[bool]:
    s = v.strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return None


def parse_song_ini(song_dir: Path) -> dict[str, str]:
    """Parse song.ini for simple metadata (artist, name, charter, pro_drums)."""
    ini_path = song_dir / "song.ini"
    info: dict[str, str] = {}
    if not ini_path.exists():
        return info
    try:
        with ini_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip().lower()
                v = v.strip()
                if k in ("artist", "name", "charter", "genre"):
                    info[k] = v
                elif k == "pro_drums":
                    b = _parse_bool_like(v)
                    # keep original string if not a clear boolean; downstream may still inspect
                    info[k] = str(b) if b is not None else v
    except Exception:
        pass
    return info


def _pack_name_for(song_dir: Path, roots: Sequence[str | os.PathLike[str]]) -> str:
    """Best-effort guess of the pack name that contains this song.

    Typically, Clone Hero layouts are <root>/Songs/<Pack>/<Song>/..., but we
    simply take the first path segment beneath the matching root. Falls back to
    the immediate parent directory name.
    """
    try:
        rp = song_dir.resolve()
    except Exception:
        rp = song_dir
    for raw in roots:
        try:
            base = Path(raw).resolve()
            rel = rp.relative_to(base)
            if len(rel.parts) > 0:
                return rel.parts[0]
        except Exception:
            continue
    return song_dir.parent.name


def _slugify(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "untitled"


def json_index_has_song(
    song_dir: Path, roots: Sequence[str | os.PathLike[str]], json_root: Path
) -> bool:
    """Return True if the song directory is present in the JSON cache."""
    meta = parse_song_ini(song_dir)
    artist = meta.get("artist") or song_dir.parent.name
    name = meta.get("name") or song_dir.name
    pack = _pack_name_for(song_dir, roots)
    idx_dir = json_root / _slugify(pack) / _slugify(artist) / _slugify(name)
    try:
        return idx_dir.exists() and any(idx_dir.glob("*.json"))
    except Exception:
        return False


def load_audio_for_training(
    song_dir: Path, config: BaseConfig
) -> tuple[np.ndarray, str]:
    """Select or synthesize training audio and return (waveform, domain).

    Domains: 'stems_full_mix', 'fullmix', 'drums_only'
    """
    sr = int(config.sample_rate)
    mix = mix_stems_to_waveform(song_dir, sr)
    if mix is not None:
        return mix, "stems_full_mix"

    stems = list_stems(song_dir)
    p = choose_audio_path(song_dir)
    if p is not None:
        y, _ = librosa.load(str(p), sr=sr)
        return y.astype(np.float32), "fullmix"

    drum_keys = [k for k in stems.keys() if "drum" in k]
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
    midi_path: Path, num_time_frames: int, processor: RbMidiProcessor
) -> Optional[torch.Tensor]:
    labels = processor.create_label_matrix(midi_path, num_time_frames)
    if labels is None or not torch.any(labels):
        return None
    return labels


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


_PARSE_CHART_FN = None


def _load_chart_obj(path: Path) -> dict | None:
    """Parse a .chart file using scripts/discover_clonehero.parse_chart."""
    global _PARSE_CHART_FN
    try:
        if _PARSE_CHART_FN is None:
            import importlib.util

            scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
            src_path = scripts_dir / "discover_clonehero.py"
            if not src_path.exists():
                return None
            spec = importlib.util.spec_from_file_location(
                "discover_clonehero", str(src_path)
            )
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            _PARSE_CHART_FN = module.parse_chart  # type: ignore[attr-defined]
        return _PARSE_CHART_FN(str(path)) if _PARSE_CHART_FN else None
    except Exception:
        return None


def build_labels_from_chart(
    chart_path: Path, num_time_frames: int, config: BaseConfig
) -> Optional[torch.Tensor]:
    obj = _load_chart_obj(chart_path)
    if not obj:
        return None
    # Only use Expert drums. Lower diffs omit notes, which conflicts with the
    # same audio waveform.
    events: list[dict] = obj.get("tracks", {}).get("ExpertDrums", [])
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
) -> tuple[torch.Tensor, int, float]:
    """Estimate a small global offset (in frames) aligning labels to audio onset envelope.

    Uses librosa onset strength at the model's hop_length, searches lags within ±250 ms.
    Returns (shifted_labels, best_lag_frames, best_score), where best_score is
    a normalized correlation-like score (higher is better alignment).
    """
    try:
        hop = int(config.hop_length)
        sr = int(config.sample_rate)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        # labels: (T, C) -> sum across classes -> (T,)
        lab_sum = labels.sum(dim=1).detach().cpu().numpy().astype(np.float32)
        T = min(len(onset_env), lab_sum.shape[0])
        if T <= 1:
            return labels, 0, 0.0
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
            # normalized dot to reduce scale dependency
            xn = x / (float(np.linalg.norm(x)) + 1e-8)
            yn = yv / (float(np.linalg.norm(yv)) + 1e-8)
            score = float(np.dot(xn, yn))
            if score > best_score:
                best_score = score
                best_lag = lag
        if best_lag == 0:
            return labels, int(best_lag), float(best_score)
        # Positive lag means labels trail audio -> shift labels left by lag
        shifted = torch.zeros_like(labels)
        if best_lag > 0:
            shifted[:-best_lag, :] = labels[best_lag:, :]
        else:
            lag = -best_lag
            shifted[lag:, :] = labels[:-lag, :]
        return shifted, int(best_lag), float(best_score)
    except Exception:
        return labels, 0, 0.0


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


def _spec_phash(
    spec: torch.Tensor, target_size: Tuple[int, int] = (16, 16)
) -> Optional[str]:
    """Compute a simple perceptual hash of a spectrogram tensor."""
    arr = spec.squeeze(0).detach().cpu().numpy()
    F, T = arr.shape
    fh, th = target_size
    if F < fh or T < th:
        return None
    f_bins = np.array_split(np.arange(F), fh)
    t_bins = np.array_split(np.arange(T), th)
    small = np.zeros((fh, th), dtype=np.float32)
    for i, fb in enumerate(f_bins):
        for j, tb in enumerate(t_bins):
            small[i, j] = float(np.mean(arr[np.ix_(fb, tb)]))
    med = float(np.median(small))
    bits = (small > med).astype(np.uint8).flatten()
    by = np.packbits(bits)
    return by.tobytes().hex()


def _process_record(
    rec: Record,
    base: str,
    split_dir: Path,
    config: BaseConfig,
    args: argparse.Namespace,
    seen_hashes: Optional[multiprocessing.managers.DictProxy],
    hash_lock: Optional[multiprocessing.synchronize.Lock],
) -> Tuple[str, Optional[str], str, Optional[str]]:
    """Process a single Record into spectrogram/label pair.

    Returns a tuple of (status, domain, song_dir, error_message).
    """
    processor = SpectrogramProcessor(config)
    midi_processor = RbMidiProcessor(config)
    try:
        y, domain = load_audio_for_training(rec.song_dir, config)
        if y.size == 0:
            return "skipped", None, str(rec.song_dir), None
        y_norm = normalize_loudness_rms(y, target_dbfs=-14.0)
        y_t = torch.from_numpy(y_norm).float().unsqueeze(0)
        spec = processor.audio_to_spectrogram(y_t)
        if spec.shape[1] != config.n_mels:
            spec = spec.transpose(1, 2)
        T = int(spec.shape[-1])
        if rec.source == "midi":
            labels = build_labels_from_midi(rec.notes_path, T, midi_processor)
        else:
            labels = build_labels_from_chart(rec.notes_path, T, config)
        if labels is None:
            return "skipped", domain, str(rec.song_dir), None
        labels, lag, score = estimate_and_apply_global_offset(y, labels, config)
        if not np.isfinite(score) or score < float(args.min_align_score):
            return "desynced", domain, str(rec.song_dir), None
        dil = int(getattr(config, "label_dilation_frames", 0) or 0)
        if dil > 0:
            labels = dilate_labels_time(labels, dil)
        max_frames = int(
            round(
                (getattr(config, "max_audio_length", 0.0) or 0.0)
                * config.sample_rate
                / max(1, int(config.hop_length))
            )
        )
        segments_written = 0
        if max_frames > 0 and T > max_frames:
            for seg_idx, start in enumerate(range(0, T, max_frames)):
                end = min(start + max_frames, T)
                spec_seg = spec[..., start:end]
                labels_seg = labels[start:end, :]
                if args.dedupe:
                    try:
                        h = _spec_phash(spec_seg)
                        if h:
                            with hash_lock:
                                if h in seen_hashes:
                                    continue
                                if len(seen_hashes) >= MAX_HASHES:
                                    # Remove an arbitrary item to keep size bounded
                                    for k in list(seen_hashes.keys())[:1]:
                                        seen_hashes.pop(k)
                                        break
                                seen_hashes[h] = True
                    except Exception:
                        pass
                seg_base = f"{base}_seg{seg_idx + 1:02d}"
                save_pair(split_dir, seg_base, spec_seg.squeeze(0), labels_seg)
                segments_written += 1
        else:
            if args.dedupe:
                try:
                    h = _spec_phash(spec)
                    if h:
                        with hash_lock:
                            if h in seen_hashes:
                                return "dup_skipped", domain, str(rec.song_dir), None
                            if len(seen_hashes) >= MAX_HASHES:
                                # Remove an arbitrary item to keep size bounded
                                for k in list(seen_hashes.keys())[:1]:
                                    seen_hashes.pop(k)
                                    break
                                seen_hashes[h] = True
                except Exception:
                    pass
            save_pair(split_dir, base, spec.squeeze(0), labels)
            segments_written = 1
        if segments_written == 0:
            return "dup_skipped", domain, str(rec.song_dir), None
        return "written", domain, str(rec.song_dir), None
    except Exception as e:
        return "error", None, str(rec.song_dir), str(e)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build high-res dataset from Clone Hero songs"
    )
    ap.add_argument("--roots", type=str, nargs="+", default=["CloneHero/Songs"])
    ap.add_argument("--out-dir", type=str, default="datasets/processed_highres")
    ap.add_argument("--config", type=str, default="local_highres")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--splits", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    ap.add_argument(
        "--json-index-dir",
        type=str,
        default=None,
        help=(
            "Optional directory of JSON chart entries (artifacts). When provided, restrict candidates to those paths."
        ),
    )
    ap.add_argument(
        "--limit-songs",
        type=int,
        default=None,
        help="Pick at most this many candidate songs using simple heuristics (prefer MIDI + stems).",
    )
    ap.add_argument(
        "--max-per-pack",
        type=int,
        default=None,
        help="Cap selected songs per pack (default ≈ limit/5).",
    )
    ap.add_argument(
        "--max-per-artist",
        type=int,
        default=2,
        help="Cap selected songs per artist (default 2).",
    )
    ap.add_argument(
        "--prefer-non-rbgh",
        type=float,
        default=0.3,
        help="Bonus weight for non Rock Band/Guitar Hero sources (0..1 recommended).",
    )
    ap.add_argument(
        "--bonus-pack-keywords",
        type=str,
        default="J-Rock,jrock,J-Pop,Anime,Alt Quest,Villainess",
        help="Comma-separated pack keywords to boost (case-insensitive).",
    )
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--discovery-cache",
        type=str,
        default=None,
        help=(
            "Optional path to cache discovered song folders as JSON for faster subsequent runs. "
            "Defaults to <out-dir>/discovery_cache.json if not provided."
        ),
    )
    ap.add_argument(
        "--refresh-discovery",
        action="store_true",
        help="Ignore any existing discovery cache and rescan roots.",
    )
    ap.add_argument(
        "--min-align-score",
        type=float,
        default=0.05,
        help="Skip charts whose global onset alignment score is below this threshold",
    )
    ap.add_argument(
        "--dedupe",
        action="store_true",
        help="Skip likely duplicate audio by simple perceptual hashing of spectrogram",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip songs that already have processed outputs in the destination",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel worker processes",
    )
    args = ap.parse_args()

    config = get_config(args.config)

    # Discovery with optional caching
    cache_path: Optional[Path]
    if args.discovery_cache:
        cache_path = Path(args.discovery_cache)
    else:
        cache_path = Path(args.out_dir) / "discovery_cache.json"

    folders: List[Path]
    use_cache = False
    if (not args.refresh_discovery) and cache_path and cache_path.exists():
        try:
            import json as _json

            payload = _json.loads(cache_path.read_text())
            roots_cached = payload.get("roots")
            if isinstance(roots_cached, list) and all(
                isinstance(x, str) for x in roots_cached
            ):
                if set(map(str, args.roots)) == set(roots_cached):
                    folders = [
                        Path(x)
                        for x in payload.get("folders", [])
                        if isinstance(x, str)
                    ]
                    use_cache = bool(folders)
        except Exception:
            use_cache = False
    if not use_cache:
        folders = find_song_folders(args.roots)
        # Persist cache for next run
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            import json as _json

            cache_path.write_text(
                _json.dumps(
                    {
                        "roots": list(map(str, args.roots)),
                        "folders": [str(x) for x in folders],
                    },
                    indent=2,
                )
            )
        except Exception:
            pass
    if not folders:
        print("No Clone Hero songs found.")
        return
    # Optional: restrict to folders referenced by JSON index
    if args.json_index_dir:
        json_root = Path(args.json_index_dir)
        if json_root.exists():
            folders = [
                d for d in folders if json_index_has_song(d, args.roots, json_root)
            ]
    if args.max_files is not None and len(folders) > args.max_files:
        rng = np.random.default_rng(seed=args.seed)
        idx = rng.choice(len(folders), size=args.max_files, replace=False)
        folders = [folders[int(i)] for i in sorted(idx)]
    print(f"Found {len(folders)} candidate song folders")

    # Collect records (prefer pro_drums charts when appropriate; Expert-only later)
    recs: List[Record] = []
    for d in folders:
        midi = d / "notes.mid"
        chart = d / "notes.chart"
        txt = d / "notes.txt"
        meta = parse_song_ini(d)
        pro_drums_flag = meta.get("pro_drums", "").strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
            "on",
        }
        # If a chart exists and pro_drums is indicated, prefer the chart; otherwise prefer MIDI
        if chart.exists() and pro_drums_flag:
            recs.append(Record(song_dir=d, notes_path=chart, source="chart"))
            continue
        if midi.exists():
            recs.append(Record(song_dir=d, notes_path=midi, source="midi"))
            continue
        # Fallback to chart/txt if present
        if chart.exists():
            recs.append(Record(song_dir=d, notes_path=chart, source="chart"))
            continue
        if txt.exists():
            recs.append(Record(song_dir=d, notes_path=txt, source="chart"))
    # If requested, choose a limited number of songs guided by simple audio/chart heuristics
    if args.limit_songs is not None and len(recs) > args.limit_songs:
        # Diversity-aware scoring and sampling
        kw = [
            k.strip().lower()
            for k in (args.bonus_pack_keywords or "").split(",")
            if k.strip()
        ]

        def _meta(r: Record) -> dict[str, str]:
            m = parse_song_ini(r.song_dir)
            # Derive artist/name from folder if missing
            if not m.get("artist") or not m.get("name"):
                base = r.song_dir.name
                if " - " in base:
                    parts = base.split(" - ", 1)
                    if not m.get("artist"):
                        m["artist"] = parts[0]
                    if not m.get("name"):
                        m["name"] = parts[1]
            m.setdefault("artist", "?")
            m.setdefault("name", r.song_dir.name)
            m.setdefault("genre", "")
            return m

        def _score_record(r: Record) -> float:
            score = 0.0
            if r.source == "midi":
                score += 2.0
            stems = list_stems(r.song_dir)
            drum_keys = [k for k in stems.keys() if "drum" in k]
            other_keys = [
                k
                for k in stems.keys()
                if any(
                    w in k
                    for w in ("guitar", "bass", "vocals", "rhythm", "keys", "backing")
                )
            ]
            if drum_keys and other_keys:
                score += 3.0
            elif drum_keys:
                score += 1.0
            # Prefer Expert when detectable from path/name
            pstr = str(r.notes_path).lower()
            if "expert" in pstr:
                score += 0.5
            # Prefer charts with explicit pro_drums flag in song.ini
            meta_r = _meta(r)
            if meta_r.get("pro_drums", "").strip().lower() in {
                "true",
                "1",
                "yes",
                "y",
                "on",
            }:
                score += 0.75
            # Penalize if no obvious audio (rare)
            if choose_audio_path(r.song_dir) is None and not drum_keys:
                score -= 2.0
            # Non RB/GH bonus
            path_l = str(r.song_dir).lower()
            is_rbgh = ("rock band" in path_l) or ("guitar hero" in path_l)
            if not is_rbgh:
                score += float(args.prefer_non_rbgh or 0.0)
            else:
                score -= 0.1
            # Pack/genre bonuses
            pack = _pack_name_for(r.song_dir, args.roots)
            pack_l = pack.lower()
            if any(k in pack_l for k in kw):
                score += 0.7
            # Genre diversity nudge
            genre = _meta(r).get("genre", "").lower()
            if genre:
                if any(g in genre for g in ("j-rock", "jrock", "j-pop", "anime")):
                    score += 0.6
                elif any(g in genre for g in ("pop", "electro", "edm", "hip", "dance")):
                    score += 0.2
            return score

        # Compute weights and metadata
        infos = []
        for r in recs:
            m = _meta(r)
            pack = _pack_name_for(r.song_dir, args.roots)
            w = max(0.01, _score_record(r))
            infos.append((r, w, pack, m.get("artist", "?")))

        # Weighted sampling with diversity caps
        rng = np.random.default_rng(seed=args.seed)
        weights = np.array([x[1] for x in infos], dtype=np.float64)
        probs = weights / weights.sum()
        order = rng.choice(len(infos), size=len(infos), replace=False, p=probs)

        max_per_pack = (
            int(args.max_per_pack)
            if args.max_per_pack is not None
            else max(1, int(round((args.limit_songs or 25) / 5)))
        )
        max_per_artist = int(args.max_per_artist or 2)
        by_pack: dict[str, int] = {}
        by_artist: dict[str, int] = {}
        chosen: List[Record] = []

        for idx in order:
            r, _w, pack, artist = infos[idx]
            if by_pack.get(pack, 0) >= max_per_pack:
                continue
            if by_artist.get(artist, 0) >= max_per_artist:
                continue
            chosen.append(r)
            by_pack[pack] = by_pack.get(pack, 0) + 1
            by_artist[artist] = by_artist.get(artist, 0) + 1
            if len(chosen) >= int(args.limit_songs):
                break
        # Fallback to fill if caps were too strict
        if len(chosen) < int(args.limit_songs):
            for idx in order:
                if len(chosen) >= int(args.limit_songs):
                    break
                r, _w, pack, artist = infos[idx]
                if r in chosen:
                    continue
                chosen.append(r)
        recs = chosen

        # Print a short summary grouped by pack → artist → song
        try:
            from collections import defaultdict

            tree: dict[str, dict[str, list[str]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for r in recs:
                meta = parse_song_ini(r.song_dir)
                artist = meta.get("artist", r.song_dir.parent.name)
                name = meta.get("name", r.song_dir.name)
                pack = _pack_name_for(r.song_dir, args.roots)
                tree[pack][artist].append(name)
            print("Selected songs (pack → artist → song):")
            for pack in sorted(tree.keys()):
                print(f"- {pack} ({len(tree[pack])} artists)")
                for artist in sorted(tree[pack].keys()):
                    songs = ", ".join(sorted(tree[pack][artist])[:5])
                    extra = len(tree[pack][artist]) - 5
                    tail = f" (+{extra})" if extra > 0 else ""
                    print(f"    · {artist}: {songs}{tail}")
        except Exception:
            pass

    if not recs:
        print("No usable charts found (notes.mid / notes.chart / notes.txt).")
        return

    # Group-aware split by (artist|name|charter) to avoid leakage
    meta = [parse_song_ini(r.song_dir) for r in recs]
    groups: Dict[str, List[int]] = {}
    for i, r in enumerate(recs):
        m = meta[i]
        key = f"{m.get('artist', '?')}|{m.get('name', '?')}|{m.get('charter', '?')}"
        groups.setdefault(key, []).append(i)
    rng = np.random.default_rng(seed=args.seed)
    keys = list(groups.keys())
    rng.shuffle(keys)
    n_groups = len(keys)
    splits = np.array(args.splits, dtype=np.float64)
    if splits.sum() <= 0:
        splits = np.array([0.8, 0.1, 0.1], dtype=np.float64)
    splits = splits / splits.sum()
    raw = splits * n_groups
    counts = np.floor(raw).astype(int)
    remainder = n_groups - int(counts.sum())
    if remainder > 0:
        order = np.argsort(raw - counts)[::-1]
        for idx in order[:remainder]:
            counts[idx] += 1
    g_train, g_val, _g_test = counts.tolist()
    key_train = set(keys[:g_train])
    key_val = set(keys[g_train : g_train + g_val])
    idx_train = {i for k in key_train for i in groups[k]}
    idx_val = {i for k in key_val for i in groups[k]}

    out_root = Path(args.out_dir)
    out_train = out_root / "train"
    out_val = out_root / "val"
    out_test = out_root / "test"
    for p in (out_train, out_val, out_test):
        p.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    desynced = 0
    dup_skipped = 0
    existing = 0
    domain_counts: Dict[str, int] = {}

    manager = multiprocessing.Manager() if args.dedupe else None
    seen_hashes = manager.dict() if manager else None
    hash_lock = manager.Lock() if manager else None

    futures = []
    with ProcessPoolExecutor(max_workers=int(args.workers or os.cpu_count())) as ex:
        for i, rec in enumerate(recs):
            base = f"{rec.song_dir.parent.name}_{rec.song_dir.name}_{i:06d}"
            split_dir = (
                out_train if i in idx_train else (out_val if i in idx_val else out_test)
            )
            spec_path = split_dir / f"{base}_mel.npy"
            lab_path = split_dir / f"{base}_label.npy"
            if args.resume and spec_path.exists() and lab_path.exists():
                existing += 1
                continue
            futures.append(
                ex.submit(
                    _process_record,
                    rec,
                    base,
                    split_dir,
                    config,
                    args,
                    seen_hashes,
                    hash_lock,
                )
            )

        for fut in as_completed(futures):
            status, domain, song_dir, err = fut.result()
            if domain:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            if status == "written":
                written += 1
            elif status == "skipped":
                skipped += 1
            elif status == "desynced":
                desynced += 1
            elif status == "dup_skipped":
                dup_skipped += 1
            elif status == "error":
                skipped += 1
                print(f"ERR processing {song_dir}: {err}")

    print(
        f"Done. Wrote {written} examples ({out_root}). Skipped {skipped} missing, {desynced} desynced, {dup_skipped} duplicates, {existing} existing."
    )
    if domain_counts:
        print("Domain distribution:")
        for k, v in sorted(domain_counts.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
