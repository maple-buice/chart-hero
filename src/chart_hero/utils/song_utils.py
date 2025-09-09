"""Utilities for working with Clone Hero style song folders.

This module centralizes simple helpers for locating and combining audio
stems that may exist within a song directory.  The logic was previously
embedded inside the dataset builder but is now shared so that evaluation
code can construct a full mix directly from a song folder.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .audio_io import load_audio

# Common audio extensions we expect to see inside Clone Hero song folders
AUDIO_EXTS = (".ogg", ".mp3", ".wav", ".flac", ".m4a")


def list_stems(song_dir: Path) -> Dict[str, Path]:
    """Return a mapping of stem name -> path for audio files in ``song_dir``."""

    stems: Dict[str, Path] = {}
    for fn in os.listdir(song_dir):
        low = fn.lower()
        if not any(low.endswith(ext) for ext in AUDIO_EXTS):
            continue
        stems[low.split(".")[0]] = song_dir / fn
    return stems


def mix_stems_to_waveform(song_dir: Path, sr: int) -> Optional[np.ndarray]:
    """Combine per-instrument stems into a single mono mix.

    Looks for at least one drum stem and at least one non-drum stem.  If both
    are present, they are loaded, padded to the same length, summed and scaled
    to avoid clipping.  Returns ``None`` when a suitable combination of stems
    cannot be found.
    """

    stems = list_stems(song_dir)
    drum_keys = [k for k in stems.keys() if "drum" in k]
    other_keys = [
        k
        for k in stems.keys()
        if any(
            w in k for w in ("guitar", "bass", "vocals", "vocal", "rhythm", "keys", "backing")
        )
    ]
    if not drum_keys or not other_keys:
        return None

    tracks: List[np.ndarray] = []
    max_len = 0
    for k in drum_keys + other_keys:
        try:
            y, _ = load_audio(stems[k], sr=sr)
            y = y.astype(np.float32)
            tracks.append(y)
            max_len = max(max_len, len(y))
        except Exception:
            continue
    if not tracks:
        return None

    mix = np.zeros(max_len, dtype=np.float32)
    for t in tracks:
        if len(t) < max_len:
            t = np.pad(t, (0, max_len - len(t)))
        mix += t

    peak = float(np.max(np.abs(mix))) if mix.size > 0 else 1.0
    if peak > 0:
        mix = 0.5 * mix / peak  # -6 dB headroom
    return mix


def choose_audio_path(song_dir: Path) -> Optional[Path]:
    """Best-effort selection of a pre-mixed audio file in ``song_dir``.

    Prefers ``song.ogg`` when present, otherwise looks for filenames that do
    not resemble isolated stems.  Falls back to the first audio file it finds.
    """

    p = song_dir / "song.ogg"
    if p.exists():
        return p
    mix_keywords = ("song", "mix", "full")
    stem_keywords = (
        "guitar",
        "bass",
        "drum",
        "vocals",
        "vocal",
        "keys",
        "rhythm",
    )
    candidates: List[Path] = []
    for fn in os.listdir(song_dir):
        if Path(fn).suffix.lower() not in AUDIO_EXTS:
            continue
        low = fn.lower()
        pth = song_dir / fn
        if any(k in low for k in mix_keywords) and not any(k in low for k in stem_keywords):
            return pth
        if not any(k in low for k in stem_keywords):
            candidates.append(pth)
    if candidates:
        return candidates[0]
    for fn in os.listdir(song_dir):
        if Path(fn).suffix.lower() in AUDIO_EXTS:
            return song_dir / fn
    return None


def save_eval_song_copy(
    song_dir: Path,
    dest_root: Path,
    prediction_rows: List[Dict[str, int]],
    *,
    bpm: float,
    ppq: int,
    sr: int,
) -> Path:
    """Copy ``song_dir`` to ``dest_root`` and replace chart with predictions.

    Creates ``dest_root / song_dir.name`` if needed, writes a new ``notes.mid``
    from ``prediction_rows``, and prefixes the song title in ``song.ini`` with
    ``"[EVAL]"`` so that the folder can be loaded in Clone Hero for manual
    testing.
    """

    from chart_hero.inference.mid_export import write_notes_mid

    dest_root.mkdir(parents=True, exist_ok=True)
    out_dir = dest_root / song_dir.name
    shutil.copytree(song_dir, out_dir, dirs_exist_ok=True)

    # Remove any existing chart files before writing our predictions
    for fn in ("notes.mid", "notes.chart", "notes.txt"):
        p = out_dir / fn
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    write_notes_mid(
        out_dir,
        bpm=bpm,
        ppq=ppq,
        sr=sr,
        prediction_rows=prediction_rows,
    )

    ini_path = out_dir / "song.ini"
    name = None
    lines: List[str]
    if ini_path.exists():
        lines = ini_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = ["[Song]"]
    updated: List[str] = []
    for ln in lines:
        if ln.strip().lower().startswith("name ="):
            name = ln.split("=", 1)[1].strip()
            updated.append(f"name = [EVAL] {name}")
        else:
            updated.append(ln)
    if name is None:
        name = song_dir.name
        updated.append(f"name = [EVAL] {name}")
    ini_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return out_dir

