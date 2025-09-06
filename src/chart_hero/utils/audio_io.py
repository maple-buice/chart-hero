from __future__ import annotations

import subprocess
from pathlib import Path
from shutil import which
from typing import Optional, Tuple

import numpy as np
import librosa


def has_ffmpeg() -> bool:
    return which("ffmpeg") is not None


def has_ffprobe() -> bool:
    return which("ffprobe") is not None


def load_audio(
    path: str | Path, sr: Optional[int] = None, mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio using ffmpeg when available to avoid librosa's deprecated audioread path.
    Falls back to librosa.load if ffmpeg is unavailable or fails.

    - path: input file path
    - sr: target sample rate; if None and ffmpeg is used, defaults to 44100
    - mono: if True, downmix to 1 channel
    Returns (y, sr)
    """
    path_str = str(path)
    target_sr = int(sr) if sr is not None else 44100

    if has_ffmpeg():
        try:
            channels = 1 if mono else 2
            cmd = [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                path_str,
                "-f",
                "f32le",
                "-ac",
                str(channels),
                "-ar",
                str(target_sr),
                "-vn",
                "-sn",
                "-nostdin",
                "pipe:1",
            ]
            proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
            audio = np.frombuffer(proc.stdout, dtype=np.float32)
            if channels > 1:
                audio = audio.reshape(-1)
            return audio, target_sr
        except Exception:
            # Fall through to librosa
            pass

    # Fallback: librosa.load (may emit deprecation warnings if it uses audioread)
    y, s = librosa.load(path_str, sr=target_sr if sr is not None else None, mono=mono)
    return y, int(s)


def get_duration(path: str | Path) -> float:
    """
    Get duration in seconds. Prefer ffprobe when available; fallback to librosa.get_duration.
    """
    path_str = str(path)
    if has_ffprobe():
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                path_str,
            ]
            out = subprocess.check_output(cmd, text=True).strip()
            return float(out)
        except Exception:
            pass
    # Fallback
    return float(librosa.get_duration(path=path_str))
