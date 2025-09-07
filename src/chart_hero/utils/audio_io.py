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
                audio = audio.reshape(-1, channels).T
            return audio, target_sr
        except Exception:
            # Fall through to librosa
            pass

    # Fallback: librosa.load (may emit deprecation warnings if it uses audioread)
    y, s = librosa.load(path_str, sr=target_sr if sr is not None else None, mono=mono)
    return y, int(s)


def get_duration(path: str | Path, tol: float = 1e-2) -> float:
    """
    Get duration in seconds. Prefer ffprobe when available; cross-check against librosa
    within ``tol`` seconds. Falls back to librosa if ffprobe is unavailable or fails.

    Raises ``RuntimeError`` if duration cannot be determined (e.g., non-audio files) or
    when ffprobe and librosa disagree beyond ``tol``.
    """
    path_str = str(path)
    ffprobe_dur: Optional[float] = None
    if has_ffprobe():
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
        try:
            out = subprocess.check_output(cmd, text=True).strip()
            ffprobe_dur = float(out)
        except Exception:
            ffprobe_dur = None
        if ffprobe_dur is not None:
            try:
                librosa_dur = float(librosa.get_duration(path=path_str))
            except Exception:
                return ffprobe_dur
            if abs(ffprobe_dur - librosa_dur) > tol:
                raise RuntimeError(
                    f"Duration mismatch: ffprobe={ffprobe_dur} vs librosa={librosa_dur}"
                )
            return ffprobe_dur

    try:
        return float(librosa.get_duration(path=path_str))
    except Exception as e:
        raise RuntimeError(f"Could not determine duration for {path_str}") from e
