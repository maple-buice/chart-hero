import json
import os
import re
import shutil
import time
from typing import Any, Dict, List, cast

import librosa
import numpy as np
from numpy.typing import NDArray
from yt_dlp import YoutubeDL

from .types import Segment, TransformerConfig


def create_transient_enhanced_spectrogram(
    y: NDArray[np.floating],
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> NDArray[np.floating]:
    """
    Creates a mel spectrogram where transients are enhanced.
    This function MUST be identical to the one in data_preparation.py
    """
    # 1. Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # 2. Onset Strength Envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # 3. Align and Gate
    min_len = min(log_mel_spec.shape[1], len(onset_env))
    log_mel_spec = log_mel_spec[:, :min_len]
    onset_env = onset_env[:min_len]

    # Normalize onset envelope to [0, 1]
    if np.max(onset_env) > 0:
        onset_env = onset_env / np.max(onset_env)

    # Gate the spectrogram
    transient_enhanced_spec = log_mel_spec * onset_env
    return cast(NDArray[np.floating], transient_enhanced_spec)


def audio_to_tensors(audio_path: str, config: TransformerConfig) -> list[Segment]:
    """
    Transforms an audio file into a list of tensor segments for the model.
    """
    try:
        y, sr_f = librosa.load(audio_path, sr=config.sample_rate)
        sr: int = int(sr_f)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return []

    # Create the full transient-enhanced spectrogram
    full_spec = create_transient_enhanced_spectrogram(
        y=y,
        sr=sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
    )

    # Segment the spectrogram into chunks the model can handle
    segment_length_frames = int(
        config.max_audio_length * config.sample_rate / config.hop_length
    )

    segments: list[Segment] = []
    num_frames = full_spec.shape[1]
    for i in range(0, num_frames, segment_length_frames):
        end_frame = i + segment_length_frames
        if end_frame > num_frames:
            spec_segment = np.pad(
                full_spec[:, i:],
                ((0, 0), (0, end_frame - num_frames)),
                mode="constant",
                constant_values=np.min(full_spec),
            )
        else:
            spec_segment = full_spec[:, i:end_frame]

        # Keep numpy for now; Charter.predict will convert to torch and batch
        segments.append(
            {
                "spec": spec_segment,  # shape: (n_mels, frames)
                "start_frame": i,
                "end_frame": min(end_frame, num_frames),
                "total_frames": num_frames,
            }
        )

    return segments


class yt_audio:
    path: str
    title: str
    description: str
    thumbnail_url: str
    from_cache: bool
    temp_dir: str | None

    def __init__(
        self,
        path: str,
        title: str,
        description: str,
        thumbnail_url: str,
        from_cache: bool = False,
        temp_dir: str | None = None,
    ):
        self.path = path
        self.title = title
        self.description = description
        self.thumbnail_url = thumbnail_url
        self.from_cache = from_cache
        self.temp_dir = temp_dir


def _cookies_opts_from_env() -> dict:
    """Build yt-dlp cookie options from env if present.

    Recognized env:
    - YTDLP_COOKIES_FROM_BROWSER: e.g., 'chrome', 'firefox', 'edge'
    - YTDLP_COOKIEFILE: path to a cookies.txt
    """
    opts: dict = {}
    try:
        cfb = os.environ.get("YTDLP_COOKIES_FROM_BROWSER")
        if cfb:
            # yt-dlp expects a tuple like ("chrome",)
            opts["cookiesfrombrowser"] = (cfb.strip(),)
        cfile = os.environ.get("YTDLP_COOKIEFILE")
        if cfile:
            opts["cookiefile"] = cfile.strip()
    except Exception:
        pass
    return opts


def get_yt_audio(link: str, no_cache: bool = False) -> yt_audio | None:
    base_dir = "music/YouTube"
    index_path = os.path.join(base_dir, "cache_index.json")
    os.makedirs(base_dir, exist_ok=True)

    def _load_index() -> Dict[str, Any]:
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {"version": 1, "entries": []}
            data.setdefault("version", 1)
            data.setdefault("entries", [])
            if not isinstance(data["entries"], list):
                data["entries"] = []
            return data
        except Exception:
            return {"version": 1, "entries": []}

    def _save_index(data: Dict[str, Any]) -> None:
        try:
            tmp = index_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, index_path)
        except Exception:
            pass

    def _prune_missing(data: Dict[str, Any]) -> Dict[str, Any]:
        entries: List[Dict[str, Any]] = []
        for e in data.get("entries", []):
            d = e.get("dir")
            if (
                isinstance(d, str)
                and os.path.isdir(d)
                and os.path.exists(os.path.join(d, e.get("file", "song.m4a")))
            ):
                entries.append(e)
        data["entries"] = entries
        return data

    def _enforce_limit(data: Dict[str, Any], limit: int = 10) -> Dict[str, Any]:
        entries: List[Dict[str, Any]] = list(data.get("entries", []))
        if len(entries) <= limit:
            return data
        # Sort by last_access ascending
        entries.sort(key=lambda e: float(e.get("last_access", 0.0)))
        to_delete = entries[:-limit]
        keep = entries[-limit:]
        for e in to_delete:
            d = e.get("dir")
            try:
                if isinstance(d, str) and os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
        data["entries"] = keep
        return data

    def _sanitize(name: str) -> str:
        # Keep simple, filesystem-safe names
        name = re.sub(r"[\\/:*?\"<>|]", "_", name)
        name = re.sub(r"\s+", " ", name).strip()
        return name[:80]

    try:
        # First fetch metadata without downloading to get a stable ID
        with YoutubeDL({"quiet": True}) as ydl:
            info = ydl.extract_info(link, download=False)
        if not info:
            return None
        vid = str(info.get("id", "unknown"))
        title = info.get("title", "Unknown Title")
        desc = info.get("description", "")
        thumb = info.get("thumbnail", None)

        # If no-cache requested: download to a temp dir and return without touching index
        if no_cache:
            # Create a temporary subfolder per run
            folder = f"tmp_{vid}_{int(time.time())}_{_sanitize(title)}"
            song_dir = os.path.join(base_dir, folder)
            os.makedirs(song_dir, exist_ok=True)
            target_path = os.path.join(song_dir, "song.m4a")

            ydl_opts = {
                "format": "bestaudio[acodec^=mp4a][ext=m4a]/bestaudio[ext=m4a]/bestaudio/best",
                "outtmpl": os.path.join(song_dir, "song.%(ext)s"),
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "m4a",
                    }
                ],
                "overwrites": True,
                "quiet": True,
                "noprogress": True,
            }
            ydl_opts.update(_cookies_opts_from_env())
            with YoutubeDL(ydl_opts) as ydl:
                info2 = ydl.extract_info(link, download=True)
                title = info2.get("title", title)
                desc = info2.get("description", desc)
                thumb = info2.get("thumbnail", thumb)
            return yt_audio(
                path=target_path,
                title=title,
                description=desc,
                thumbnail_url=thumb,
                from_cache=False,
                temp_dir=song_dir,
            )

        # Else: cache-enabled flow. Load and sanitize cache index first
        index = _load_index()
        index = _prune_missing(index)

        # Try cache hit by video id
        now = time.time()
        for e in index.get("entries", []):
            if e.get("id") == vid:
                d = e.get("dir")
                f = e.get("file", "song.m4a")
                full = os.path.join(d, f) if isinstance(d, str) else None
                if full and os.path.exists(full):
                    # Update last_access and persist
                    e["last_access"] = now
                    _save_index(index)
                    return yt_audio(
                        path=full,
                        title=title,
                        description=desc,
                        thumbnail_url=thumb,
                        from_cache=True,
                    )

        # Cache miss: prepare folder and download
        folder = f"{vid}_{_sanitize(title)}"
        song_dir = os.path.join(base_dir, folder)
        os.makedirs(song_dir, exist_ok=True)
        target_path = os.path.join(song_dir, "song.m4a")

        ydl_opts = {
            "format": "bestaudio[acodec^=mp4a][ext=m4a]/bestaudio[ext=m4a]/bestaudio/best",
            "outtmpl": os.path.join(song_dir, "song.%(ext)s"),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "m4a",
                }
            ],
            "overwrites": False,
            "quiet": True,
            "noprogress": True,
        }
        ydl_opts.update(_cookies_opts_from_env())
        with YoutubeDL(ydl_opts) as ydl:
            info2 = ydl.extract_info(link, download=True)
            # Prefer updated metadata if available
            title = info2.get("title", title)
            desc = info2.get("description", desc)
            thumb = info2.get("thumbnail", thumb)

        # Update index with new entry
        size = 0
        try:
            size = os.path.getsize(target_path)
        except Exception:
            pass
        entry = {
            "id": vid,
            "dir": song_dir,
            "file": "song.m4a",
            "title": title,
            "last_access": now,
            "size_bytes": int(size),
        }
        entries_list: List[Dict[str, Any]] = index.get("entries", [])
        # If any stale entries for same id exist, replace
        entries_list = [e for e in entries_list if e.get("id") != vid]
        entries_list.append(entry)
        index["entries"] = entries_list

        # Enforce limit and persist index
        index = _enforce_limit(index, limit=10)
        _save_index(index)

        return yt_audio(
            path=target_path,
            title=title,
            description=desc,
            thumbnail_url=thumb,
            from_cache=False,
        )
    except Exception:
        return None
