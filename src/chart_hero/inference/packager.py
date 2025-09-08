from __future__ import annotations

import re
import subprocess
from pathlib import Path
from shutil import which
from typing import Optional

import soundfile as sf

from chart_hero.utils.audio_io import get_duration, load_audio

from .chart_writer import SongMeta, write_chart
from .types import PredictionRow


def sanitize_name(name: str) -> str:
    # Remove path-unsafe chars
    return re.sub(r"[\\/:*?\"<>|]", "_", name).strip()


def convert_to_ogg(src: Path, dst: Path, target_sr: int = 44100) -> tuple[Path, float]:
    """
    Convert an audio file to OGG Vorbis.

    Preference order:
    1) ffmpeg CLI (robust, avoids libsndfile crashes)
    2) librosa + soundfile fallback

    Returns: (output_path, duration_seconds)
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Prefer ffmpeg to avoid potential libsndfile-related segfaults
    if which("ffmpeg") is not None:
        try:
            cmd = [
                "ffmpeg",
                "-y",  # overwrite
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-vn",  # no video
                "-ac",
                "2",
                "-ar",
                str(target_sr),
                "-c:a",
                "libvorbis",
                str(dst),
            ]
            subprocess.run(cmd, check=True)
            info = sf.info(str(dst))
            return dst, float(info.duration)
        except Exception:
            # fall back below
            pass

    # Fallback: librosa read + soundfile write
    y, _ = load_audio(str(src), sr=target_sr, mono=True)
    sf.write(str(dst), y, target_sr, format="OGG", subtype="VORBIS")
    return dst, float(len(y) / target_sr)


def package_clonehero_song(
    clonehero_root: Path,
    title: str,
    artist: Optional[str],
    bpm: float,
    resolution: int,
    sr_model: int,
    prediction_rows: list[PredictionRow],
    source_audio: Path,
    album_path: Optional[Path] = None,
    background_path: Optional[Path] = None,
    convert_audio: bool = True,
    write_chart: bool = False,
) -> Path:
    """
    Create a Clone Hero song folder with song.ini, audio, and art.
    Optionally writes a legacy notes.chart if write_chart=True; default flow
    is to emit notes.mid elsewhere and only use this to prep the folder.
    Returns the path to the created song folder.
    """
    pack_root = clonehero_root / "Songs" / "Chart Hero"
    pack_root.mkdir(parents=True, exist_ok=True)

    folder_name = sanitize_name(
        f"{artist} - {title} [chart-hero]" if artist else f"{title} [chart-hero]"
    )
    out_dir = pack_root / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Audio
    if convert_audio:
        music_file = out_dir / "song.ogg"
        music_path, dur_sec = convert_to_ogg(source_audio, music_file)
    else:
        # Just reference original (not recommended for distribution)
        music_file = out_dir / source_audio.name
        if source_audio.resolve() != music_file.resolve():
            music_file.write_bytes(source_audio.read_bytes())
        dur_sec = get_duration(str(music_file))

    # Ensure song.ini exists (for either .chart or .mid flows)
    ini_path = out_dir / "song.ini"
    if not ini_path.exists():
        ini_lines = [
            "[Song]",
            f"name = {title}",
            f"artist = {artist or ''}",
            f"charter = chart-hero",
            f"genre = Unknown",
            f"year = ",
            f"diff_drums = 3",
            f"pro_drums = True",
        ]
        ini_path.write_text("\n".join(ini_lines) + "\n", encoding="utf-8")

    # Optionally write a .chart (legacy path). Default is False now that we use notes.mid.
    if write_chart:
        meta = SongMeta(name=title, artist=artist or None, charter="chart-hero")
        write_chart(
            out_dir,
            meta,
            bpm=bpm,
            resolution=resolution,
            sr=sr_model,
            prediction_rows=prediction_rows,
            music_stream=music_file.name,
        )

    # Add song.ini extras (background/banner) by appending keys if files present
    if album_path and album_path.exists():
        (out_dir / "album.png").write_bytes(album_path.read_bytes())
    if background_path and background_path.exists():
        (out_dir / "background.jpg").write_bytes(background_path.read_bytes())

    # Optionally update song.ini with length and art hints
    try:
        lines = ini_path.read_text(encoding="utf-8").splitlines()
        # Insert/replace song_length (ms)
        ms = int(round(dur_sec * 1000))
        updated = []
        have_len = False
        have_bg = False
        for ln in lines:
            if ln.lower().startswith("song_length ="):
                updated.append(f"song_length = {ms}")
                have_len = True
            elif ln.lower().startswith("background ="):
                updated.append("background = background.jpg")
                have_bg = True
            else:
                updated.append(ln)
        if not have_len:
            updated.append(f"song_length = {ms}")
        if background_path and not have_bg:
            updated.append("background = background.jpg")
        ini_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    except Exception:
        pass

    return out_dir
