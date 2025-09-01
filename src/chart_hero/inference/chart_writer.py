from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa

from .types import PredictionRow


@dataclass
class SongMeta:
    name: str
    artist: str | None = None
    album: str | None = None
    year: str | int | None = None
    charter: str | None = "chart-hero"


def seconds_to_ticks(seconds: float, bpm: float, resolution: int) -> int:
    # ticks = seconds * (resolution * bpm / 60)
    return int(round(seconds * (resolution * bpm / 60.0)))


def write_chart(
    out_dir: Path,
    meta: SongMeta,
    bpm: float,
    resolution: int,
    sr: int,
    prediction_rows: Iterable["PredictionRow"],
    music_stream: str | None = None,
) -> None:
    """
    Write a minimal Clone Hero-compatible .chart for ExpertDrums using Moonscraper-compatible encoding.

    prediction_rows: iterable of dicts with keys including 'peak_sample' and
    drum class labels like '0','1','2','3','4','66','67','68'.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    name = meta.name
    notes_lines: list[str] = []

    # [Song]
    song_lines = ["[Song]", "{"]
    song_lines.append(f'  Name = "{name}"')
    if meta.artist:
        song_lines.append(f'  Artist = "{meta.artist}"')
    if meta.charter:
        song_lines.append(f'  Charter = "{meta.charter}"')
    if meta.album:
        song_lines.append(f'  Album = "{meta.album}"')
    if meta.year is not None:
        song_lines.append(f'  Year = "{meta.year}"')
    song_lines.append(f"  Offset = 0")
    song_lines.append(f"  Resolution = {resolution}")
    if music_stream:
        song_lines.append(f'  MusicStream = "{music_stream}"')
    song_lines.append("}")

    # [SyncTrack]
    us_per_qn = int(round(60000000.0 / bpm))
    sync_lines = ["[SyncTrack]", "{"]
    sync_lines.append("  0 = TS 4")
    sync_lines.append(f"  0 = B {us_per_qn}")
    sync_lines.append("}")

    # [Events]
    events_lines = ["[Events]", "{", '  0 = E "section Intro"', "}"]

    # [ExpertDrums]
    track_lines = ["[ExpertDrums]", "{"]

    def add_N(tick: int, code: int, length: int = 0) -> None:
        track_lines.append(f"  {tick} = N {code} {length}")

    # mapping for classes -> (base lane, cymbal_flag_or_None)
    # Encodings informed by Moonscraper Chart Editor (BSD-3-Clause), see THIRD_PARTY_NOTICES.md
    # Map string class labels to base lane and optional cymbal flag
    CLASS_TO_NOTES: dict[str, list[tuple[int, int | None]]] = {
        "0": [(0, None)],  # kick
        "1": [(1, None)],  # snare
        "2": [(2, None)],  # hi tom (Y pad)
        "3": [(3, None)],  # mid tom (B pad)
        "4": [(4, None)],  # low/green pad (4-lane)
        "66": [(2, 66)],  # yellow cymbal
        "67": [(3, 67)],  # blue cymbal
        "68": [(4, 68)],  # green cymbal
    }

    for row in prediction_rows:
        peak = row.get("peak_sample")
        if peak is None:
            continue
        t = librosa.samples_to_time(peak, sr=sr)
        tick = seconds_to_ticks(float(t), bpm, resolution)

        # collect notes for this tick
        emitted_bases: set[int] = set()
        for cls, mapping in CLASS_TO_NOTES.items():
            val = int(row.get(cls, 0))
            if val == 1:
                for base, cym in mapping:
                    if base not in emitted_bases:
                        add_N(tick, base, 0)
                        emitted_bases.add(base)
                    if cym is not None:
                        add_N(tick, cym, 0)

    track_lines.append("}")

    chart_text = "\n".join(song_lines + sync_lines + events_lines + track_lines) + "\n"
    (out_dir / "notes.chart").write_text(chart_text, encoding="utf-8")

    # song.ini minimal
    ini_lines = [
        "[Song]",
        f"name = {name}",
        f"artist = {meta.artist or ''}",
        f"charter = {meta.charter or ''}",
        f"year = {meta.year or ''}",
        f"genre = Unknown",
        f"song_length = 0",
        f"diff_drums = 3",
        f"pro_drums = True",
    ]
    (out_dir / "song.ini").write_text("\n".join(ini_lines) + "\n", encoding="utf-8")
