from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

from chart_hero.utils.time_utils import seconds_to_ticks, ticks_to_seconds

from .types import PredictionRow


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


@dataclass
class SongMeta:
    name: str
    artist: str | None = None
    album: str | None = None
    year: str | int | None = None
    charter: str | None = "chart-hero"


def write_chart(
    out_dir: Path,
    meta: SongMeta,
    bpm: float,
    resolution: int,
    sr: int,
    prediction_rows: Iterable["PredictionRow"],
    music_stream: str | None = None,
    offset_seconds: float = 0.0,
) -> None:
    """Write a minimal Clone Hero-compatible .chart for ExpertDrums.

    ``prediction_rows`` should contain ``peak_sample`` and drum class labels
    like ``'0','1','2','3','4','66','67','68'``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    name = meta.name

    bpm_fraction = Fraction(str(bpm))
    ticks_per_second = Fraction(resolution, 1) * bpm_fraction / 60
    ticks_per_sample = ticks_per_second / sr
    tick_offset = seconds_to_ticks(offset_seconds, bpm, resolution)

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
    song_lines.append(f"  Offset = {int(round(offset_seconds * 1000))}")
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

    tick_to_notes: dict[int, set[int]] = defaultdict(set)
    for row in sorted(prediction_rows, key=lambda r: r.get("peak_sample", 0)):
        peak = row.get("peak_sample")
        if peak is None:
            continue
        tick = round(peak * ticks_per_sample) + tick_offset

        notes_emitted = tick_to_notes[tick]
        for cls, mapping in CLASS_TO_NOTES.items():
            if int(row.get(cls, 0)) == 1:
                for base, cym in mapping:
                    if base not in notes_emitted:
                        add_N(tick, base, 0)
                        notes_emitted.add(base)
                    if cym is not None and cym not in notes_emitted:
                        add_N(tick, cym, 0)
                        notes_emitted.add(cym)

    track_lines.append("}")

    last_tick = max(tick_to_notes, default=0)
    song_length_seconds = ticks_to_seconds(last_tick, bpm, resolution)

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
        f"song_length = {int(round(song_length_seconds * 1000))}",
        f"diff_drums = 3",
        f"pro_drums = True",
    ]
    (out_dir / "song.ini").write_text("\n".join(ini_lines) + "\n", encoding="utf-8")
