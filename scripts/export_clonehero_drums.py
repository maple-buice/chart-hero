#!/usr/bin/env python3
"""
Export Clone Hero drum charts (.chart/.txt and notes.mid) to a normalized JSON
format and produce an inventory/summary.

Usage:
  python scripts/export_clonehero_drums.py \
    --songs-root CloneHero/Songs \
    --out-dir artifacts/clonehero_charts_json

Outputs one JSON per song per difficulty per source (chart/midi).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable, Tuple
import shutil
import subprocess
import tempfile

from chart_hero.utils.rb_midi_utils import RbMidiProcessor
from chart_hero.model_training.transformer_config import get_config

# Reuse/align with discover_clonehero constants
ALLOWED_DRUM_NOTE_BASE = {0, 1, 2, 3, 4, 5}
PRO_DRUMS_OFFSET = 64
INSTRUMENT_PLUS_OFFSET = 32
DRUMS_ACCENT_OFFSET = 33
DRUMS_GHOST_OFFSET = 39


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "untitled"


def parse_song_ini(path: Path) -> dict[str, Optional[str]]:
    meta: dict[str, Optional[str]] = {
        "title": None,
        "artist": None,
        "album": None,
        "charter": None,
    }
    if not path.exists():
        return meta
    in_song = False
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith(("#", ";")):
                    continue
                if line.startswith("[") and line.endswith("]"):
                    in_song = line.lower() == "[song]"
                    continue
                if not in_song:
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    key = k.strip().lower()
                    val = v.strip()
                    if key == "name":
                        meta["title"] = val
                    elif key == "artist":
                        meta["artist"] = val
                    elif key == "album":
                        meta["album"] = val
                    elif key == "charter":
                        meta["charter"] = val
    except Exception:
        pass
    return meta


def load_chart(path: Path) -> dict[str, Any] | None:
    """Use the parser from scripts/discover_clonehero.py to read .chart/.txt."""
    # Local import to avoid a heavy import cost at module import time
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "discover_clonehero", str(Path(__file__).parent / "discover_clonehero.py")
    )
    if spec is None or spec.loader is None:
        print("ERR: could not import discover_clonehero.py")
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    try:
        return module.parse_chart(str(path))  # type: ignore[attr-defined]
    except Exception as e:
        print(f"ERR parsing chart {path}: {e}")
        return None


def chart_sync_to_tempos(sync_track: List[dict[str, Any]]) -> dict[str, Any]:
    """Convert SyncTrack entries into a tempo structure.

    We keep chart's B values as-is (likely BPM*1000) to avoid mis-scaling; consumers
    can normalize later.
    """
    tempos_b = [
        {"tick": int(x["tick"]), "bpm_x1000": int(x["value"])}
        for x in sync_track
        if x.get("type") == "B"
    ]
    ts = [
        {"tick": int(x["tick"]), "ts": int(x["value"])}
        for x in sync_track
        if x.get("type") == "TS"
    ]
    return {"tempos_bpm_x1000": tempos_b, "time_signatures": ts}


def chart_events_to_normalized(events: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """Merge flags and map lanes to normalized 8-class hits for a drum track."""
    # Group by tick
    by_tick: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        if ev.get("kind") not in ("N", "S"):
            continue
        by_tick[int(ev["tick"])].append(ev)

    out: List[dict[str, Any]] = []
    for tick in sorted(by_tick.keys()):
        bucket = by_tick[tick]
        base_lanes = {
            e["code"]
            for e in bucket
            if e["kind"] == "N" and e["code"] in ALLOWED_DRUM_NOTE_BASE
        }
        flags_cym = {
            e["code"] - PRO_DRUMS_OFFSET
            for e in bucket
            if e["kind"] == "N"
            and e["code"] >= PRO_DRUMS_OFFSET
            and (e["code"] - PRO_DRUMS_OFFSET) in ALLOWED_DRUM_NOTE_BASE
        }
        flags_acc = {
            e["code"] - DRUMS_ACCENT_OFFSET
            for e in bucket
            if e["kind"] == "N"
            and e["code"] >= DRUMS_ACCENT_OFFSET
            and (e["code"] - DRUMS_ACCENT_OFFSET) in ALLOWED_DRUM_NOTE_BASE
        }
        flags_ghost = {
            e["code"] - DRUMS_GHOST_OFFSET
            for e in bucket
            if e["kind"] == "N"
            and e["code"] >= DRUMS_GHOST_OFFSET
            and (e["code"] - DRUMS_GHOST_OFFSET) in ALLOWED_DRUM_NOTE_BASE
        }
        flags_plus = {
            e["code"] - INSTRUMENT_PLUS_OFFSET
            for e in bucket
            if e["kind"] == "N"
            and e["code"] >= INSTRUMENT_PLUS_OFFSET
            and (e["code"] - INSTRUMENT_PLUS_OFFSET) in ALLOWED_DRUM_NOTE_BASE
        }

        # Build one event per base lane
        for lane in sorted(base_lanes):
            # Map to 8-class hit
            if lane == 0:
                hit = "0"
            elif lane == 1:
                hit = "1"
            elif lane in (2, 3, 4):
                if lane in flags_cym:
                    hit = {2: "66", 3: "67", 4: "68"}[lane]
                else:
                    hit = {2: "2", 3: "3", 4: "4"}[lane]
            else:  # lane == 5 -> map to LowTom
                hit = "4"

            # Length: use first matching base event's length at this tick/lane
            len_ticks = 0
            for e in bucket:
                if e["kind"] == "N" and e["code"] == lane:
                    len_ticks = int(e.get("length") or 0)
                    break

            flags = {
                "cymbal": bool(lane in (2, 3, 4) and lane in flags_cym),
                "accent": bool(lane in flags_acc),
                "ghost": bool(lane in flags_ghost),
                "double_kick": bool(lane == 0 and lane in flags_plus),
            }

            out.append(
                {
                    "tick": int(tick),
                    "time_sec": None,  # left None for .chart to avoid bpm unit ambiguity
                    "lane": int(lane),
                    "hit": hit,
                    "flags": flags,
                    "len_ticks": len_ticks,
                }
            )
    return out


@dataclass
class ExportResult:
    out_paths: list[Path]
    stats: dict[str, Any]


def export_song_folder(
    songs_root: Path, song_dir: Path, out_dir: Path, config: Any
) -> ExportResult:
    out_paths: list[Path] = []
    stats_acc = {
        "sources": Counter(),
        "difficulties": Counter(),
        "class_hist": Counter(),
        "errors": [],
    }

    meta = parse_song_ini(song_dir / "song.ini")
    # Fallback: infer title/artist from path
    title = meta.get("title") or song_dir.name
    artist = meta.get("artist") or song_dir.parent.name
    base_slug = f"{slugify(artist)}-{slugify(title)}"

    # .chart / .txt
    chart_path = None
    for base in ("notes.chart", "notes.txt"):
        p = song_dir / base
        if p.exists():
            chart_path = p
            break
    if chart_path is not None:
        obj = load_chart(chart_path)
        if obj:
            resolution = int(obj.get("song", {}).get("Resolution") or 192)
            timing = chart_sync_to_tempos(obj.get("sync_track", []))
            # Map difficulty names to compact labels
            diff_map = {
                "EasyDrums": "Easy",
                "MediumDrums": "Medium",
                "HardDrums": "Hard",
                "ExpertDrums": "Expert",
            }
            for chart_diff, norm_diff in diff_map.items():
                evs = obj.get("tracks", {}).get(chart_diff, [])
                if not evs:
                    continue
                norm_events = chart_events_to_normalized(evs)
                # Update stats
                stats_acc["sources"].update(["chart"])
                stats_acc["difficulties"].update([f"chart:{norm_diff}"])
                stats_acc["class_hist"].update([e["hit"] for e in norm_events])

                doc = {
                    "version": 1,
                    "source": "chart",
                    "path": str(chart_path),
                    "song": {
                        "title": title,
                        "artist": artist,
                        "album": meta.get("album"),
                        "charter": meta.get("charter"),
                    },
                    "timing": {
                        "unit": "tick",
                        "resolution": resolution,
                        **timing,
                    },
                    "difficulties": {norm_diff: {"events": norm_events}},
                }

                # Write file
                out_name = f"{base_slug}.{norm_diff}.chart.json"
                dest = out_dir / out_name
                i = 1
                while dest.exists():
                    dest = out_dir / f"{base_slug}.{norm_diff}.chart.{i}.json"
                    i += 1
                out_dir.mkdir(parents=True, exist_ok=True)
                with dest.open("w", encoding="utf-8") as f:
                    json.dump(doc, f)
                out_paths.append(dest)

    # notes.mid
    midi_path = song_dir / "notes.mid"
    if midi_path.exists():
        proc = RbMidiProcessor(config)
        evdoc = proc.extract_events_per_difficulty(midi_path)
        if evdoc:
            for diff in ("Easy", "Medium", "Hard", "Expert"):
                events = evdoc["difficulties"].get(diff, {}).get("events", [])
                if not events:
                    continue
                stats_acc["sources"].update(["midi"])
                stats_acc["difficulties"].update([f"midi:{diff}"])
                stats_acc["class_hist"].update([e["hit"] for e in events])

                doc = {
                    "version": 1,
                    "source": "midi",
                    "path": str(midi_path),
                    "song": {
                        "title": title,
                        "artist": artist,
                        "album": meta.get("album"),
                        "charter": meta.get("charter"),
                    },
                    "timing": {
                        "unit": "tick",
                        "tpq": evdoc.get("tpq"),
                        "tempos": evdoc.get("tempos", []),
                    },
                    "difficulties": {diff: {"events": events}},
                }

                out_name = f"{base_slug}.{diff}.midi.json"
                dest = out_dir / out_name
                i = 1
                while dest.exists():
                    dest = out_dir / f"{base_slug}.{diff}.midi.{i}.json"
                    i += 1
                out_dir.mkdir(parents=True, exist_ok=True)
                with dest.open("w", encoding="utf-8") as f:
                    json.dump(doc, f)
                out_paths.append(dest)

    return ExportResult(out_paths=out_paths, stats=stats_acc)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Clone Hero drum charts to JSON")
    ap.add_argument("--songs-root", type=str, default="CloneHero/Songs")
    ap.add_argument("--out-dir", type=str, default="artifacts/clonehero_charts_json")
    args = ap.parse_args()

    songs_root = Path(args.songs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = get_config("local")

    total_stats = {
        "songs": 0,
        "sources": Counter(),
        "difficulties": Counter(),
        "class_hist": Counter(),
        "errors": [],
        "files_written": 0,
    }

    # Iterate song folders (two-level structure Artist/Song/)
    for dirpath, dirnames, filenames in os.walk(songs_root):
        dirpath_p = Path(dirpath)
        # Heuristic: treat folders containing song.ini as song root
        if "song.ini" in filenames:
            total_stats["songs"] += 1
            res = export_song_folder(songs_root, dirpath_p, out_dir, config)
            total_stats["files_written"] += len(res.out_paths)
            # merge counters
            total_stats["sources"].update(res.stats["sources"])  # type: ignore[arg-type]
            total_stats["difficulties"].update(res.stats["difficulties"])  # type: ignore[arg-type]
            total_stats["class_hist"].update(res.stats["class_hist"])  # type: ignore[arg-type]
            total_stats["errors"].extend(res.stats["errors"])  # type: ignore[arg-type]

    # Write summaries
    summary_json = out_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        # Convert Counters to normal dicts for JSON
        dumpable = {
            **{
                k: v
                for k, v in total_stats.items()
                if k not in {"sources", "difficulties", "class_hist"}
            },
            "sources": dict(total_stats["sources"]),
            "difficulties": dict(total_stats["difficulties"]),
            "class_hist": dict(total_stats["class_hist"]),
        }
        json.dump(dumpable, f, indent=2)

    summary_txt = out_dir / "summary.txt"
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write(f"Songs scanned: {total_stats['songs']}\n")
        f.write(f"Files written: {total_stats['files_written']}\n\n")
        f.write("Sources:\n")
        for k, v in total_stats["sources"].most_common():  # type: ignore[union-attr]
            f.write(f"  {k}: {v}\n")
        f.write("\nDifficulties (by source):\n")
        for k, v in total_stats["difficulties"].most_common():  # type: ignore[union-attr]
            f.write(f"  {k}: {v}\n")
        f.write("\nClass histogram (8-class IDs):\n")
        for k, v in total_stats["class_hist"].most_common():  # type: ignore[union-attr]
            f.write(f"  {k}: {v}\n")

    print(f"\nSummary written to: {summary_json} and {summary_txt}")


if __name__ == "__main__":
    main()
