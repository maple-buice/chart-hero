from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, cast

import jsonschema


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _schemas_dir() -> Path:
    return _repo_root() / "schemas"


def load_schema(name: str) -> Dict[str, Any]:
    # Simple JSON loader for schema files
    import json

    p = _schemas_dir() / name
    with open(p, "r", encoding="utf-8") as f:
        return cast(Dict[str, Any], json.load(f))


def read_schema(name: str) -> Dict[str, Any]:
    # Simple loader (json module) without relying on jsonschema internals
    import json

    with open(_schemas_dir() / name, "r", encoding="utf-8") as f:
        return cast(Dict[str, Any], json.load(f))


def validate_with_schema(instance: Dict[str, Any], schema_name: str) -> None:
    schema = read_schema(schema_name)
    jsonschema.validate(instance=instance, schema=schema)


def parse_song_ini(path: Path) -> Dict[str, Any]:
    # Very simple INI parser for the [Song] section
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_song = False
    data: Dict[str, Any] = {}
    for ln in text:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("[") and s.endswith("]"):
            in_song = s.lower() == "[song]"
            continue
        if not in_song:
            continue
        if "=" in s:
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip()

            # Try casting numeric-looking values to int for known keys.
            def _to_int_if_numeric(val: str) -> Any:
                try:
                    # handle simple integers; ignore floats
                    if val.lower() in ("true", "false"):
                        return val  # schema allows strings for booleans
                    return int(val)
                except Exception:
                    return val

            numeric_keys = {
                "song_length",
                "preview_start_time",
                "preview_end_time",
                "delay",
                "kit_type",
                "multiplier_note",
                "hopo_frequency",
                "frets",
                "playlist_track",
                "album_track",
                "track",
                "year",
            }
            if k.startswith("diff_") or k in numeric_keys:
                data[k] = _to_int_if_numeric(v)
            else:
                data[k] = v
    return data


def parse_chart(path: Path) -> Dict[str, Any]:
    # Minimal parser matching our writer's format
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    cur = None
    song: Dict[str, Any] = {}
    sync: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    tracks: Dict[str, List[Dict[str, Any]]] = {}

    def parse_block(header: str, start: int) -> int:
        nonlocal cur
        cur = header
        j = start + 1
        while j < len(lines):
            ln = lines[j].strip()
            if ln == "}":
                return j
            if cur == "Song":
                m = re.match(r"([A-Za-z]+)\s*=\s*(.*)", ln)
                if m:
                    key = m.group(1)
                    val = m.group(2).strip()
                    if val.startswith('"') and val.endswith('"'):
                        val = val[1:-1]
                    # cast numeric fields where expected
                    if key in ("Offset", "Resolution"):
                        try:
                            val = int(val)
                        except Exception:
                            pass
                    song[key] = val
            elif cur == "SyncTrack":
                m = re.match(r"(\d+)\s*=\s*(TS|B)\s+(\d+)", ln)
                if m:
                    sync.append(
                        {
                            "tick": int(m.group(1)),
                            "type": m.group(2),
                            "value": int(m.group(3)),
                        }
                    )
            elif cur == "Events":
                m = re.match(r"(\d+)\s*=\s*E\s+\"(.*)\"", ln)
                if m:
                    events.append(
                        {
                            "tick": int(m.group(1)),
                            "type": "E",
                            "text": m.group(2),
                        }
                    )
            else:
                # Track section (e.g., ExpertDrums)
                m = re.match(r"(\d+)\s*=\s*(N|S)\s+(\d+)\s+(\d+)", ln)
                if m:
                    tracks.setdefault(cur, []).append(
                        {
                            "tick": int(m.group(1)),
                            "kind": m.group(2),
                            "code": int(m.group(3)),
                            "length": int(m.group(4)),
                        }
                    )
            j += 1
        return j

    while i < len(lines):
        ln = lines[i].strip()
        m = re.match(r"\[(.+)\]", ln)
        if m and i + 1 < len(lines) and lines[i + 1].strip() == "{":
            header = m.group(1)
            i = parse_block(header, i + 1)
        i += 1

    return {
        "format": "chart",
        "path": str(path),
        "song": song,
        "sync_track": sync,
        "events": events,
        "tracks": tracks,
    }


def validate_chart_file(path: Path) -> None:
    instance = parse_chart(path)
    validate_with_schema(instance, "chart.schema.json")


def validate_song_ini_file(path: Path) -> None:
    data = parse_song_ini(path)
    validate_with_schema(data, "song_ini.schema.json")
