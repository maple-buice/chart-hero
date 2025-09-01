#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import Counter, defaultdict

ROOT = os.path.join("CloneHero", "Songs")
ALLOWED_DRUM_NOTE_BASE = {
    0,
    1,
    2,
    3,
    4,
    5,
}  # Kick, R, Y, B, Orange, Green (Moonscraper supports Orange & Green separately)
PRO_DRUMS_OFFSET = 64  # + lane -> cymbal toggle (eg 64+2=66 for yellow cymbal)
INSTRUMENT_PLUS_OFFSET = 32  # + lane -> DoubleKick/instrument-plus
DRUMS_ACCENT_OFFSET = 33  # + lane -> Accent
DRUMS_GHOST_OFFSET = 39  # + lane -> Ghost
ALLOWED_DRUM_SPECIALS = {2, 64}  # 2=OD/SP phrase, 64=fill window
DRUM_ROLL_STANDARD = 65  # S 65 drum roll (standard)
DRUM_ROLL_SPECIAL = 66  # S 66 drum roll (special)

section_re = re.compile(r"^\[(?P<section>[^\]]+)\]\s*$")


def scan_chart_sections(path: str):
    sections = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = section_re.match(line)
                if m:
                    sections.append(m.group("section"))
    except Exception as e:
        print(f"ERR reading {path}: {e}")
    return sections


def scan_song_ini_keys(path: str):
    keys = []
    try:
        # Raw parse: we only grab [Song] keys; .ini files are simple key=value
        in_song = False
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or line.startswith(";"):
                    continue
                if line.startswith("[") and line.endswith("]"):
                    in_song = line.lower() == "[song]"
                    continue
                if not in_song:
                    continue
                if "=" in line:
                    k, _ = line.split("=", 1)
                    keys.append(k.strip().lower())
    except Exception as e:
        print(f"ERR reading {path}: {e}")
    return keys


def parse_chart(path: str):
    data = {
        "format": "chart" if path.endswith(".chart") else "txt",
        "path": path,
        "song": {},
        "sync_track": [],
        "events": [],
        "tracks": defaultdict(list),
    }
    section = None
    in_block = False
    current_lines = []
    sec_header_re = re.compile(r"^\[(?P<name>[^\]]+)\]\s*$")
    kv_re = re.compile(r"^(?P<k>[^=]+)=(?P<v>.*)$")
    ev_re = re.compile(r"^(?P<tick>\d+)\s*=\s*E\s*\"(?P<text>.*)\"\s*$")
    sync_b_re = re.compile(r"^(?P<tick>\d+)\s*=\s*B\s+(?P<val>\d+)\s*$")
    sync_ts_re = re.compile(r"^(?P<tick>\d+)\s*=\s*TS\s+(?P<val>\d+)\s*$")
    note_re = re.compile(
        r"^(?P<tick>\d+)\s*=\s*N\s+(?P<code>-?\d+)\s+(?P<len>-?\d+)\s*$"
    )
    spec_re = re.compile(
        r"^(?P<tick>\d+)\s*=\s*S\s+(?P<code>-?\d+)\s+(?P<len>-?\d+)\s*$"
    )

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not in_block:
                    m = sec_header_re.match(line)
                    if m:
                        section = m.group("name")
                        continue
                    if line == "{":
                        in_block = True
                        current_lines = []
                        continue
                else:
                    if line == "}":
                        # process block by section type
                        if section == "Song":
                            for l in current_lines:
                                mk = kv_re.match(l)
                                if mk:
                                    k = mk.group("k").strip()
                                    v = mk.group("v").strip()
                                    # Trim optional quotes
                                    if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
                                        v = v[1:-1]
                                    # Cast simple ints when plausible
                                    if k in {
                                        "Offset",
                                        "Resolution",
                                        "Difficulty",
                                        "PreviewStart",
                                        "PreviewEnd",
                                    }:
                                        try:
                                            data["song"][k] = int(v)
                                        except ValueError:
                                            data["song"][k] = v
                                    else:
                                        data["song"][k] = v
                        elif section == "SyncTrack":
                            for l in current_lines:
                                mb = sync_b_re.match(l)
                                mts = sync_ts_re.match(l)
                                if mb:
                                    data["sync_track"].append(
                                        {
                                            "tick": int(mb.group("tick")),
                                            "type": "B",
                                            "value": int(mb.group("val")),
                                        }
                                    )
                                elif mts:
                                    data["sync_track"].append(
                                        {
                                            "tick": int(mts.group("tick")),
                                            "type": "TS",
                                            "value": int(mts.group("val")),
                                        }
                                    )
                        elif section == "Events":
                            for l in current_lines:
                                me = ev_re.match(l)
                                if me:
                                    data["events"].append(
                                        {
                                            "tick": int(me.group("tick")),
                                            "type": "E",
                                            "text": me.group("text"),
                                        }
                                    )
                        else:
                            # Instrument or other track; we capture drum diffs explicitly but keep generic
                            for l in current_lines:
                                mn = note_re.match(l)
                                ms = spec_re.match(l)
                                if mn:
                                    data["tracks"][section].append(
                                        {
                                            "tick": int(mn.group("tick")),
                                            "kind": "N",
                                            "code": int(mn.group("code")),
                                            "length": int(mn.group("len")),
                                        }
                                    )
                                elif ms:
                                    data["tracks"][section].append(
                                        {
                                            "tick": int(ms.group("tick")),
                                            "kind": "S",
                                            "code": int(ms.group("code")),
                                            "length": int(ms.group("len")),
                                        }
                                    )
                        # reset
                        in_block = False
                        section = None
                        current_lines = []
                    else:
                        current_lines.append(line)
    except Exception as e:
        print(f"ERR reading {path}: {e}")
    # Convert tracks defaultdict to dict for JSON friendliness
    data["tracks"] = dict(data["tracks"])
    return data


def validate_drum_tracks(chart_obj):
    issues = []
    cymbal_count = 0
    accent_count = 0
    ghost_count = 0
    sp_count = 0
    fill_count = 0
    base_note_count = 0
    doublekick_count = 0
    unexpected_notes = Counter()
    unexpected_specs = Counter()

    for diff in ("ExpertDrums", "HardDrums", "MediumDrums", "EasyDrums"):
        events = chart_obj.get("tracks", {}).get(diff, [])
        for ev in events:
            if ev["kind"] == "N":
                code = ev["code"]
                if code in ALLOWED_DRUM_NOTE_BASE:
                    base_note_count += 1
                elif code in {PRO_DRUMS_OFFSET + i for i in ALLOWED_DRUM_NOTE_BASE}:
                    cymbal_count += 1
                elif code in {DRUMS_ACCENT_OFFSET + i for i in ALLOWED_DRUM_NOTE_BASE}:
                    accent_count += 1
                elif code in {DRUMS_GHOST_OFFSET + i for i in ALLOWED_DRUM_NOTE_BASE}:
                    ghost_count += 1
                elif code in {
                    INSTRUMENT_PLUS_OFFSET + i for i in ALLOWED_DRUM_NOTE_BASE
                }:
                    doublekick_count += 1
                else:
                    unexpected_notes[code] += 1
            elif ev["kind"] == "S":
                code = ev["code"]
                if code == 2:
                    sp_count += 1
                elif code == 64:
                    fill_count += 1
                elif code in {DRUM_ROLL_STANDARD, DRUM_ROLL_SPECIAL}:
                    # drum roll sections; tracked separately if needed
                    pass
                else:
                    unexpected_specs[code] += 1

    # SyncTrack sanity
    if not any(x["type"] == "B" for x in chart_obj.get("sync_track", [])):
        issues.append("no_tempo_map_B")
    if any(
        x["type"] == "B" and x["value"] <= 0 for x in chart_obj.get("sync_track", [])
    ):
        issues.append("invalid_B_nonpositive")

    # Compose result
    result = {
        "cymbal_flags": cymbal_count,
        "accent_flags": accent_count,
        "ghost_flags": ghost_count,
        "sp_phrases": sp_count,
        "fill_windows": fill_count,
        "base_notes": base_note_count,
        "doublekick_flags": doublekick_count,
        "unexpected_note_codes": dict(unexpected_notes),
        "unexpected_special_codes": dict(unexpected_specs),
        "issues": issues,
    }
    return result


def load_json_schema(schema_path: str):
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"ERR loading schema {schema_path}: {e}")
        return None


def jsonschema_validate(instance, schema):
    try:
        import jsonschema  # type: ignore

        jsonschema.validate(instance=instance, schema=schema)
        return True, None
    except ModuleNotFoundError:
        # Fallback: simple required-check only
        missing = [k for k in schema.get("required", []) if k not in instance]
        if missing:
            return False, f"missing required keys: {missing}"
        return True, None
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Discover and validate Clone Hero charts"
    )
    parser.add_argument(
        "--export-json",
        action="store_true",
        help="Export parsed chart JSON next to chart files",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate parsed charts and INIs against schemas",
    )
    args = parser.parse_args()

    chart_counts = Counter()
    drum_diffs = Counter()
    section_examples = defaultdict(list)
    ini_key_counts = Counter()

    # Validation accumulators
    unknown_note_codes = Counter()
    unknown_spec_codes = Counter()
    charts_with_issues = 0
    charts_no_B = 0
    totals = Counter()

    for dirpath, _dirnames, filenames in os.walk(ROOT):
        # charts
        for base in ("notes.chart", "notes.txt"):
            if base in filenames:
                p = os.path.join(dirpath, base)
                sections = scan_chart_sections(p)
                for s in sections:
                    chart_counts[s] += 1
                    if len(section_examples[s]) < 3:
                        section_examples[s].append(p)
                for diff in ("EasyDrums", "MediumDrums", "HardDrums", "ExpertDrums"):
                    if diff in sections:
                        drum_diffs[diff] += 1

                # Parse and validate
                chart_obj = parse_chart(p)
                val = validate_drum_tracks(chart_obj)
                totals.update(
                    {
                        "cymbal_flags": val["cymbal_flags"],
                        "accent_flags": val["accent_flags"],
                        "ghost_flags": val["ghost_flags"],
                        "sp_phrases": val["sp_phrases"],
                        "fill_windows": val["fill_windows"],
                        "base_notes": val["base_notes"],
                        "doublekick_flags": val["doublekick_flags"],
                    }
                )
                if args.export_json:
                    out_path = os.path.splitext(p)[0] + ".chart.json"
                    try:
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(chart_obj, f, indent=2)
                    except Exception as e:
                        print(f"ERR writing {out_path}: {e}")

                if args.validate:
                    chart_schema = load_json_schema(
                        os.path.join("schemas", "chart.schema.json")
                    )
                    if chart_schema:
                        ok, err = jsonschema_validate(chart_obj, chart_schema)
                        if not ok:
                            print(f"Schema validation failed for {p}: {err}")
                unknown_note_codes.update(val["unexpected_note_codes"])
                unknown_spec_codes.update(val["unexpected_special_codes"])
                if val["issues"]:
                    charts_with_issues += 1
                if "no_tempo_map_B" in val["issues"]:
                    charts_no_B += 1

        # song.ini
        if "song.ini" in filenames:
            ini_path = os.path.join(dirpath, "song.ini")
            keys = scan_song_ini_keys(ini_path)
            ini_key_counts.update(keys)
            if args.validate:
                ini_data = {}
                try:
                    in_song = False
                    with open(ini_path, "r", encoding="utf-8", errors="ignore") as f:
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
                                ini_data[k.strip()] = v.strip()
                except Exception as e:
                    print(f"ERR reading ini {ini_path}: {e}")

                schema = load_json_schema(
                    os.path.join("schemas", "song_ini.schema.json")
                )
                if schema:
                    ok, err = jsonschema_validate(ini_data, schema)
                    if not ok:
                        print(f"song.ini schema check failed for {ini_path}: {err}")

    print("=== Chart section counts (top 30) ===")
    for sec, c in chart_counts.most_common(30):
        print(f"{sec}: {c}")

    print("\n=== Drum difficulties present (counts) ===")
    for k in ("ExpertDrums", "HardDrums", "MediumDrums", "EasyDrums"):
        print(f"{k}: {drum_diffs.get(k, 0)}")

    print("\n=== Examples for key drum sections ===")
    for k in ("ExpertDrums", "HardDrums", "MediumDrums", "EasyDrums"):
        ex = section_examples.get(k, [])
        for p in ex:
            print(f"{k} -> {p}")

    print("\n=== song.ini key frequency (top 40) ===")
    for k, c in ini_key_counts.most_common(40):
        print(f"{k}: {c}")

    print("\n=== Drum encoding summary ===")
    print(f"Base notes: {totals['base_notes']}")
    print(f"Cymbal flags (64+lane): {totals['cymbal_flags']}")
    print(f"Accent flags (33+lane): {totals['accent_flags']}")
    print(f"Ghost flags (39+lane): {totals['ghost_flags']}")
    print(f"Star Power phrases (S 2): {totals['sp_phrases']}")
    print(f"Fill windows (S 64): {totals['fill_windows']}")
    print(f"DoubleKick/InstrumentPlus (32+lane): {totals['doublekick_flags']}")
    if unknown_note_codes:
        print("Unknown note codes in drum tracks:")
        for code, cnt in unknown_note_codes.most_common():
            print(f"  code {code}: {cnt}")
    if unknown_spec_codes:
        print("Unknown special codes in drum tracks:")
        for code, cnt in unknown_spec_codes.most_common():
            print(f"  code {code}: {cnt}")
    print(f"Charts with issues: {charts_with_issues}")
    print(f"Charts missing any B tempo events: {charts_no_B}")


if __name__ == "__main__":
    main()
