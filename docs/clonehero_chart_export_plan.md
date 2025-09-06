# Clone Hero Drum Chart Export — Plan & TODOs

## Goals
- Discover all Clone Hero songs under `CloneHero/Songs` and detect drum charts in both `.chart/.txt` and `notes.mid` formats.
- Normalize and export drum charts into a common JSON format for downstream analysis and modeling.
- Produce per-difficulty outputs (Easy/Medium/Hard/Expert) where available.
- Generate a summary of counts (songs, charts per difficulty, source types) to gauge dataset size.

## Output
- Directory: `artifacts/clonehero_charts_json/`
- One JSON per song per difficulty per source (chart/midi), named:
  - `{artist}-{title}.{difficulty}.{source}.json` (slugified), with a collision-safe numeric suffix if needed.
- Also write a `summary.json` and `summary.txt` with aggregate stats.

## Canonical JSON Structure (per file)
```
{
  "version": 1,
  "source": "chart" | "midi",
  "path": "<absolute or workspace-relative path to the source file>",
  "song": {"title": str, "artist": str, "album": str|null, "charter": str|null},
  "timing": {
    "unit": "tick",
    "resolution": int|null,        // chart only
    "tpq": int|null,               // midi only (ticks per quarter note)
    "tempos": [                    // piecewise tempo map
      {"tick": int, "us_per_beat": int}   // midi (mido set_tempo)
      // for .chart: convert SyncTrack B events to us_per_beat if possible; otherwise keep BPM as {"tick": int, "bpm": float}
    ]
  },
  "difficulties": {
    "Easy":   {"events": [<Event>]},
    "Medium": {"events": [<Event>]},
    "Hard":   {"events": [<Event>]},
    "Expert": {"events": [<Event>]}
  }
}

Event := {
  "tick": int,                     // absolute tick within song
  "time_sec": float|null,          // optional (computed when feasible)
  "lane": int|null,                // 0..5 for CH lanes if applicable
  "hit": "0"|"1"|"2"|"3"|"4"|"66"|"67"|"68", // normalized 8-class ID
  "flags": {                       // optional modifiers
    "cymbal": bool|null,
    "accent": bool|null,
    "ghost": bool|null,
    "double_kick": bool|null
  },
  "len_ticks": int|null            // sustain length when present
}
```

Notes:
- The normalized `hit` values align with `TARGET_CLASSES` in `transformer_config.py`.
- For `.chart`, cymbal/ghost/accent/doublekick flags are detected via note codes at the same tick.
- For `notes.mid`, cymbal state can toggle via 110/111/112; apply current state at each note event.

## Discovery & Parsing
- Walk `CloneHero/Songs/**`.
- For each song folder:
  - Read `song.ini` (parse `[Song]` block) for metadata: `name`, `artist`, `album`, `charter`.
  - If `notes.chart` or `notes.txt` exists: parse sections and tracks using existing `parse_chart` from `scripts/discover_clonehero.py`.
    - For each of `EasyDrums`, `MediumDrums`, `HardDrums`, `ExpertDrums`:
      - Group events by `tick` and merge flags:
        - Base lanes: 0..5
        - Cymbal: `64 + lane`
        - Accent: `33 + lane`
        - Ghost: `39 + lane`
        - Double kick: `32 + lane`
      - Map to 8-class `hit` using lane+flag (for 2/3/4 lanes use cymbal flag to choose cymbal vs tom class).
  - If `notes.mid` exists: use `RbMidiProcessor` to extract per-difficulty events:
    - Reuse its tempo collection. Add an event-extraction helper to return events by difficulty, computing absolute tick (and `time_sec`).
    - Apply cymbal toggles (110/111/112), double kick (95), and per-difficulty note ranges (60/72/84/96 .. +5).

## Metrics to Report
- Count of songs scanned.
- Count of charts discovered by source type (.chart/.mid).
- Per-difficulty counts (how many songs have Easy/Medium/Hard/Expert for each source).
- Total events per class (8-class histogram) per source.
- List of problematic files (parsing errors, missing tempo map, etc.).

## Edge Cases
- Missing or malformed `SyncTrack` tempo (chart) — keep ticks; mark tempo map as empty.
- Multiple chart files in a folder (`notes.chart` and `notes.txt`) — prefer `notes.chart`.
- Multiple MIDI tracks — use `RbMidiProcessor` selection logic (`PART DRUMS`, fallbacks).
- Name collisions on output — add numeric suffix.

## TODO Checklist
- [ ] Implement `RbMidiProcessor.extract_events_per_difficulty(midi_path)` returning a dict with tempo map, tpq, and diff->events (with tick/time_sec, lane, hit, flags, len_ticks).
- [ ] Implement `scripts/export_clonehero_drums.py`:
  - [ ] Walk `CloneHero/Songs`.
  - [ ] Parse `song.ini` metadata.
  - [ ] Parse `.chart` using existing parser and normalize per difficulty.
  - [ ] Parse `notes.mid` via the new `RbMidiProcessor` method.
  - [ ] Emit JSON files per difficulty per source to `artifacts/clonehero_charts_json`.
  - [ ] Generate `summary.json` and `summary.txt` with counts and histograms.
- [ ] Dry-run safe guards and clear logging; skip on errors but record them.
- [ ] Optionally add a small schema in `schemas/clonehero_drum_export.schema.json` (future).
