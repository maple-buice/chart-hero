"""
Rock Band / Clone Hero MIDI utilities.

Parses RB-style MIDI charts for drums and produces frame-level multilabel targets
compatible with chart-hero TARGET_CLASSES: ['0','1','2','3','4','66','67','68']

Semantics (informed by Moonscraper MidIOHelper):
- Difficulty ranges (drums): Easy=60, Medium=72, Hard=84, Expert=96; pads 0..5
- Pro-cymbal toggles: 110 (Y), 111 (B), 112 (G) flip cymbal/tom for that pad
- Double kick note: 95 -> Kick ('0')
- Timing: supports tempo changes via tick-accumulation over a tempo map

This is separate from utils.midi_utils (GM-focused) to avoid interference with
training-data label generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from bisect import bisect_right
from collections import deque
import logging

import mido
import torch

from chart_hero.model_training.transformer_config import (
    DRUM_HIT_TO_INDEX,
    TARGET_CLASSES,
)


class RbMidiProcessor:
    def __init__(self, config: Any) -> None:
        self.config = config

    def create_label_matrix(
        self, midi_path: Path, num_time_frames: int
    ) -> torch.Tensor | None:
        """
        Build frame-level labels from an RB/CH-style MIDI drum chart.

        Returns: tensor (num_time_frames, len(TARGET_CLASSES)).
        """
        try:
            # Some community MIDIs have out-of-range data bytes; clip to 0..127 if supported.
            try:
                mf = mido.MidiFile(midi_path, clip=True)  # type: ignore[call-arg]
            except TypeError:
                # Older mido versions don't support clip; fall back without it.
                mf = mido.MidiFile(midi_path)
        except Exception as e:
            logging.warning("Could not read MIDI file %s: %s", midi_path, e)
            return None

        tpq = mf.ticks_per_beat or 480
        tempo_map = self._collect_tempo_changes(mf)
        tempo_segments = self._build_tempo_segments(tempo_map, tpq)

        drum_tracks = self._find_drum_tracks(mf)
        if not drum_tracks:
            # Fallback: last track
            drum_tracks = [mf.tracks[-1]]

        # Flatten events to absolute tick
        events: List[Tuple[int, mido.Message]] = []
        for tr in drum_tracks:
            t = 0
            for msg in tr:
                t += msg.time
                events.append((t, msg))
        events.sort(key=lambda x: x[0])

        diff_start = {"Expert": 96, "Hard": 84, "Medium": 72, "Easy": 60}
        counts = {k: 0 for k in diff_start}
        for _, msg in events:
            if msg.type == "note_on" and msg.velocity > 0:
                for dname, start in diff_start.items():
                    if start <= msg.note <= start + 5:
                        counts[dname] += 1
                        break
        active = None
        for dname in ("Expert", "Hard", "Medium", "Easy"):
            if counts[dname] > 0:
                active = dname
                break

        label = torch.zeros((num_time_frames, len(TARGET_CLASSES)), dtype=torch.float32)
        if active is None:
            # No gem notes found; return empty labels
            return label

        start = diff_start[active]

        # Default cymbal state ON for Y/B/G (pads 2/3/4)
        cym_toggle: Dict[int, bool] = {2: True, 3: True, 4: True}

        for abs_tick, msg in events:
            if msg.type != "note_on" or msg.velocity <= 0:
                continue
            n = msg.note
            # Cymbal toggle notes
            if n in (110, 111, 112):
                pad = {110: 2, 111: 3, 112: 4}[n]
                cym_toggle[pad] = not cym_toggle.get(pad, True)
                continue
            # Double-kick note
            if n == 95:
                frame = self._tick_to_frame(abs_tick, tempo_segments, tpq)
                self._mark(label, frame, "0", num_time_frames)
                continue
            # Difficulty gem
            if start <= n <= start + 5:
                pad = n - start  # 0..5
                frame = self._tick_to_frame(abs_tick, tempo_segments, tpq)
                if pad == 0:
                    self._mark(label, frame, "0", num_time_frames)
                elif pad == 1:
                    self._mark(label, frame, "1", num_time_frames)
                elif pad in (2, 3, 4):
                    if cym_toggle.get(pad, True):
                        cym = {2: "66", 3: "67", 4: "68"}[pad]
                        self._mark(label, frame, cym, num_time_frames)
                    else:
                        tom = {2: "2", 3: "3", 4: "4"}[pad]
                        self._mark(label, frame, tom, num_time_frames)
                elif pad == 5:
                    # 5-lane green -> map to tom '4'
                    self._mark(label, frame, "4", num_time_frames)

        return label

    # --- Helpers ---
    def _find_drum_tracks(self, mf: mido.MidiFile) -> List[mido.MidiTrack]:
        names: List[Tuple[str, mido.MidiTrack]] = []
        for tr in mf.tracks:
            tr_name = None
            for msg in tr:
                if msg.is_meta and msg.type == "track_name":
                    tr_name = msg.name
                    break
            name = (tr_name or "").strip().lower().replace("_", " ")
            names.append((name, tr))
        # Prefer pro/real drums style tracks when present, otherwise PART DRUMS.
        preferred = ["part real drums ps", "real drums", "pro drums", "part drums"]
        fallbacks = ["drums", "part drums", "real drums", "part real drums"]
        out: List[mido.MidiTrack] = []
        for p in preferred:
            out = [tr for n, tr in names if n == p]
            if out:
                return out
        for key in fallbacks:
            for n, tr in names:
                if key in n:
                    return [tr]
        return []

    def _collect_tempo_changes(self, mf: mido.MidiFile) -> List[Tuple[int, int]]:
        changes: Dict[int, int] = {}
        for tr in mf.tracks:
            t = 0
            for msg in tr:
                t += msg.time
                if msg.is_meta and msg.type == "set_tempo" and t not in changes:
                    changes[t] = int(msg.tempo)
        if 0 not in changes:
            changes[0] = 500000  # 120 BPM default at 0
        return sorted(changes.items(), key=lambda x: x[0])

    def _build_tempo_segments(
        self, tempo_changes: List[Tuple[int, int]], tpq: int
    ) -> Tuple[List[int], List[float], List[int]]:
        ticks: List[int] = []
        secs: List[float] = []
        tempos: List[int] = []
        sec = 0.0
        last_tick = tempo_changes[0][0]
        last_tempo = tempo_changes[0][1]
        ticks.append(last_tick)
        secs.append(0.0)
        tempos.append(last_tempo)
        for tick, tempo in tempo_changes[1:]:
            sec += (tick - last_tick) * (last_tempo / 1_000_000.0) / tpq
            ticks.append(tick)
            secs.append(sec)
            tempos.append(tempo)
            last_tick = tick
            last_tempo = tempo
        return ticks, secs, tempos

    def _tick_to_seconds(
        self, tick: int, tempo_segments: Tuple[List[int], List[float], List[int]], tpq: int
    ) -> float:
        ticks, secs, tempos = tempo_segments
        idx = bisect_right(ticks, tick) - 1
        return secs[idx] + (tick - ticks[idx]) * (tempos[idx] / 1_000_000.0) / tpq

    def _tick_to_frame(
        self, tick: int, tempo_segments: Tuple[List[int], List[float], List[int]], tpq: int
    ) -> int:
        sec = self._tick_to_seconds(tick, tempo_segments, tpq)
        return int(sec * self.config.sample_rate / self.config.hop_length)

    def _mark(
        self, label: torch.Tensor, frame: int, cls_key: str, max_frames: int
    ) -> None:
        if 0 <= frame < max_frames:
            idx = DRUM_HIT_TO_INDEX.get(cls_key)
            if idx is not None:
                label[frame, idx] = 1.0

    # --- Event-level extraction for inventory/export ---
    def extract_events_per_difficulty(self, midi_path: Path) -> dict[str, Any] | None:
        """
        Parse a Rock Band/Clone Hero-style MIDI and return per-difficulty drum events.

        Returns a dict with keys:
          - tpq: ticks per quarter note
          - tempos: list of {tick, us_per_beat}
          - difficulties: {"Easy"|"Medium"|"Hard"|"Expert": {"events": [...]}}

        Event entries include {tick, time_sec, lane, hit, flags, len_ticks} using
        the repo's normalized 8-class mapping (TARGET_CLASSES).
        """
        try:
            # Use clipping to handle invalid data bytes in some MIDIs when supported.
            try:
                mf = mido.MidiFile(midi_path, clip=True)  # type: ignore[call-arg]
            except TypeError:
                mf = mido.MidiFile(midi_path)
        except Exception as e:
            logging.warning("Could not read MIDI file %s: %s", midi_path, e)
            return None

        tpq = mf.ticks_per_beat or 480
        tempo_map = self._collect_tempo_changes(mf)
        tempo_segments = self._build_tempo_segments(tempo_map, tpq)
        drum_tracks = self._find_drum_tracks(mf)
        if not drum_tracks:
            drum_tracks = [mf.tracks[-1]]

        # Flatten messages with absolute tick
        events: list[tuple[int, mido.Message]] = []
        for tr in drum_tracks:
            t = 0
            for msg in tr:
                t += msg.time
                events.append((t, msg))
        events.sort(key=lambda x: x[0])

        diffs = {"Easy": 60, "Medium": 72, "Hard": 84, "Expert": 96}
        out: dict[str, dict[str, list[dict[str, Any]]]] = {
            d: {"events": []} for d in diffs
        }

        # Per-note stacks (start_tick) for len_ticks calculation per difficulty/note
        pending: dict[tuple[str, int], deque[int]] = {}

        # Cymbal toggles for Y/B/G pads (2/3/4) per difficulty
        cym_toggle: dict[str, dict[int, bool]] = {
            d: {2: True, 3: True, 4: True} for d in diffs
        }

        for abs_tick, msg in events:
            if msg.type == "note_on" and msg.velocity > 0:
                n = msg.note
                # Cymbal toggles (apply to all difficulties)
                if n in (110, 111, 112):
                    pad = {110: 2, 111: 3, 112: 4}[n]
                    for state in cym_toggle.values():
                        state[pad] = not state.get(pad, True)
                    continue
                # Double kick note (map as kick on Expert)
                if n == 95:
                    t_sec = self._tick_to_seconds(abs_tick, tempo_segments, tpq)
                    out["Expert"]["events"].append(
                        {
                            "tick": int(abs_tick),
                            "time_sec": t_sec,
                            "lane": 0,
                            "hit": "0",
                            "flags": {"double_kick": True},
                            "len_ticks": 0,
                        }
                    )
                    continue
                # Difficulty gem starts
                for dname, start in diffs.items():
                    if start <= n <= start + 5:
                        key = (dname, n)
                        pending.setdefault(key, deque()).append(abs_tick)
                        break
            # Handle note_off or note_on with velocity 0 => close sustain
            elif (msg.type == "note_off") or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                n = msg.note
                for dname, start in diffs.items():
                    if start <= n <= start + 5:
                        key = (dname, n)
                        stack = pending.get(key)
                        if stack:
                            on_tick = stack.popleft()
                            pad = n - start  # 0..5
                            lane = pad
                            # Map to normalized hit
                            if pad == 0:
                                hit = "0"
                            elif pad == 1:
                                hit = "1"
                            elif pad in (2, 3, 4):
                                if cym_toggle[dname].get(pad, True):
                                    hit = {2: "66", 3: "67", 4: "68"}[pad]
                                else:
                                    hit = {2: "2", 3: "3", 4: "4"}[pad]
                            else:  # pad == 5
                                hit = "4"
                            t_sec = self._tick_to_seconds(on_tick, tempo_segments, tpq)
                            out[dname]["events"].append(
                                {
                                    "tick": int(on_tick),
                                    "time_sec": t_sec,
                                    "lane": int(lane),
                                    "hit": hit,
                                    "flags": {
                                        "cymbal": bool(
                                            pad in (2, 3, 4)
                                            and cym_toggle[dname].get(pad, True)
                                        )
                                    },
                                    "len_ticks": int(abs_tick - on_tick),
                                }
                            )
                        break

        return {
            "tpq": int(tpq),
            "tempos": [
                {"tick": int(t), "us_per_beat": int(us)} for (t, us) in tempo_map
            ],
            "difficulties": out,
        }
