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
from typing import Dict, List, Tuple

import mido
import torch

from chart_hero.model_training.transformer_config import (
    DRUM_HIT_TO_INDEX,
    TARGET_CLASSES,
)


class RbMidiProcessor:
    def __init__(self, config):
        self.config = config

    def create_label_matrix(
        self, midi_path: Path, num_time_frames: int
    ) -> torch.Tensor | None:
        """
        Build frame-level labels from an RB/CH-style MIDI drum chart.

        Returns: tensor (num_time_frames, len(TARGET_CLASSES)).
        """
        try:
            mf = mido.MidiFile(midi_path)
        except Exception as e:
            print(f"Warning: Could not read MIDI file {midi_path}: {e}")
            return None

        tpq = mf.ticks_per_beat or 480
        tempo_map = self._collect_tempo_changes(mf)

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

        # Pick highest present difficulty
        diff_start = {"Expert": 96, "Hard": 84, "Medium": 72, "Easy": 60}
        counts = {k: 0 for k in diff_start}
        for _, msg in events:
            if msg.type == "note_on" and msg.velocity > 0:
                for d, s in diff_start.items():
                    if s <= msg.note <= s + 5:
                        counts[d] += 1
        active = next(
            (d for d in ["Expert", "Hard", "Medium", "Easy"] if counts[d] > 0), None
        )

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
                frame = self._tick_to_frame(abs_tick, tempo_map, tpq)
                self._mark(label, frame, "0", num_time_frames)
                continue
            # Difficulty gem
            if start <= n <= start + 5:
                pad = n - start  # 0..5
                frame = self._tick_to_frame(abs_tick, tempo_map, tpq)
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
            names.append(((tr_name or "").lower(), tr))
        preferred = ["part drums"]
        fallbacks = ["part real_drums_ps", "drums", "part_drums"]
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
        changes: List[Tuple[int, int]] = []
        for tr in mf.tracks:
            t = 0
            for msg in tr:
                t += msg.time
                if msg.is_meta and msg.type == "set_tempo":
                    changes.append((t, int(msg.tempo)))
        if not changes or changes[0][0] != 0:
            changes.insert(0, (0, 500000))  # 120 BPM default at 0
        changes.sort(key=lambda x: x[0])
        return changes

    def _tick_to_frame(
        self, tick: int, tempo_changes: List[Tuple[int, int]], tpq: int
    ) -> int:
        sec = 0.0
        last_tick = 0
        last_tempo = 500000
        for t_tick, tempo in tempo_changes:
            if t_tick > tick:
                break
            if t_tick > last_tick:
                sec += (t_tick - last_tick) * (last_tempo / 1_000_000.0) / tpq
                last_tick = t_tick
            last_tempo = tempo
        if tick > last_tick:
            sec += (tick - last_tick) * (last_tempo / 1_000_000.0) / tpq
        frame = int(sec * self.config.sample_rate / self.config.hop_length)
        return frame

    def _mark(
        self, label: torch.Tensor, frame: int, cls_key: str, max_frames: int
    ) -> None:
        if 0 <= frame < max_frames:
            idx = DRUM_HIT_TO_INDEX.get(cls_key)
            if idx is not None:
                label[frame, idx] = 1.0
