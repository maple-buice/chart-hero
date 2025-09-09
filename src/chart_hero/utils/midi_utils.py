"""Utilities for processing General MIDI drum performances."""

from pathlib import Path
from typing import Any

import logging
import mido
import torch

from chart_hero.model_training.transformer_config import (
    DRUM_HIT_MAP,
    DRUM_HIT_TO_INDEX,
    TARGET_CLASSES,
)


class MidiProcessor:
    """Process GM MIDI files and create frame-by-frame drum label matrices."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def create_label_matrix(
        self, midi_path: Path, num_time_frames: int
    ) -> torch.Tensor | None:
        """Build frame-level labels from a General MIDI-style drum performance."""
        try:
            mf = mido.MidiFile(midi_path)
        except Exception as e:
            print(f"Warning: Could not read MIDI file {midi_path}: {e}")
            return None

        tpq = mf.ticks_per_beat or 480
        tempo = self._get_midi_tempo(mf)
        track = self._find_drum_track(mf)
        if track is None:
            logging.warning("No drum track found in %s", midi_path)
            return torch.zeros(
                (num_time_frames, len(TARGET_CLASSES)), dtype=torch.float32
            )

        label = torch.zeros((num_time_frames, len(TARGET_CLASSES)), dtype=torch.float32)
        time_log = 0
        for msg in track:
            time_log += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                cls_key = DRUM_HIT_MAP.get(msg.note)
                if cls_key is None:
                    continue
                time_sec = mido.tick2second(time_log, tpq, tempo)
                frame = int(time_sec * self.config.sample_rate / self.config.hop_length)
                self._mark(label, frame, cls_key, num_time_frames)

        return label

    # --- Helpers ---
    # RB-style helpers removed; this utility is for GM-style training MIDI only.

    def _mark(
        self, label: torch.Tensor, frame: int, cls_key: str, max_frames: int
    ) -> None:
        if 0 <= frame < max_frames:
            idx = DRUM_HIT_TO_INDEX.get(cls_key)
            if idx is not None:
                label[frame, idx] = 1.0

    def _get_midi_tempo(self, mf: mido.MidiFile) -> int:
        """Extract a single tempo (track 0); default 120 BPM."""
        try:
            for msg in mf.tracks[0]:
                if msg.is_meta and msg.type == "set_tempo":
                    return int(msg.tempo)
        except Exception:
            pass
        return 500000

    def _find_drum_track(self, mf: mido.MidiFile) -> mido.MidiTrack | None:
        """Locate the track containing GM drum events."""
        # Prefer track with any note-on messages on channel 9 (GM drums)
        for tr in mf.tracks:
            for msg in tr:
                if msg.type == "note_on" and getattr(msg, "channel", None) == 9:
                    return tr
        # Fallback: track name contains 'drum' or 'perc'
        for tr in mf.tracks:
            for msg in tr:
                if msg.is_meta and msg.type == "track_name":
                    name = str(msg.name).lower()
                    if "drum" in name or "perc" in name:
                        return tr
        # Last resort: if there's only one track, assume it's drums
        if len(mf.tracks) == 1:
            return mf.tracks[0]
        return None
