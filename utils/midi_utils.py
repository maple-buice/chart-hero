"""
This module provides utilities for processing MIDI files, specifically for
creating label matrices for drum transcription.
"""

from pathlib import Path

import mido
import torch
from chart_hero.model_training.transformer_config import (
    DRUM_HIT_MAP,
    DRUM_HIT_TO_INDEX,
    TARGET_CLASSES,
)


class MidiProcessor:
    """A class to process MIDI files and create frame-by-frame label matrices."""

    def __init__(self, config):
        self.config = config

    def create_label_matrix(
        self, midi_path: Path, num_time_frames: int
    ) -> torch.Tensor | None:
        """
        Creates a frame-by-frame multi-label matrix from a MIDI file.

        Args:
            midi_path: Path to the MIDI file.
            num_time_frames: The number of time frames in the corresponding
                             spectrogram, used to determine the matrix width.

        Returns:
            A torch.Tensor of shape (num_time_frames, num_drum_classes).
        """
        try:
            midi_file = mido.MidiFile(midi_path)
        except Exception as e:
            print(f"Warning: Could not read MIDI file {midi_path}: {e}")
            return None

        ticks_per_beat = midi_file.ticks_per_beat or 480
        tempo = self._get_midi_tempo(midi_file)

        label_matrix = torch.zeros(
            (num_time_frames, len(TARGET_CLASSES)), dtype=torch.float32
        )

        time_log = 0
        for msg in midi_file.tracks[-1]:
            time_log += msg.time
            time_sec = mido.tick2second(time_log, ticks_per_beat, tempo)
            frame_index = int(
                time_sec * self.config.sample_rate / self.config.hop_length
            )

            if frame_index >= num_time_frames:
                continue

            if msg.type == "note_on" and msg.velocity > 0:
                target_hit = DRUM_HIT_MAP.get(msg.note)
                if target_hit:
                    target_index = DRUM_HIT_TO_INDEX.get(target_hit)
                    if target_index is not None:
                        label_matrix[frame_index, target_index] = 1

        return label_matrix

    def _get_midi_tempo(self, midi_file: mido.MidiFile) -> int:
        """Extracts tempo from a MIDI file, defaulting to 120 BPM."""
        for msg in midi_file.tracks[0]:
            if msg.type == "set_tempo":
                return msg.tempo
        return 500000  # Default tempo (120 BPM)
