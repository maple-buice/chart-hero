"""
Tests for the RbMidiProcessor in utils/rb_midi_utils.py using an existing Clone Hero RB2 MIDI.
Skips gracefully if the reference MIDI file is not present.
"""

from pathlib import Path

import pytest
import torch

from chart_hero.model_training.transformer_config import get_config
from chart_hero.utils.rb_midi_utils import RbMidiProcessor


@pytest.fixture
def config():
    return get_config("local")


def test_rb_midi_processor_on_rb2_mid(config) -> None:
    midi_path = Path("CloneHero/Songs/Rock Band 2/Interpol - PDA/notes.mid")
    if not midi_path.exists():
        pytest.skip(f"RB2 MIDI not found at {midi_path}")

    proc = RbMidiProcessor(config)

    # Use a generous frame budget (e.g., 15 minutes) to ensure indices fit
    secs = 15 * 60
    frames = int(secs * (config.sample_rate / config.hop_length))
    labels = proc.create_label_matrix(midi_path, frames)

    assert isinstance(labels, torch.Tensor)
    assert labels.shape == (frames, config.num_drum_classes)
    # Expect some drum content
    assert torch.sum(labels) > 0


def test_rb_midi_processor_no_drum_track(config) -> None:
    """RbMidiProcessor should yield no events when a MIDI lacks a drum track."""
    midi_path = Path("tests/assets/midi_examples/no_drums/notes.mid")
    if not midi_path.exists():
        pytest.skip(f"Test MIDI not found at {midi_path}")

    proc = RbMidiProcessor(config)
    frames = 1000
    labels = proc.create_label_matrix(midi_path, frames)
    assert torch.sum(labels) == 0

    evdoc = proc.extract_events_per_difficulty(midi_path)
    assert evdoc is not None
    for diff in ("Easy", "Medium", "Hard", "Expert"):
        assert evdoc["difficulties"][diff]["events"] == []
