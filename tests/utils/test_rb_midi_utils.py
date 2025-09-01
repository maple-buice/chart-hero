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


def test_rb_midi_processor_on_rb2_mid(config):
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
