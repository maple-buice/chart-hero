"""
Tests for the MidiProcessor in utils/midi_utils.py.
"""

import mido
import pytest
import torch

from chart_hero.model_training.transformer_config import get_config
from chart_hero.utils.midi_utils import MidiProcessor


@pytest.fixture
def config():
    """Returns a default config for testing."""
    return get_config("local")


@pytest.fixture
def midi_processor(config):
    """Returns a MidiProcessor instance."""
    return MidiProcessor(config)


@pytest.fixture
def dummy_midi_file(tmp_path):
    """Creates a dummy MIDI file for testing."""
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    track.append(mido.Message("note_on", note=36, velocity=64, time=480))  # Kick
    track.append(mido.Message("note_on", note=38, velocity=64, time=480))  # Snare
    mid_path = tmp_path / "dummy.mid"
    mid.save(str(mid_path))
    return mid_path


def test_create_label_matrix(midi_processor, dummy_midi_file, config):
    """Test the create_label_matrix method."""
    num_time_frames = int(2 * config.sample_rate / config.hop_length)  # 2 seconds
    label_matrix = midi_processor.create_label_matrix(dummy_midi_file, num_time_frames)

    assert isinstance(label_matrix, torch.Tensor)
    assert label_matrix.shape == (num_time_frames, config.num_drum_classes)
    assert torch.sum(label_matrix) > 0
