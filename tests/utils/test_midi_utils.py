"""Tests for the MidiProcessor in utils/midi_utils.py."""

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

    track.append(mido.MetaMessage("track_name", name="Drums"))
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    track.append(mido.Message("note_on", note=36, velocity=64, time=480, channel=9))  # Kick
    track.append(mido.Message("note_on", note=38, velocity=64, time=480, channel=9))  # Snare
    mid_path = tmp_path / "dummy.mid"
    mid.save(str(mid_path))
    return mid_path


@pytest.fixture
def no_drum_midi_file(tmp_path):
    """Creates a MIDI file with no drum track."""
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("track_name", name="Piano"))
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    track.append(mido.Message("note_on", note=100, velocity=64, time=480, channel=0))
    mid_path = tmp_path / "nodrum.mid"
    mid.save(str(mid_path))
    return mid_path


def test_create_label_matrix(midi_processor, dummy_midi_file, config):
    """Drum track is parsed into non-empty label matrix."""
    num_time_frames = int(2 * config.sample_rate / config.hop_length)  # 2 seconds
    label_matrix = midi_processor.create_label_matrix(dummy_midi_file, num_time_frames)

    assert isinstance(label_matrix, torch.Tensor)
    assert label_matrix.shape == (num_time_frames, config.num_drum_classes)
    assert torch.sum(label_matrix) > 0  # Check that some notes were found


def test_no_drum_track_returns_empty_labels(
    midi_processor, no_drum_midi_file, config
):
    """MIDIs without drum tracks yield an all-zero label matrix."""
    num_time_frames = int(2 * config.sample_rate / config.hop_length)
    label_matrix = midi_processor.create_label_matrix(no_drum_midi_file, num_time_frames)

    assert isinstance(label_matrix, torch.Tensor)
    assert label_matrix.shape == (num_time_frames, config.num_drum_classes)
    assert torch.count_nonzero(label_matrix) == 0

