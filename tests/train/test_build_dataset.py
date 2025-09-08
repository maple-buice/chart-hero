import mido

from chart_hero.train.build_dataset import build_labels_from_midi
from chart_hero.utils.rb_midi_utils import RbMidiProcessor
from chart_hero.model_training.transformer_config import get_config


def test_build_labels_from_midi_returns_none_without_drums(tmp_path):
    midi_path = tmp_path / "nondrum.mid"
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    track.append(mido.Message("note_on", note=40, velocity=64, time=0))
    track.append(mido.Message("note_off", note=40, velocity=64, time=480))
    mid.tracks.append(track)
    mid.save(midi_path)

    config = get_config("local")
    processor = RbMidiProcessor(config)
    labels = build_labels_from_midi(midi_path, num_time_frames=100, processor=processor)
    assert labels is None
