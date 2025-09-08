import mido

import json

from chart_hero.train.build_dataset import build_labels_from_midi, json_index_has_song, _slugify
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


def test_json_index_has_song_filters_by_pack(tmp_path):
    root = tmp_path / "Songs"
    pack_drum = root / "PackDrum" / "Artist - Song"
    pack_nodrum = root / "PackNoDrum" / "Artist - Song"
    for p in (pack_drum, pack_nodrum):
        p.mkdir(parents=True)
        ini = p / "song.ini"
        ini.write_text("[song]\nartist = Artist\nname = Song\n")

    # Drum MIDI
    drum_mid = pack_drum / "notes.mid"
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    track.append(mido.Message("note_on", note=96, velocity=100, time=0))
    track.append(mido.Message("note_off", note=96, velocity=100, time=480))
    mid.tracks.append(track)
    mid.save(drum_mid)

    # Non-drum MIDI
    nodrum_mid = pack_nodrum / "notes.mid"
    mid2 = mido.MidiFile()
    t2 = mido.MidiTrack()
    t2.append(mido.Message("note_on", note=40, velocity=100, time=0))
    t2.append(mido.Message("note_off", note=40, velocity=100, time=480))
    mid2.tracks.append(t2)
    mid2.save(nodrum_mid)

    # Build JSON index with only drum pack
    idx_root = tmp_path / "index"
    idx_dir = idx_root / _slugify("PackDrum") / _slugify("Artist") / _slugify("Song")
    idx_dir.mkdir(parents=True)
    (idx_dir / "dummy.midi.json").write_text(
        json.dumps({"path": str(drum_mid)})
    )

    roots = [root]
    assert json_index_has_song(pack_drum, roots, idx_root)
    assert not json_index_has_song(pack_nodrum, roots, idx_root)
