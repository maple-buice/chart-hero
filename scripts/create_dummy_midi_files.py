import os

import mido

for i in range(10):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("note_on", note=60, velocity=64, time=32))
    track.append(mido.Message("note_off", note=60, velocity=127, time=32))
    mid.save(os.path.join("tests/assets/dummy_data", f"dummy_{i}.mid"))
