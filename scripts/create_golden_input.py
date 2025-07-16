import mido
import torch
import torchaudio

# Create a dummy MIDI file
mid = mido.MidiFile()
track = mido.MidiTrack()
mid.tracks.append(track)
track.append(mido.Message("note_on", note=60, velocity=64, time=32))
track.append(mido.Message("note_off", note=60, velocity=127, time=32))
mid.save("/Users/maple/Repos/chart-hero/tests/assets/golden_input/golden.mid")

# Create a dummy audio file
torch.manual_seed(42)
audio = torch.randn(1, 22050 * 5)
torchaudio.save(
    "/Users/maple/Repos/chart-hero/tests/assets/golden_input/golden.wav", audio, 22050
)
