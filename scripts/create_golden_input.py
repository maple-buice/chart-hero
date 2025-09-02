from pathlib import Path

import mido
import torch
import torchaudio


def main() -> None:
    base = Path("tests/assets/golden_input")
    base.mkdir(parents=True, exist_ok=True)

    # Create a dummy MIDI file
    mid_path = base / "golden.mid"
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("note_on", note=60, velocity=64, time=32))
    track.append(mido.Message("note_off", note=60, velocity=127, time=32))
    mid.save(str(mid_path))

    # Create a dummy audio file (5 seconds @ 22.05kHz)
    sr = 22050
    torch.manual_seed(42)
    audio = torch.randn(1, sr * 5)
    wav_path = base / "golden.wav"
    torchaudio.save(str(wav_path), audio, sr)

    print(f"Wrote golden input files to {base}")


if __name__ == "__main__":
    main()
