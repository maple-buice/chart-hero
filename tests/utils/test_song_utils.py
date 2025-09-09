import numpy as np
import soundfile as sf

from chart_hero.utils.song_utils import mix_stems_to_waveform, save_eval_song_copy


def test_mix_stems_to_waveform(tmp_path):
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False, dtype=np.float32)
    drums = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    guitar = 0.5 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    sf.write(tmp_path / "drums.wav", drums, sr)
    sf.write(tmp_path / "guitar.wav", guitar, sr)

    mix = mix_stems_to_waveform(tmp_path, sr)
    assert mix is not None
    assert mix.shape[0] == sr
    assert np.max(np.abs(mix)) <= 1.0


def test_save_eval_song_copy(tmp_path):
    # Prepare minimal song directory
    song_dir = tmp_path / "orig"
    song_dir.mkdir()
    (song_dir / "song.ini").write_text("[Song]\nname = Test Song\n", encoding="utf-8")
    # Dummy existing notes.mid
    import mido

    mido.MidiFile().save(song_dir / "notes.mid")

    sr = 22050
    rows = [{"peak_sample": 0, "0": 1}]
    out_root = tmp_path / "eval"
    out_dir = save_eval_song_copy(
        song_dir,
        out_root,
        rows,
        bpm=120.0,
        ppq=480,
        sr=sr,
    )
    assert (out_dir / "notes.mid").exists()
    ini_text = (out_dir / "song.ini").read_text(encoding="utf-8")
    assert "name = [EVAL] Test Song" in ini_text
