import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from chart_hero.inference.chart_writer import SongMeta, write_chart
from chart_hero.inference.packager import package_clonehero_song
from chart_hero.inference.validation import (
    validate_chart_file,
    validate_song_ini_file,
)


def _make_dummy_audio(path: Path, sr: int = 22050, seconds: float = 0.5) -> Path:
    n = int(sr * seconds)
    # short silence with tiny noise
    y = np.random.randn(n) * 1e-4
    sf.write(str(path), y, sr)
    return path


def test_write_chart_and_validate(tmp_path: Path):
    out = tmp_path / "song"
    out.mkdir()
    meta = SongMeta(name="Unit Test Song", artist="Unit Test Artist")
    bpm = 120.0
    resolution = 192
    sr_model = 22050
    # Two notes at samples 0 and 22050: kick and snare
    rows = [
        {
            "peak_sample": 0,
            "0": 1,
            "1": 0,
            "2": 0,
            "3": 0,
            "4": 0,
            "66": 0,
            "67": 0,
            "68": 0,
        },
        {
            "peak_sample": sr_model,
            "0": 0,
            "1": 1,
            "2": 0,
            "3": 0,
            "4": 0,
            "66": 0,
            "67": 0,
            "68": 0,
        },
    ]
    write_chart(out, meta, bpm, resolution, sr_model, rows, music_stream="song.ogg")

    notes_path = out / "notes.chart"
    ini_path = out / "song.ini"

    assert notes_path.exists()
    assert ini_path.exists()

    # Validate against schemas
    validate_chart_file(notes_path)
    validate_song_ini_file(ini_path)


def test_package_clonehero_song(tmp_path: Path):
    # Create dummy audio to convert to OGG
    wav = _make_dummy_audio(tmp_path / "dummy.wav", sr=22050, seconds=0.25)

    # Minimal rows
    rows = [
        {
            "peak_sample": 0,
            "0": 1,
            "1": 0,
            "2": 0,
            "3": 0,
            "4": 0,
            "66": 0,
            "67": 0,
            "68": 0,
        }
    ]

    out_dir = package_clonehero_song(
        clonehero_root=tmp_path / "CloneHero",
        title="Packaged Song",
        artist="Packager",
        bpm=120.0,
        resolution=192,
        sr_model=22050,
        prediction_rows=rows,
        source_audio=wav,
        album_path=None,
        background_path=None,
        convert_audio=True,
    )

    assert (out_dir / "notes.chart").exists()
    assert (out_dir / "song.ini").exists()
    assert (out_dir / "song.ogg").exists()

    validate_chart_file(out_dir / "notes.chart")
    validate_song_ini_file(out_dir / "song.ini")
