from pathlib import Path

import mido

from chart_hero.inference.mid_export import write_notes_mid
from chart_hero.inference.mid_vocals import (
    SyllableEvent as VoxSyllable,
    Phrase as VoxPhrase,
)


def test_write_notes_mid_drums_and_vocals(tmp_path: Path):
    # Minimal prediction rows: one kick at t=0, one snare at t=0.5s (sr=22050)
    sr = 22050
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
            "peak_sample": sr // 2,
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

    # Simple vocals phrase ~0..1s with one syllable
    syllables = [VoxSyllable(text="la", t0=0.0, t1=0.5)]
    phrases = [VoxPhrase(t0=0.0, t1=1.0)]

    out = write_notes_mid(
        out_dir=tmp_path,
        bpm=120.0,
        ppq=480,
        sr=sr,
        prediction_rows=rows,
        vocals_syllables=syllables,
        vocals_phrases=phrases,
    )

    assert out.exists()

    # Load and check track names are present
    mf = mido.MidiFile(out)
    names = []
    for tr in mf.tracks:
        for msg in tr:
            if msg.type == "track_name":
                names.append(msg.name.upper())
                break
    assert any("PART DRUMS" in n for n in names)
    assert any("PART VOCALS" in n for n in names)
