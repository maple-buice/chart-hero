from pathlib import Path

import mido

from chart_hero.inference.mid_export import write_notes_mid
from chart_hero.inference.mid_vocals import (
    SyllableEvent as VoxSyllable,
    Phrase as VoxPhrase,
    write_vocals_midi,
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


def test_pro_cymbal_toggles_order(tmp_path: Path) -> None:
    """Pro-cymbal toggles should precede gems and emit only on state change."""
    sr = 22050
    rows = [
        {"peak_sample": 0, "67": 1},  # cymbal: no toggle at start
        {"peak_sample": sr // 2, "2": 1},  # tom: toggle OFF
        {"peak_sample": sr, "67": 1},  # cymbal: toggle back ON
    ]

    out = write_notes_mid(
        out_dir=tmp_path,
        bpm=120.0,
        ppq=480,
        sr=sr,
        prediction_rows=rows,
    )

    mf = mido.MidiFile(out)
    drum_track = next(
        tr
        for tr in mf.tracks
        if any(msg.type == "track_name" and msg.name == "PART DRUMS" for msg in tr)
    )

    time = 0
    events: list[tuple[int, str, int]] = []
    for msg in drum_track:
        time += msg.time
        if msg.type in ("note_on", "note_off"):
            events.append((time, msg.type, msg.note))

    events_by_time: dict[int, list[tuple[str, int]]] = {}
    for t, typ, note in events:
        events_by_time.setdefault(t, []).append((typ, note))

    toggle_times = [
        t
        for t, entries in events_by_time.items()
        for typ, n in entries
        if typ == "note_on" and n == 110
    ]

    assert toggle_times == [480, 960]  # only two toggles, at tom then cymbal return
    assert all(n != 110 for typ, n in events_by_time.get(0, []))  # default ON

    for t in toggle_times:
        notes = [n for typ, n in events_by_time[t] if typ == "note_on"]
    assert notes[:2] == [110, 98]  # toggle precedes gem


def test_write_vocals_auto_phrase_and_skip_blank(tmp_path: Path) -> None:
    sylls = [
        VoxSyllable(text="hi", t0=0.0, t1=0.5),
        VoxSyllable(text="", t0=0.5, t1=1.0),
    ]
    out = write_vocals_midi(tmp_path / "vox.mid", sylls)
    mf = mido.MidiFile(out)
    vox_track = next(
        tr
        for tr in mf.tracks
        if any(msg.type == "track_name" and msg.name == "PART VOCALS" for msg in tr)
    )
    texts = [msg.text for msg in vox_track if msg.type == "text"]
    lyrics = [msg.text for msg in vox_track if msg.type == "lyrics"]
    assert "[phrase_start]" in texts and "[phrase_end]" in texts
    assert lyrics == ["hi"]
