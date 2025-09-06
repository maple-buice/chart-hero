from __future__ import annotations

"""
Export a single notes.mid containing PART DRUMS and optional PART VOCALS.

Drums encoding uses Rock Band/Clone Hero MIDI semantics:
- Difficulty: Expert (notes 96..101 for pads 0..5)
- Pro-cymbal toggles: 110 (Y), 111 (B), 112 (G) to flip cymbal/tom state
- Default cymbal state ON for pads 2/3/4; toggle OFF for tom hits
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import mido

from chart_hero.inference.mid_vocals import SyllableEvent, Phrase
from chart_hero.inference.types import PredictionRow


def seconds_to_ticks(seconds: float, bpm: float, ppq: int) -> int:
    return int(round(seconds * (ppq * bpm / 60.0)))


def _add_vocals_track(
    mid: mido.MidiFile,
    syllables: Iterable[SyllableEvent],
    phrases: Optional[Iterable[Phrase]],
    bpm: float,
    ppq: int,
) -> None:
    vox = mido.MidiTrack()
    mid.tracks.append(vox)
    vox.append(mido.MetaMessage("track_name", name="PART VOCALS", time=0))

    events: List[tuple[int, mido.Message]] = []
    talkies_note = 100

    def add_abs(tick: int, msg: mido.Message) -> None:
        events.append((max(0, tick), msg))

    for syl in syllables:
        t0 = max(0.0, float(syl.t0))
        t1 = max(t0 + 0.01, float(syl.t1))
        start = seconds_to_ticks(t0, bpm, ppq)
        end = seconds_to_ticks(t1, bpm, ppq)
        dur = max(1, end - start)
        text = (syl.text or "").strip()
        add_abs(start, mido.MetaMessage("lyrics", text=text, time=0))
        add_abs(
            start,
            mido.Message("note_on", note=talkies_note, velocity=96, channel=0, time=0),
        )
        add_abs(
            start + dur,
            mido.Message("note_off", note=talkies_note, velocity=0, channel=0, time=0),
        )

    if phrases:
        for ph in phrases:
            ps = seconds_to_ticks(max(0.0, float(ph.t0)), bpm, ppq)
            pe = seconds_to_ticks(max(0.0, float(ph.t1)), bpm, ppq)
            if pe <= ps:
                pe = ps + 1
            add_abs(ps, mido.MetaMessage("text", text="[phrase_start]", time=0))
            add_abs(pe, mido.MetaMessage("text", text="[phrase_end]", time=0))

    type_order = {"text": 0, "lyrics": 1, "note_on": 2, "note_off": 3}
    events.sort(key=lambda x: (x[0], type_order.get(x[1].type, 99)))
    last = 0
    for t, msg in events:
        delta = t - last
        last = t
        msg.time = max(0, delta)
        vox.append(msg)


def _add_drums_track(
    mid: mido.MidiFile,
    rows: Iterable[PredictionRow],
    *,
    sr: int,
    bpm: float,
    ppq: int,
) -> None:
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("track_name", name="PART DRUMS", time=0))

    # Map model classes to pads/cymbals
    # Pads: 0=kick,1=snare,2=Y,3=B,4=G,5=5-lane green (we map to 4)
    # Expert difficulty base note
    EXPERT_BASE = 96
    GEM_NOTE = {
        0: EXPERT_BASE + 0,
        1: EXPERT_BASE + 1,
        2: EXPERT_BASE + 2,
        3: EXPERT_BASE + 3,
        4: EXPERT_BASE + 4,
        5: EXPERT_BASE + 5,
    }
    # Cymbal toggle notes for Y/B/G pads
    TOGGLE = {2: 110, 3: 111, 4: 112}

    # Default cymbal state ON
    cym_on = {2: True, 3: True, 4: True}

    events: List[tuple[int, mido.Message]] = []

    def add_abs(tick: int, msg: mido.Message) -> None:
        events.append((max(0, tick), msg))

    # Duration for gems/toggles in ticks (short)
    gem_dur = max(1, int(round(ppq * 0.125)))  # ~1/8 note
    tog_dur = 1

    for row in rows:
        peak = row.get("peak_sample")
        if peak is None:
            continue
        t_sec = float(int(peak) / float(sr))
        tick = seconds_to_ticks(t_sec, bpm, ppq)

        # Determine desired cymbal state per pad at this tick
        desired: dict[int, Optional[bool]] = {2: None, 3: None, 4: None}
        if int(row.get("66", 0)) == 1:
            desired[2] = True
        if int(row.get("67", 0)) == 1:
            desired[3] = True
        if int(row.get("68", 0)) == 1:
            desired[4] = True
        if int(row.get("2", 0)) == 1:
            desired[2] = False if desired[2] is None else desired[2]
        if int(row.get("3", 0)) == 1:
            desired[3] = False if desired[3] is None else desired[3]
        if int(row.get("4", 0)) == 1:
            desired[4] = False if desired[4] is None else desired[4]

        # Apply toggles if needed
        for pad in (2, 3, 4):
            want = desired[pad]
            if want is None or want == cym_on[pad]:
                continue
            note = TOGGLE[pad]
            add_abs(
                tick,
                mido.Message("note_on", note=note, velocity=100, channel=9, time=0),
            )
            add_abs(
                tick + tog_dur,
                mido.Message("note_off", note=note, velocity=0, channel=9, time=0),
            )
            cym_on[pad] = want

        # Emit gems (dedupe bases)
        lanes: List[int] = []
        if int(row.get("0", 0)) == 1:
            lanes.append(0)
        if int(row.get("1", 0)) == 1:
            lanes.append(1)
        # Pad 2/3/4 depending on tom vs cym
        if int(row.get("66", 0)) == 1 or int(row.get("2", 0)) == 1:
            lanes.append(2)
        if int(row.get("67", 0)) == 1 or int(row.get("3", 0)) == 1:
            lanes.append(3)
        if int(row.get("68", 0)) == 1 or int(row.get("4", 0)) == 1:
            lanes.append(4)

        seen = set()
        for pad in lanes:
            if pad in seen:
                continue
            seen.add(pad)
            note = GEM_NOTE[pad]
            add_abs(
                tick,
                mido.Message("note_on", note=note, velocity=100, channel=9, time=0),
            )
            add_abs(
                tick + gem_dur,
                mido.Message("note_off", note=note, velocity=0, channel=9, time=0),
            )

    # Sort and emit deltas
    type_order = {"note_on": 0, "note_off": 1}
    events.sort(key=lambda x: (x[0], type_order.get(x[1].type, 99), x[1].note))
    last = 0
    for t, msg in events:
        delta = t - last
        last = t
        msg.time = max(0, delta)
        tr.append(msg)


def write_notes_mid(
    out_dir: Path,
    *,
    bpm: float,
    ppq: int,
    sr: int,
    prediction_rows: Iterable[PredictionRow],
    vocals_syllables: Optional[Iterable[SyllableEvent]] = None,
    vocals_phrases: Optional[Iterable[Phrase]] = None,
) -> Path:
    """
    Create `notes.mid` in `out_dir` with PART DRUMS and optional PART VOCALS.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "notes.mid"

    mid = mido.MidiFile(ticks_per_beat=ppq)
    # Tempo track
    tempo_track = mido.MidiTrack()
    mid.tracks.append(tempo_track)
    micro = int(round(60000000.0 / bpm))
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=micro, time=0))
    tempo_track.append(
        mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0)
    )

    # Drums first
    _add_drums_track(mid, prediction_rows, sr=sr, bpm=bpm, ppq=ppq)

    # Optional vocals
    if vocals_syllables:
        _add_vocals_track(mid, vocals_syllables, vocals_phrases, bpm=bpm, ppq=ppq)

    mid.save(path)
    return path
