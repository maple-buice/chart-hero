from __future__ import annotations

"""
Minimal MIDI exporter for Clone Hero/Rock Band-style vocals (PART VOCALS).

Emits talkies by default (MIDI note 100), with lyric meta events and phrase
markers. Accepts syllable timings in seconds and converts to ticks using a
constant BPM for now. Future extension can accept a tempo map.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import mido


@dataclass
class SyllableEvent:
    text: str  # Rock Band token already (use trailing '-' for continuations)
    t0: float  # seconds
    t1: float  # seconds


@dataclass
class Phrase:
    t0: float
    t1: float


def seconds_to_ticks(seconds: float, bpm: float, ppq: int) -> int:
    return int(round(seconds * (ppq * bpm / 60.0)))


def write_vocals_midi(
    out_path: str | Path,
    syllables: Iterable[SyllableEvent],
    *,
    phrases: Optional[Iterable[Phrase]] = None,
    bpm: float = 120.0,
    ppq: int = 480,
    talkies_note: int = 100,
) -> Path:
    """
    Write a notes.mid with a PART VOCALS track containing the given syllables.
    Phrases are optional; if provided, [phrase_start]/[phrase_end] text events
    are emitted.
    """
    outp = Path(out_path)

    mid = mido.MidiFile(ticks_per_beat=ppq)

    # Tempo/Meta track
    tempo_track = mido.MidiTrack()
    mid.tracks.append(tempo_track)
    micro = int(round(60000000.0 / bpm))
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=micro, time=0))
    tempo_track.append(
        mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0)
    )

    # Vocals track
    vox = mido.MidiTrack()
    mid.tracks.append(vox)
    vox.append(mido.MetaMessage("track_name", name="PART VOCALS", time=0))

    # Collect absolute tick events then convert to deltas in order
    events: List[tuple[int, mido.Message]] = []

    def add_abs(tick: int, msg: mido.Message) -> None:
        events.append((max(0, tick), msg))

    # Add syllables: lyric meta + talky note on/off
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

    # Phrase markers
    if phrases:
        for ph in phrases:
            ps = seconds_to_ticks(max(0.0, float(ph.t0)), bpm, ppq)
            pe = seconds_to_ticks(max(0.0, float(ph.t1)), bpm, ppq)
            if pe <= ps:
                pe = ps + 1
            add_abs(ps, mido.MetaMessage("text", text="[phrase_start]", time=0))
            add_abs(pe, mido.MetaMessage("text", text="[phrase_end]", time=0))

    # Sort by absolute tick, then a stable message type order
    type_order = {"text": 0, "lyrics": 1, "note_on": 2, "note_off": 3}
    events.sort(key=lambda x: (x[0], type_order.get(x[1].type, 99)))

    # Emit as delta times
    last_tick = 0
    for tick, msg in events:
        delta = tick - last_tick
        last_tick = tick
        msg.time = max(0, delta)
        vox.append(msg)

    outp.parent.mkdir(parents=True, exist_ok=True)
    mid.save(outp)
    return outp
