from __future__ import annotations

"""Utility functions for converting between seconds and MIDI ticks.

These helpers centralize the tick/second conversions so that timing math
remains consistent across modules.
"""

from fractions import Fraction


def seconds_to_ticks(seconds: float, bpm: float, resolution: int) -> int:
    """Convert seconds to MIDI ticks using fractional math for precision."""
    bpm_fraction = Fraction(str(bpm))
    ticks_per_second = Fraction(resolution, 1) * bpm_fraction / 60
    return round(Fraction(str(seconds)) * ticks_per_second)


def ticks_to_seconds(ticks: int, bpm: float, resolution: int) -> float:
    """Convert MIDI ticks to seconds using fractional math for precision."""
    bpm_fraction = Fraction(str(bpm))
    seconds_per_tick = Fraction(60, 1) / (Fraction(resolution, 1) * bpm_fraction)
    return float(Fraction(ticks, 1) * seconds_per_tick)
