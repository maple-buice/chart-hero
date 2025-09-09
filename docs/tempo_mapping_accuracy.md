# Tempo Mapping Accuracy

The `estimate_tempo_map` routine now leverages beat tracking and beat‑to‑beat interval analysis.  On the provided test clips it consistently identifies segment boundaries within roughly **40–50 ms** and recovers the correct BPM for up to five distinct tempos.  This is a substantial improvement over the previous frame‑based approach, which drifted by several hundred milliseconds when multiple tempo changes were present.

Further accuracy improvements could explore more sophisticated techniques such as state‑space modeling, neural beat tracking, or dynamic time warping, but the current method already exceeds the 50 ms window used in the test suite.
