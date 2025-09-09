# Tempo Mapping Accuracy

Our current `estimate_tempo_map` routine can localize a single tempo change within about 3 ms on a two‑tempo track. When additional changes are introduced accuracy degrades: on three‑ and four‑tempo clips the detected boundaries drift by roughly 350–560 ms, and a five‑tempo piece shows similar 300–560 ms error. The test suite therefore uses a 10 ms window for the single‑change case and roughly 600 ms for tracks with three or more tempos. Achieving sub‑250 ms accuracy consistently may require more sophisticated techniques:

- **Onset-aware beat tracking** – Incorporating onset strength envelopes and dynamic programming (e.g., `librosa.beat.beat_track` or `madmom`'s RNN beat tracker) can yield tighter beat localization.
- **Change-point detection** – Running a dedicated change-point detector on the beat interval sequence may better capture abrupt tempo shifts.
- **State-space modeling** – Using Kalman filters or particle filters to smooth tempo estimates while allowing quick jumps can reduce latency.
- **Neural approaches** – Training a transformer or CNN/RNN model on annotated datasets such as GTZAN or Ballroom can directly predict beat times with high resolution.
- **Dynamic Time Warping** – Aligning the audio to a generated click track via DTW can refine segment boundaries beyond frame resolution.

These approaches could be explored to reach the desired <250 ms segment boundary accuracy on complex real-world audio.
