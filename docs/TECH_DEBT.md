Tech Debt Tracker
=================

Purpose: lightweight list of items to revisit with tests, refactors, or polish.

Unit Testing TODOs (new features)
- lyrics.py: LRC parsing
  - Parse lines with multiple [mm:ss.xx] stamps; inline <mm:ss.xx>word tags.
  - Line-only timing → words/ syllables allocation consistency.
  - Edge cases: empty lines, malformed timestamps, overlapping tags.
- lyrics.py: VTT parsing
  - Multi-line cues; hh:mm:ss.mmm and mm:ss.mmm forms; BOM handling.
  - Even split across words; minimum durations and ordering.
- lyrics.py: Syllabification helpers
  - Deterministic splits for common words; fallback behavior without optional deps.
  - Proportional time allocation across syllables preserves span.
- lyrics.py: LRCLIB integration
  - Mock HTTP for /get (spotifyId) and /search; duration windowing logic.
  - Ensure graceful fallback on network/API errors.
- lyrics.py: to_rb_tokens
  - Trailing '-' on non-final syllables only; punctuation preserved.
- mid_vocals.py: write_vocals_midi
  - Generates PART VOCALS track with lyric meta + talky notes.
  - Phrase markers ordering and boundaries; tempo/ticks math sanity.
  - Round-trip validation by loading with mido and verifying events.
- main integration: lyrics → notes.mid
  - Integration flow with/without AudD; missing metadata; local/linked audio.
  - No crash if lyrics unavailable; still exports chart.

- audio_io.py: ffmpeg-based loader + duration helper
  - Mocked runs with/without ffmpeg/ffprobe in PATH; fallback to librosa.
  - Decode mono/stereo; resample to requested `sr`; dtype/shape invariants.
  - get_duration cross-check vs librosa within tolerance; error paths on non-audio files.
- main.py: BPM estimation with `librosa.feature.rhythm.tempo`
  - Regression vs previous alias on small corpus within tolerance; hop_length impact.
- inference/input_transform.py + training/data_preparation.py
  - Ensure consistent spectrogram shapes and `sr` when using `load_audio`.
  - Augmentation path still produces expected frame counts and padding behavior.
- inference/packager.py: OGG conversion via ffmpeg with soundfile fallback
  - Convert short fixture; verify output codec/sample rate/duration within tolerance.
  - Exercise fallback path by simulating missing ffmpeg; no crash/segfault when writing OGG.

General
- Add fixture LRC/VTT samples and golden MIDI for quick validation.
- Consider dependency-injection for HTTP and yt_dlp in tests to avoid network.
