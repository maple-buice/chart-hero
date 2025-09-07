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
- mid_vocals.py: write_vocals_midi
  - Generates PART VOCALS track with lyric meta + talky notes.
  - Phrase markers ordering and boundaries; tempo/ticks math sanity.
  - Round-trip validation by loading with mido and verifying events.
- main integration: lyrics → notes.mid
  - Integration flow with/without AudD; missing metadata; local/linked audio.
  - No crash if lyrics unavailable; still exports chart.
- main.py: BPM estimation with `librosa.feature.rhythm.tempo`
  - Regression vs previous alias on small corpus within tolerance; hop_length impact.
- inference/input_transform.py + training/data_preparation.py
  - Ensure consistent spectrogram shapes and `sr` when using `load_audio`.
  - Augmentation path still produces expected frame counts and padding behavior.
- mid_export.py: write_notes_mid (PART DRUMS + PART VOCALS)
  - Drums mapping: model classes → Expert gems 96..101; dedupe bases per tick.
  - Pro-cymbal toggles: emit 110/111/112 only when state change required; default cymbals ON.
  - Event ordering and delta times stable; short gem/toggle durations; channel 10 messages.
  - Vocals merge: lyrics meta, phrase markers, talkies timing do not collide with drums.
  - Round-trip sanity: parse notes.mid and verify expected lane counts per small fixture.
- main+packager integration: prefer notes.mid over notes.chart
  - No notes.chart written by default; notes.mid present; song.ini created; song_length populated.
  - Smoke test: generated CH folder loads with notes.mid only; MusicStream presence not required.

General
- Add fixture LRC/VTT samples and golden MIDI for quick validation.
- Consider dependency-injection for HTTP and yt_dlp in tests to avoid network.
