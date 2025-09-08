Project Roadmap
===============

Status: living document to track planned work and priorities.

## Near-Term (High Priority)
- Lyrics & Vocals
  - Synced lyrics integration
    - Primary: LRCLIB by Spotify ID or text+duration
    - Fallback: YouTube captions (WebVTT) via yt-dlp
    - Syllabification (CMUdict/pyphen heuristic) to split words to syllables
    - Output normalized structure (lines → words → syllables)
    - Scaffolded: `src/chart_hero/inference/lyrics.py`
  - PART VOCALS MIDI exporter
    - Emit lyric tokens per syllable and phrase markers
    - MVP: talkies only (MIDI note 100), constant tempo
    - Scaffolded: `src/chart_hero/inference/mid_vocals.py`
- YouTube Music Premium Cookies
  - Add cookies support to yt-dlp downloads to prefer YT Music Premium sources
  - Detect env: `YTDLP_COOKIES_FROM_BROWSER` or `YTDLP_COOKIEFILE`
  - Pass via `cookiesfrombrowser` or `cookiefile` in `get_yt_audio`
  - Tighten format to prefer AAC/M4A
  - Status: TODO

## Modeling & Inference (Next)
- Threshold optimization
  - Sweep per-class thresholds on val set; auto-select and save with model
- Inference smoothing
  - Median filters + non-maximum suppression on per-class logits
  - Enforce minimum inter-onset spacing (e.g., 30–40 ms) for kicks/snares
- Beat/grid quantization
  - Snap events to nearest grid at 192 TPQN; optional swing triplets
- SP/Fill/Roll heuristics
  - Place S 2 (OD), S 64 (fills), S 65/66 (drum rolls) with musical heuristics
- Difficulty reductions
  - Produce Hard/Medium/Easy via density/pruning and cymbal→tom simplifications

## Training Data & Labels
- TARGET_CLASSES alignment
  - Confirm `['0','1','2','3','4','66','67','68']` consistently across label creation and heads
- RB MIDI dynamics
  - Optionally use velocity for accent/ghost class weighting or auxiliary heads
- Track selection robustness
  - Prefer expert but fall back intelligently; support merged multi-track charts

## Advanced Modeling
- Multi-task heads
  - Separate base pads (0–4/5) and cymbal flags (66/67/68) heads; combine at write time
- Variable tempo/TS
  - Beat tracking or metadata to emit full `[SyncTrack]` with tempo/metre changes
- Calibration
  - Temperature scaling / Platt scaling of logits per class for stable thresholds

## Tooling & Integration
- Validation using song files and expert/pro drum tracks
  - Explore adding a secondary validation suite that reads Clone Hero songs (notes.chart/.mid) and compares event-level predictions to expert/pro drums tracks with tolerance
  - Add a `--validate-against-songs` mode to run the model over aligned audio and compute CH‑style metrics
- Class-threshold sweeps & reports
  - Implement sweeps with plots; persist best thresholds to config/ckpt

## High-Resolution Drum Model (v1)
- [ ] (R-101) Add IOI/subdivision metrics to evaluator
  - Per-class IOI-binned recall/precision and subdivision recall (16th/32nd/64th/128th)
  - Export CSV + human-readable summary
- [ ] (R-102) Add per-class offset calibration + application in evaluator
  - Learn constant offsets on dev; apply during evaluation and export
- [ ] (R-103) New `local_highres` config (hop=128, patch_size=(8,16), stride=1)
  - label_dilation_frames=3, event_tolerance_patches=3, focal loss on
  - cap auto pos_weight <= 10
- [ ] (R-104) Dataset builder (live scan only)
  - Scan Clone Hero song folders (notes.chart/.txt/.mid + audio); no JSON stage in the pipeline
  - Audio priority & FMID logic (see plan); resolve audio from the song directory
  - Compute global offset via onset-envelope xcorr; write frame labels with ± dilation
  - Shard dataset; stratified splits by artist/title/charter
- [ ] (R-104a) FMID selection & detection
  - Detect whether `song.ogg` contains drums; if not, synthesize full mix with drums (FMID) from stems with LUFS normalization and headroom
- [ ] (R-104b) Adversarial stem mixing
  - Generate decoy negatives (non-drum stems only), confuser mixes (non-drum SNR > drums), and positive mixes across SNR sweeps; oversample windows with strong non-drum onsets and no drum labels
- [ ] (R-105) Model heads: onset auxiliary (+ optional offset regression)
  - Train Stage A (onset) then Stage B (onset+classes); hard-negative mining for cymbal/tom confusions
- [ ] (R-106) Inference decode revamp
  - Gate class logits by onset; NMS + min inter-onset spacing; apply offset corrections
- [ ] (R-107) Calibration & presets
  - Export per-class thresholds/temperatures; conservative/aggressive hi-res presets in CLI
- [ ] (R-108) Acceptance criteria gates
  - Kick/Snare F1 ≥ 0.70; 32nd/64th recall targets; tom precision/recall targets; |median offset| < 5 ms
  - FP/min targets on decoy windows below thresholds per class (e.g., kick/snare < 0.2/min, toms < 0.3/min, cymbals < 0.6/min)

See detailed plan: `docs/plans/high_resolution_drum_model.md`.

## Attribution & Licensing
- Moonscraper BSD‑3‑Clause
  - DONE: `THIRD_PARTY_NOTICES.md`; behavior ported, no direct code copied

## Backlog / Ideas
- Human-in-the-loop correction tools
- Active learning from hard negatives (polyrhythms, ghost notes)
- Sample-aware augmentation (kit-specific timbre perturbations)
