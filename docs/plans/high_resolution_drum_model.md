High‑Resolution Drum Transcription Plan (v1)
===========================================

Owner: chart‑hero
Status: In Progress (partially implemented)
Last updated: [auto]

Goals
- Accurately detect kick/snare and fast stickings (32nd/64th/128th where musically present) with tight timing and manageable FPs.
- Substantially reduce tom false positives while improving tom recall on tom‑heavy songs.
- Improve cymbal recall (hi‑hat/crash/ride) without flooding FPs.
- Median timing offsets |median| < 5 ms across classes on a representative dev set.

Key Ideas
- Increase temporal resolution end‑to‑end (feature hop, inference step, loss tolerance).
- Train on thousands of real charts as supervised labels with offset correction and label dilation.
- Use an onset auxiliary head (any‑drum) to gate class predictions; optional offset regression for sub‑frame timing.
- Evaluate with IOI‑binned recall and subdivision recall to verify 32nd/64th coverage.

Resolution Choices
------------------

Baseline (Hi‑res, default)
- hop_length=128 (≈5.8 ms/frame)
- patch_size=(8,16), patch_stride=1 (≈5.8 ms step)
- n_fft=1024 (≈46 ms window) or 512 (≈23 ms) for less temporal smearing vs the current 2048 (~93 ms)
- Covers: 64ths up to ~240 BPM (15.6 ms spacing) with margin; many 128th flams/drags captured with offset refinement

Micro mode (extreme passages)
- hop_length=64 (≈2.9 ms/frame)
- patch_size=(8,16), patch_stride=1 (≈2.9 ms step)
- n_fft=512 (≈23 ms) or 384 (~17 ms equivalent via zero‑pad) to avoid excessive smearing
- Usage: short windows or specific sections flagged by a first‑pass scan; or an alternate model/preset

Decode knobs by mode
- Baseline 5.8 ms: NMS k≈9–11; activity_gate≈0.55–0.65; min spacing (kick/snare)=25–35 ms; cymbal thresholds moderate.
- Micro 2.9 ms: NMS k≈17–21; activity_gate≈0.65–0.75; stronger min spacing (kick/snare)=28–40 ms; higher cymbal and ride thresholds to avoid floods.

Why not always 2.9 ms?
- Quadratic attention cost rises with sequence length; more ticks increase FPs and make noisy charts more painful. Micro mode targets the few sections that need it while keeping training/inference tractable.

Data Sources (live scan)
- Scan Clone Hero song folders directly (no intermediate JSON in the pipeline).
  - Inputs: one or more roots that contain per‑song folders with `song.ini` and one of `notes.chart`/`notes.txt`/`notes.mid`, plus audio files (stems and/or song.ogg).
  - Parser: reuse `scripts/discover_clonehero.py.parse_chart` for `.chart`/`.txt` and `utils.rb_midi_utils.RbMidiProcessor` for MIDI.
  - The previous JSON export is a useful reference for formats, not a dependency.

Milestones & Deliverables
1) Metrics & Evaluator upgrades (deliverable: evaluator reports)
   - [x] IOI‑binned recall/precision: compute per‑class recall vs inter‑onset‑interval buckets (≤10, ≤20, ≤30, … ms).
   - [x] Subdivision recall: using the song BPM/tempo map, compute recall at target subdivisions (16th/32nd/64th/128th windows).
   - [x] Constant per‑class offset correction learned on dev; apply during evaluation. (tools: `scripts/evaluate_highres_metrics.py --learn-offsets/--save-offsets`, apply via `--offsets-json` in `chart_hero.eval.evaluate_chart` and `chart_hero.main`).
   - [x] Output CSV and pretty summary; keep current summary for continuity.

2) High‑Resolution Config & Inference (deliverable: config + CLI)
   - [x] New config `local_highres`:
     - `hop_length=128` (≈5.8 ms), `patch_size=(8,16)`, `patch_stride=1`, `event_tolerance_patches=3`, `label_dilation_frames=3`.
     - `use_focal_loss=True`, `focal_alpha=0.25`, `focal_gamma=2.0`.
     - [x] Cap auto `pos_weight` to e.g., ≤10.
   - [x] Inference presets:
     - [x] conservative: strong NMS, higher activity gate, higher cym/ride thresholds.
     - [x] aggressive: lower gate/thresholds for editing assistance.
   - [x] Micro mode config `local_micro` for extreme passages (hop=64, stride=1).

3) Dataset Builder (deliverable: `python -m chart_hero.train.build_dataset`)
   - Inputs: one or more Clone Hero roots (scan live folders only; no JSON required).
   - Unified Record (internal):
     ```
     {
       path: <original notes path>,
       source: "chart"|"midi",
       song_dir: <directory containing audio>,
       timing: {tpq, tempos:[{tick, us_per_beat}], time_signatures?},
       difficulties: {Expert|Hard|Medium|Easy: {events:[{tick, time_sec?, lane, hit, flags, len_ticks}]}}
     }
     ```
   - The builder constructs this schema from the live scan.
   - Locate audio root: `song_dir = Path(doc["path"]).parent`.
   - Audio selection priority (by availability), plus sampling ratio for training:
     1. Full mix WITH drums present (FMID) — target 60–70% of batches
        - Prefer `song.ogg` only if it actually contains drums.
        - If `song.ogg` is a backing track without drums but drums stems exist, build FMID by mixing backing (song.ogg or other stems) + drums stems (`drums.ogg` or `drums_*.ogg`).
        - If no backing exists and only drums are present, skip FMID for that song.
     2. Drums‑only (`drums.ogg` OR `mix(drums_1..3)`) — target 15–25%
     3. Stems full‑mix (`mix(drums + other stems)` when no single full mix exists) — target 10–15%
     4. Optional Demucs drums from a full mix — ≤10%
     Rationale: explicitly favor full‑mix audio that contains drums to avoid overfitting to stems or backing‑only tracks.
   - Detecting whether `song.ogg` contains drums:
     - Compute a percussive onset envelope from `song.ogg` (HPSS → onset_strength with same hop as training) and from `drums.ogg` if available, or from chart onsets (impulse train).
     - If normalized correlation with drums onsets < τ (e.g., 0.15–0.25 across the song or within windows), mark `song.ogg` as “backing_no_drums”.
     - When marked “backing_no_drums” and drums stems exist, construct FMID by summing normalized backing + drums stems at −6 dB headroom.
   - Alignment:
     - Compute onset envelope from chosen audio (same hop as training) and from chart onsets; find global offset via cross‑correlation; store `offset_sec`.
     - Validate drift: if multiple tempo segments, recompute envelope vs tempo‑warped chart; drop files with excessive residual (>30 ms median absolute error) unless forced.
   - Labels:
     - Frame grid at `hop_length` (128 or 64). Mark 8‑class labels at `time_sec + offset_sec`; apply ±K‑frame dilation.
     - Build an `onset_any` channel for the onset head (OR of all classes).
     - Persist event lists for evaluation (seconds & frame indices).
   - Sharding & caching:
     - Write shard files (`.npz` or WebDataset/Parquet) with spectrograms or raw audio choice (flag), frame labels, lengths, metadata.
     - Optionally precompute transient‑enhanced mels (reuse `create_transient_enhanced_spectrogram`).
   - Splits:
     - Stratify train/val/test by artist/title and charter; ensure genre/style variety; keep at least 300 songs for dev/test.

4) Model Architecture (deliverable: updated Lightning module)
   - Heads: add `onset_head` (binary), keep `class_head` (8‑way multilabel). Loss = `L_class + λ * L_onset` (λ≈0.5–1.0 initially).
   - Optional `offset_head`: regress fractional offset relative to patch center within ±K frames for detected onsets (Huber loss).
   - Minor backbone changes: allow smaller patch size and stride; keep parameter count similar.

5) Training Schedule (deliverable: training entry + checkpoints)
   - Stage A (Onset warmup): train onset head only to high recall.
   - Stage B (Full): initialize from A, train onset+classes jointly; include hard‑negative mining for ride/hat spill and tom vs cymbal confusions.
   - Domain mix: mix batches from `drums.ogg` and `song.ogg` (or Demucs) to generalize.
   - Early stopping on event‑level F1 and subdivision recall (64ths at 180–240 BPM).
   - Export per‑class calibration (temperatures/thresholds) into `class_thresholds.json`.

6) Inference Decode (deliverable: Charter improvements)
   - Use onset head to gate class logits; NMS per class; minimum inter‑onset spacing per class (kick/snare 25–35 ms).
   - Apply per‑class constant offset correction; optional offset‑head adjustment.
   - Maintain tom↔cym arbitration with learned margin; keep configurable.

7) Acceptance Criteria (measured on dev set)
   - Kick/Snare: F1 ≥ 0.70 (or +0.25 absolute over baseline), P≥0.65, R≥0.70.
   - 32nd recall ≥ 0.65 at 180–220 BPM; 64th recall ≥ 0.45 at 180–220 BPM.
   - Cymbals: ride precision ≥ 0.55 with recall ≥ 0.50; hi‑hat/crash F1 ≥ 0.55.
   - Toms: precision ≥ 0.65 on tom‑heavy tracks with recall ≥ 0.55.
   - Timing: per‑class |median offset| < 5 ms; |mean| < 10 ms.

8) Risks & Mitigations
   - Label noise/desync: auto offset + dilation; drop worst offenders; robust losses.
   - Compute/memory at high res: use smaller patch size/stride; gradient checkpointing; shorter `max_audio_length` with chunking.
   - Domain mismatch (stems vs mix): train on both; light timbre augmentation.

Implementation Tasks
- Metrics
  - [x] Add IOI/subdivision stats to `chart_hero.eval.evaluate_chart` and CSV (new `--metrics-csv`, printed per-class bins).
  - [x] Per‑class constant offset calibration tool and application (learn via `scripts/evaluate_highres_metrics.py`; apply via `--offsets-json`).
- Config/Model
  - [x] Add `local_highres` config (hop=128, patch_size=(8,16), stride=1, dilation=3, tolerance=3, focal loss on).
  - [x] Add onset auxiliary head (+ optional offset head) in `transformer_model.py` + `lightning_module.py`.
- Dataset Builder
  - [x] New module `chart_hero.train.build_dataset` that scans Clone Hero song folders (notes.chart/.txt/.mid + audio); no JSON stage.
  - [x] Audio selection (drums.ogg→stems→song.ogg) and global offset via normalized xcorr; logs domain type.
  - [x] Frame label writer with dilation and shard writer. (Onset head uses OR(labels) computed in training, no extra channel saved.)
  - [x] Train/val/test splitting utilities (by artist/title/charter) via group-aware split.
- Preparation & Data Quality
  - [x] Loudness normalize audio to a reference (approximate -14 dBFS RMS) before feature extraction for stable dynamics across charts.
  - [x] Detect/skip desynced charts using onset-envelope alignment score threshold; flag and skip low-quality alignments.
  - [x] Duplicate/near‑duplicate detection using simple spectrogram perceptual hash; skip duplicates within a build run (opt-in `--dedupe`).
  - [ ] Domain balancing with explicit sampling ratios favoring FMID in training sampler (dataset builder logs realized domain distribution).
  - [ ] Genre/tempo stratification to guarantee high‑BPM and tom/ride‑heavy material in every epoch.

Domain Robustness
- Domain token: add a small learned embedding indicating audio domain (FMID, stems_full_mix, drums_only, backing_no_drums, Demucs) concatenated to the transformer input; helps the model adapt without separate heads.
- Dual calibration: maintain per‑domain threshold/temperature sets (FMID vs drums_only), pick at inference based on detected domain; default to FMID.
- Augment SNR: simulate bleed by mixing low‑level band‑limited noise or backing instruments; random crowd/reverb at low SNR to toughen the classifier on real songs.
- Training
  - [x] Training entry (`scripts/train_highres.py`) with local_highres preset and passthrough args.
  - [x] Calibration export (`class_thresholds.json`) and validator that prints the IOI/subdivision dashboard.
- Inference
  - [x] Decode using onset gating + min spacing; apply offset corrections.
  - [x] Two presets (conservative/aggressive) and CLI flags.

Additional Training Enhancements
--------------------------------

Architecture
- Onset auxiliary head (binary) + class head (8‑way); optional offset regression head (predict fractional position within ±K frames) to attain sub‑frame placement without shrinking hop further.
- Multi‑resolution features: concatenate mel features at two hops (e.g., 64 and 128) or use a small Conv front‑end that aggregates multiple receptive fields before the transformer.
- Local attention (chunked self‑attention) if sequence length at 2.9 ms becomes heavy; or use a conformer/CRNN hybrid for stronger local temporal modeling.

Objectives & Losses
- Label dilation ±2–3 frames at the chosen hop; event‑tolerance patches 2–3.
- Focal loss with capped positive weighting; optionally symmetric/bootstrapped CE for robustness.
- Pairwise ranking loss for tom vs cymbal on the same color (encourage a margin), to complement the decode‑time arbitration margin.
- Offset regression (Huber) trained only around positive windows to refine timing.

Sampling & Curriculum
- Class/tempo/IOI balancing: oversample ride‑heavy, tom‑heavy, and high‑BPM windows; under‑sample easy 4‑on‑the‑floor.
- Curriculum by IOI: start with 16th/8th dominant material; progressively include more 32nd/64th‑dense windows.
- Hard‑negative mining: periodically collect recent false positives (e.g., ride spill or snare ghost) and upweight them.

Augmentation
- SpecAugment (already on), light time‑stretch (±3–6%) with tempo map compensation for labels; pitch/timbre jitter for kit variance; small dynamic EQ.
- Mix domain jitter: random cross‑fade between FMID and drums‑only; optional additive crowd noise/reverb at low SNR.

Adversarial Stem Mixing (FP Suppression)
- Purpose: teach the model not to fire on rhythmic vocals, guitars, keys, claps, and other non‑drum transients.
- From songs with stems (`drums.ogg`, `guitar.ogg`, `vocals.ogg`, `bass.ogg`, `rhythm.ogg`, etc.), synthesize mixes with controlled SNRs:
  - Positive mixes: FMID where drums SNR vs other stems is sampled in [−6 dB, +6 dB]; retain ground‑truth drum labels.
  - Decoy negatives: mixes of non‑drum stems only (no drums tracks) at realistic loudness; label all frames negative.
  - Confuser mixes: add non‑drum stems on top of drums with SNR sweep in [+6, +18] dB (non‑drums louder than drums) to stress precision; labels unchanged.
- Window‑level sampling:
  - Actively mine windows with strong non‑drum onset energy but no charted drums (from parsed chart/MIDI events) and oversample them as hard negatives.
  - Randomly misalign non‑drum stems by ±10–30 ms for a fraction of samples to decorrelate drum onsets from rhythmic strums/syllables.
- Loudness handling: LUFS‑normalize stems before mixing; apply −6 dB headroom; prevent clipping.
- Curriculum: start with moderate SNR (drums ≥ others), then increase confuser SNR over epochs.
- Optional auxiliary loss: add a lightweight binary “drum presence” head trained on decoy negatives to reduce class activations when drumness is low (can be the onset head itself).

Calibration & Thresholding
- [x] Learn per‑class temperatures/thresholds on a dev set; save to checkpoint (`class_thresholds.json`) and apply at decode.
- [x] Learn constant per‑class offset corrections on dev; apply during inference to keep medians near 0 ms.

Compute on a Laptop
- Smaller hidden size/heads, gradient checkpointing, chunked sequences (`max_audio_length` shorter), and aggressive accumulation to keep VRAM/CPU RAM within limits.
- Pilot on ~300 songs to validate metrics, then scale up; snapshot every N hours to avoid long reruns.


Notes on Existing Export Script
- We will learn from `scripts/export_clonehero_drums.py` (mapping, normalization rules), but we will not rely on its JSON outputs in this pipeline.

Timeline (rough)
- Week 1: Metrics, high‑res config, dataset builder prototype on ~300 songs; quick pilot train.
- Week 2: Onset/offset heads; full dataset run (thousands); calibration; evaluator dashboards; iterate.
- Week 3: Integrate decode changes; presets; export/playtest; finalize thresholds; write docs.
Evaluation Additions (for FP analysis)
- Report FP/minute on “decoy windows” (non‑drum‑only mixes) for each class; target near‑zero for kick/snare and toms, low for cymbals.
- Per‑stem ablation tests: FP/min when adding only vocals, only guitar, only bass over silence; publish a small table in the dev dashboard.
