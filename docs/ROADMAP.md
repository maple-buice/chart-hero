Project Roadmap
===============

Status: living document to track planned work and priorities.

Near-Term (High Priority)
- MIDI labels: Robust RB drum parsing
  - Select correct tracks (PART DRUMS, REAL_DRUMS_PS), handle tempo maps
  - Handle Pro-Cymbal toggles (110/111/112), double-kick (95)
  - DONE: Implemented in `src/chart_hero/utils/midi_utils.py`
- Event-tolerant training objectives
  - Label dilation across time frames before pooling to patches
  - Optional focal loss for sparse onsets; keep pos_weight
  - Per-class thresholds for better precision/recall balance
  - DONE: Implemented in `lightning_module.py` + `transformer_config.py`
- Event-level validation metrics
  - Precision/Recall/F1 with onset tolerance in patch units
  - DONE: Implemented in `lightning_module.py`
- Clone Hero export path
  - Moonscraper-compatible `.chart` writer + `song.ini`
  - Integrated `--export-clonehero`
  - DONE: `src/chart_hero/inference/chart_writer.py`, `main.py`
- Data/Schema validation
  - Discovery script exports JSON + schema validates charts and INIs
  - DONE: `scripts/discover_clonehero.py`, `schemas/`

Modeling & Inference (Next)
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

Training Data & Labels
- TARGET_CLASSES alignment
  - Confirm `['0','1','2','3','4','66','67','68']` consistently across label creation and heads
- RB MIDI dynamics
  - Optionally use velocity for accent/ghost class weighting or auxiliary heads
- Track selection robustness
  - Prefer expert but fall back intelligently; support merged multi-track charts

Advanced Modeling
- Multi-task heads
  - Separate base pads (0–4/5) and cymbal flags (66/67/68) heads; combine at write time
- Variable tempo/TS
  - Beat tracking or metadata to emit full `[SyncTrack]` with tempo/metre changes
- Calibration
  - Temperature scaling / Platt scaling of logits per class for stable thresholds

Tooling & Integration
- Validation using song files and expert/pro drum tracks
  - Explore adding a secondary validation suite that reads Clone Hero songs (notes.chart/.mid) and compares event-level predictions to expert/pro drums tracks with tolerance
  - Add a `--validate-against-songs` mode to run the model over aligned audio and compute CH‑style metrics
- Class-threshold sweeps & reports
  - Implement sweeps with plots; persist best thresholds to config/ckpt

Attribution & Licensing
- Moonscraper BSD‑3‑Clause
  - DONE: `THIRD_PARTY_NOTICES.md`; behavior ported, no direct code copied

Backlog / Ideas
- Human-in-the-loop correction tools
- Active learning from hard negatives (polyrhythms, ghost notes)
- Sample-aware augmentation (kit-specific timbre perturbations)
