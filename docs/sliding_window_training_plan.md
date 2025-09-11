# Sliding Window Training & Inference Implementation Plan

**Document Version**: 1.2
**Date**: September 11, 2025
**Author**: Claude (AI Assistant)
**Project**: Chart Hero Drum Transcription Model

## Executive Summary

The current training pipeline feeds entire songs (220–338s) to a transformer capped at a 1,024‑patch sequence length. This causes severe memory pressure, poor learning dynamics, and only one training example per song. A sliding‑window approach will slice songs into manageable, overlapping windows that preserve musical context and multiply the amount of training data.

## Current Challenges

1. **Sequence Length Mismatch** – ~52k patches per song vs. model capacity of 1,024 patches
2. **Memory Constraints** – full songs require ~50 GB per batch vs. ~2 GB available
3. **Training Instability** – F1 scores around 0.19 with frequent truncation
4. **Sparse Data** – one training example per song instead of 10–30
5. **Inference Boundaries** – non-overlapping segments split drum events

## Phased Implementation

### Phase 1—Sliding Window Training (High Priority)
- Implement `SlidingWindowDataset` that extracts windows from full-song spectrograms.
- Training mode: random windows with systematic coverage and epoch-dependent seeds.
- Validation/Test mode: deterministic windows with 50 % overlap.
- Integrate with `transformer_data.py` and update the training loop to call `dataset.set_epoch`.
- Expected results: ~10–x increase in examples, memory footprint ~2 GB per batch, F1 > 0.4 within 5 epochs.

### Phase 2—Sliding Window Inference (Medium Priority)
- Generate overlapping inference segments (default 25 % overlap).
- Aggregate predictions using weighting by distance to window center, confidence selection, or multi-scale NMS.
- Target: recover >95 % of boundary events with <50 % slowdown.

### Phase 3—Window Size Optimization (Medium Priority)
- Parameterize window length; test 30–s windows (5,160 patches) against the 20–s baseline.
- Adjust model/config (`max_seq_len`, batch size) accordingly.
- Analyze memory/performance trade-offs and musical context benefits.

### Phase 4—Sequential Window Training (Low Priority)
- Extend dataset to emit sequences of consecutive windows.
- Add temporal modeling module (currently an LSTM operating on CLS embeddings) on top of per-window embeddings.
- Evaluate memory impact and loss computation across sequences.

## Implementation Considerations

- **I/O Efficiency**: Slice `.npy` files via `numpy.memmap` or cached handles to avoid disk thrashing when drawing many windows.
- **Deterministic Seeds**: Derive random jitter seeds from `(epoch, song_idx, window_idx)` to ensure reproducibility in distributed training.
- **Boundary Safety**: Guarantee coverage of song ends; handle short tracks by padding or cropping to a single window.
- **Sequential Windows**: Emit contiguous window sequences with zero‑padding for short songs and pass CLS embeddings through a light LSTM to share context across windows.
- **Prediction Normalization**: When aggregating overlapping inference results, weight contributions by distance to window center and renormalize probabilities.
- **Central Configuration**: Group sliding-window parameters (`windows_per_song`, `window_length_seconds`, `inference_overlap_ratio`, etc.) in a config dataclass with sensible defaults.
- **Regression Tests**: Add unit tests for window extraction, jitter reproducibility, boundary handling, and inference aggregation to maintain backward compatibility.

## Success Metrics

- F1 score improvement from 0.19 to >0.4 after Phase 1.
- Boundary event recall increases by ≥5 % after Phase 2.
- 30–s windows yield measurable performance gains without exceeding memory limits.

## Testing Strategy

- Unit tests covering dataset windowing logic, sequential window assembly, training-step integration, seed determinism, boundary conditions, and aggregation methods.
- Integration tests for the full training and inference pipelines using sliding windows.
- Performance and memory profiling for different window sizes and overlap ratios.

## Risk & Mitigation

| Risk | Mitigation |
|------|------------|
| Memory blow-up with longer windows | Start with small batch sizes and monitor usage |
| Training noise from random windows | Track validation metrics; use early stopping |
| Inference slowdown | Allow configurable overlap ratios and parallel processing |

## Configuration Snippet
```python
@dataclass
class SlidingWindowConfig:
    windows_per_song: int = 10
    window_jitter_ratio: float = 0.1
    inference_overlap_ratio: float = 0.25
    aggregation_method: str = 'weighted'
    window_length_seconds: float = 20.0
    enable_sequential_windows: bool = False
    sequence_length: int = 2
```

This plan addresses the core training bottlenecks while laying out a path for improved inference and advanced temporal modeling. The phased rollout enables incremental validation and manageable risk.
