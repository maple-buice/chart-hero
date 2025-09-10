# Sliding Window Training & Inference Implementation Plan

**Document Version**: 1.0
**Date**: September 10, 2025
**Author**: Claude (AI Assistant)
**Project**: Chart Hero Drum Transcription Model

## Executive Summary

This document outlines a comprehensive plan to implement sliding window training and inference for the Chart Hero drum transcription model. The current approach uses full-length songs (220-338 seconds) that create memory issues and poor training dynamics. The sliding window approach will enable efficient training while preserving musical context and providing significantly more training examples.

## Current State Analysis

### Problems Identified
1. **Sequence Length Mismatch**: Dataset contains ~52,000 patches per song vs model capacity of 1,024 patches
2. **Memory Constraints**: Full songs require ~50GB+ memory per batch vs available ~2GB
3. **Poor Training Dynamics**: Very low F1 scores (0.19) due to sequence length issues
4. **Limited Training Examples**: Only 1 example per song instead of potential 10-30 examples
5. **Inference Boundary Issues**: Non-overlapping segments may split drum events

### Current Architecture
- **Model**: Transformer with 1,024 max sequence length, 20-second windows expected
- **Dataset**: Full Clone Hero songs saved as individual .npy files (220-338s each)
- **Inference**: Non-overlapping 20s segments with potential boundary event loss

## Proposed Solution Overview

Implement a three-phase sliding window system that processes full songs in manageable chunks while preserving musical context and maximizing training data utilization.

## Phase 1: Basic Sliding Window Training (Priority: HIGH)

### 1.1 Objectives
- Enable training on existing full-song dataset without regeneration
- Increase training examples by 10-30x per song
- Fix memory and sequence length issues
- Achieve immediate improvement in F1 scores

### 1.2 Technical Implementation

#### 1.2.1 SlidingWindowDataset Class
```python
class SlidingWindowDataset(Dataset):
    """
    Dataset that extracts sliding windows from full-song spectrograms.
    Supports both random training windows and systematic validation windows.
    """

    def __init__(self, data_files, config, mode='train', windows_per_song=10):
        self.data_files = data_files  # List of (spec_file, label_file) tuples
        self.config = config
        self.mode = mode
        self.windows_per_song = windows_per_song
        self.window_frames = int(config.max_audio_length * config.sample_rate / config.hop_length)
        self.epoch_num = 0  # Updated by trainer

    def __len__(self):
        return len(self.data_files) * self.windows_per_song

    def __getitem__(self, idx):
        # Implementation details below
        pass

    def set_epoch(self, epoch_num):
        """Called by trainer to enable different random windows per epoch"""
        self.epoch_num = epoch_num
```

#### 1.2.2 Window Extraction Strategy

**Training Mode** (Random with Coverage):
- Generate `windows_per_song` systematic positions across the song
- Add random jitter (±10% of window length) to each position
- Use epoch-dependent random seeds for variety across epochs
- Ensure no window extends beyond song boundaries

**Validation/Test Mode** (Systematic):
- Use 50% overlapping windows for complete coverage
- Deterministic positioning for reproducible results
- Handle edge cases (short songs, boundary conditions)

#### 1.2.3 Random Seed Strategy
```python
def _get_window_start(self, song_idx, window_idx, song_length):
    if self.mode == 'train':
        # Base systematic position
        hop_frames = (song_length - self.window_frames) // (self.windows_per_song - 1)
        base_start = window_idx * hop_frames

        # Add random jitter
        max_jitter = min(hop_frames // 4, self.window_frames // 10)
        random.seed(self.epoch_num * 100000 + song_idx * 1000 + window_idx)
        jitter = random.randint(-max_jitter, max_jitter)

        return max(0, min(base_start + jitter, song_length - self.window_frames))
    else:
        # Systematic overlap for validation
        hop_frames = self.window_frames // 2
        return min(window_idx * hop_frames, song_length - self.window_frames)
```

### 1.3 Integration Points

#### 1.3.1 Data Loading Modifications
- Modify `transformer_data.py` to use `SlidingWindowDataset` instead of `NpyDrumDataset`
- Update data loader creation to pass epoch information
- Maintain existing caching and performance optimizations

#### 1.3.2 Training Loop Updates
- Add `dataset.set_epoch(epoch_num)` call in training loop
- Monitor memory usage and adjust `windows_per_song` if needed
- Update logging to reflect increased number of training examples

### 1.4 Expected Outcomes
- **Training Examples**: Increase from ~1,000 to ~15,000 examples
- **Memory Usage**: Reduce from 50GB to 2GB per batch
- **F1 Score**: Improve from 0.19 to 0.4+ within 5 epochs
- **Training Stability**: Eliminate sequence truncation issues

### 1.5 Testing Strategy
- Start with `windows_per_song=5` to test implementation
- Gradually increase to optimal value based on memory/performance
- Compare F1 scores before/after implementation
- Validate that all drum classes learn (not just kicks)

## Phase 2: Sliding Window Inference (Priority: MEDIUM)

### 2.1 Objectives
- Eliminate boundary event splitting in current non-overlapping inference
- Improve prediction quality through ensemble methods
- Maintain reasonable inference speed (<2x slowdown)

### 2.2 Technical Implementation

#### 2.2.1 Overlapping Segment Generation
```python
def audio_to_sliding_segments(audio_path: str, config: TransformerConfig,
                             overlap_ratio: float = 0.25) -> list[Segment]:
    """
    Generate overlapping audio segments for inference.
    25% overlap provides good boundary coverage without excessive processing time.
    """
    segment_length_samples = int(config.max_audio_length * sr)
    hop_samples = int(segment_length_samples * (1 - overlap_ratio))

    segments = []
    for i in range(0, len(y) - segment_length_samples + hop_samples, hop_samples):
        # Create segment with global frame positioning
        # Implementation details...

    return segments
```

#### 2.2.2 Prediction Ensemble Strategy

**Method 1: Weighted Averaging by Distance from Window Center**
- Predictions near window centers get higher weights
- Predictions near boundaries get lower weights
- Smooth weight function to avoid discontinuities

**Method 2: Confidence-Based Selection**
- For overlapping regions, select prediction with highest confidence
- Use model probability outputs as confidence scores
- Apply per-class confidence thresholds

**Method 3: Multi-Scale NMS**
- Apply NMS within each window first
- Then apply global NMS across all windows
- Use temporal proximity and class-specific tolerances

#### 2.2.3 Implementation Architecture
```python
class SlidingWindowInference:
    def __init__(self, charter, overlap_ratio=0.25, aggregation_method='weighted'):
        self.charter = charter
        self.overlap_ratio = overlap_ratio
        self.aggregation_method = aggregation_method

    def predict(self, segments):
        # Run inference on overlapping segments
        predictions = [self.charter.predict_single_segment(seg) for seg in segments]

        # Aggregate predictions based on chosen method
        return self.aggregate_predictions(predictions)

    def aggregate_predictions(self, predictions):
        # Implementation based on selected method
        pass
```

### 2.3 Expected Outcomes
- **Boundary Event Recovery**: Capture 95%+ of events that span segment boundaries
- **Prediction Quality**: 5-10% improvement in overall F1 score
- **Processing Time**: 1.33x increase (acceptable for improved quality)

## Phase 3: Window Size Optimization (Priority: MEDIUM)

### 3.1 Objectives
- Determine optimal window length for musical context vs computational efficiency
- Test 30-second windows for better verse/chorus transition modeling
- Evaluate memory and performance trade-offs

### 3.2 Analysis Framework

#### 3.2.1 Window Length Comparison
| Window Length | Patches | Memory/Sample | Musical Context | Feasibility |
|---------------|---------|---------------|-----------------|-------------|
| 20s (current) | 3,438 | 5.0MB | Drum fill + verse | ✅ Proven |
| 30s | 5,160 | 7.6MB | Full verse + chorus transition | 🟡 Test needed |
| 45s | 7,744 | 11.3MB | Verse + chorus + bridge | 🔴 Memory risk |

#### 3.2.2 Configuration Updates Required
For 30-second windows:
- Update `max_audio_length: 30.0` in configs
- Update `max_seq_len: 5200` (with safety margin)
- Adjust batch sizes to accommodate memory increase

### 3.3 Testing Methodology
1. **Memory Profiling**: Test 30s windows with reduced batch sizes
2. **Musical Analysis**: Evaluate improvement in complex drum patterns
3. **Performance Benchmarking**: Compare training/inference speed
4. **Ablation Study**: 20s vs 30s on identical data subsets

### 3.4 Implementation Strategy
- Implement as config parameter (`window_length_seconds`)
- Enable A/B testing between window sizes
- Use existing sliding window infrastructure
- Gradual rollout: test → validate → deploy

## Phase 4: Sequential Window Training (Priority: LOW)

### 4.1 Objectives
- Capture temporal dependencies across multiple consecutive windows
- Model drum pattern evolution over time
- Learn musical transitions and build-ups

### 4.2 Technical Approach

#### 4.2.1 Sequential Dataset Modification
```python
class SequentialSlidingWindowDataset(SlidingWindowDataset):
    """
    Extension that returns sequences of consecutive windows.
    Enables modeling of temporal dependencies across windows.
    """

    def __init__(self, *args, sequence_length=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence_length = sequence_length  # Number of consecutive windows

    def __getitem__(self, idx):
        # Return sequence_length consecutive windows
        windows, labels = [], []
        for i in range(self.sequence_length):
            window, label = self._get_single_window(idx, offset=i)
            windows.append(window)
            labels.append(label)

        return torch.stack(windows), torch.stack(labels)
```

#### 4.2.2 Model Architecture Extensions
```python
class SequentialWindowTransformer(nn.Module):
    def __init__(self, base_transformer, temporal_modeling='attention'):
        self.base_transformer = base_transformer
        self.temporal_modeling = temporal_modeling

        if temporal_modeling == 'attention':
            self.temporal_attention = nn.MultiheadAttention(hidden_size, num_heads)
        elif temporal_modeling == 'lstm':
            self.temporal_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

    def forward(self, x):
        # x shape: [batch, seq_len, channels, time, freq]
        batch_size, seq_len = x.shape[:2]

        # Process each window independently
        window_outputs = []
        for i in range(seq_len):
            output = self.base_transformer(x[:, i])  # [batch, time_patches, hidden]
            # Pool to fixed representation
            pooled = output.mean(dim=1)  # [batch, hidden]
            window_outputs.append(pooled)

        # Model temporal dependencies
        temporal_features = torch.stack(window_outputs, dim=1)  # [batch, seq_len, hidden]

        if self.temporal_modeling == 'attention':
            attended, _ = self.temporal_attention(temporal_features, temporal_features, temporal_features)
        else:  # LSTM
            attended, _ = self.temporal_lstm(temporal_features)

        # Return predictions for all time steps
        return self.classifier(attended)  # [batch, seq_len, num_classes, time_patches]
```

### 4.3 Implementation Considerations
- **Memory Usage**: 2-3x increase for sequence modeling
- **Training Complexity**: More complex loss computation across sequences
- **Inference Adaptation**: Need to handle sequential prediction aggregation

## Implementation Timeline

### Week 1: Phase 1 Implementation
- **Days 1-2**: Implement `SlidingWindowDataset` class
- **Days 3-4**: Integrate with existing training pipeline
- **Days 5-7**: Testing, debugging, and optimization

### Week 2: Phase 1 Validation
- **Days 1-3**: Train models and compare F1 scores
- **Days 4-5**: Hyperparameter tuning (windows_per_song, jitter amount)
- **Days 6-7**: Documentation and code review

### Week 3: Phase 2 Implementation
- **Days 1-3**: Implement sliding window inference
- **Days 4-5**: Implement prediction aggregation methods
- **Days 6-7**: Integration testing and validation

### Week 4: Phase 3 Evaluation
- **Days 1-3**: Test 30-second window configurations
- **Days 4-5**: Memory profiling and performance optimization
- **Days 6-7**: Comparative analysis and recommendations

### Future: Phase 4 (Optional)
- Implementation timeline TBD based on results from Phases 1-3
- Estimated 2-3 weeks for full sequential modeling implementation

## Risk Assessment and Mitigation

### Technical Risks
1. **Memory Constraints**: 30s windows may exceed available memory
   - *Mitigation*: Implement graceful degradation to smaller batch sizes

2. **Training Instability**: Sliding windows may introduce training noise
   - *Mitigation*: Careful validation monitoring and early stopping

3. **Inference Performance**: Overlapping windows may be too slow
   - *Mitigation*: Configurable overlap ratio, parallel processing

### Implementation Risks
1. **Code Complexity**: Sliding window logic may introduce bugs
   - *Mitigation*: Comprehensive unit testing and gradual rollout

2. **Data Loading Performance**: Random window extraction may be slow
   - *Mitigation*: Caching strategies and optimized memory mapping

## Success Metrics

### Phase 1 Success Criteria
- [ ] F1 score improvement from 0.19 to >0.4
- [ ] All drum classes show learning (recall > 0.05)
- [ ] Training completes without memory errors
- [ ] Training examples increase by 10x+

### Phase 2 Success Criteria
- [ ] Boundary event detection improves by 5%+
- [ ] Overall F1 score improves by 3-5%
- [ ] Inference time increases by <50%

### Phase 3 Success Criteria
- [ ] 30s windows show measurable improvement over 20s
- [ ] Memory usage remains within acceptable limits
- [ ] Musical transition modeling improves

## Configuration Management

### New Configuration Parameters
```python
@dataclass
class SlidingWindowConfig:
    # Training parameters
    windows_per_song: int = 10
    window_jitter_ratio: float = 0.1  # ±10% jitter

    # Inference parameters
    inference_overlap_ratio: float = 0.25
    aggregation_method: str = 'weighted'  # 'weighted', 'confidence', 'nms'

    # Window sizing
    window_length_seconds: float = 20.0
    enable_sequential_windows: bool = False
    sequence_length: int = 2
```

### Backward Compatibility
- Maintain support for existing non-sliding training mode
- Add feature flags to enable/disable sliding window components
- Preserve existing model checkpoint compatibility

## Testing Strategy

### Unit Tests
- Window extraction logic with edge cases
- Random seed consistency across epochs
- Memory usage validation
- Prediction aggregation correctness

### Integration Tests
- Full training pipeline with sliding windows
- Inference pipeline with overlapping segments
- End-to-end model performance validation

### Performance Tests
- Memory usage profiling under various configurations
- Training speed comparison (sliding vs non-sliding)
- Inference latency benchmarking

## Documentation Requirements

### Code Documentation
- Comprehensive docstrings for all new classes/methods
- Usage examples and configuration guides
- Architecture decision records for major design choices

### User Documentation
- Training configuration guide for sliding windows
- Inference optimization recommendations
- Troubleshooting guide for common issues

## Conclusion

This sliding window implementation plan addresses the fundamental training issues in the Chart Hero drum transcription model while preserving the benefits of full-song musical context. The phased approach allows for incremental validation and risk mitigation, with clear success criteria and fallback options.

The expected outcomes include:
- **Immediate**: 2x+ improvement in F1 scores through proper sequence handling
- **Short-term**: Better boundary event detection and prediction quality
- **Long-term**: Advanced temporal modeling capabilities for state-of-the-art performance

Implementation should begin with Phase 1 to address the most critical training issues, followed by graduated implementation of inference improvements and advanced features based on demonstrated success.
