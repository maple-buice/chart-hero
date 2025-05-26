# Chart-Hero Transformer Refactoring Plan

## Executive Summary

This document outlines a comprehensive plan to modernize the Chart-Hero project by replacing the current CNN-based drum transcription model with state-of-the-art transformer architectures. The refactoring will support both local training on M1-Max MacBook Pro (64GB RAM, 1TB storage) and cloud training on Google Colab with GPUs, while leveraging The Expanded Groove MIDI Dataset.

## Current Architecture Analysis

### Existing System
- **Model**: Sequential CNN with Conv2D + MaxPooling + Dense layers
- **Input**: Mel-spectrograms from audio files
- **Output**: Multi-label classification for drum hits (kick, snare, toms, cymbals)
- **Training**: Keras with BatchSequence for efficient data loading
- **Dataset**: Expanded Groove MIDI Dataset (EGMD) with audio-MIDI pairs

### Key Limitations
- CNN architecture cannot capture long-range temporal dependencies effectively
- Limited context window for drum pattern understanding
- No self-attention mechanism for complex rhythmic relationships
- Outdated architecture compared to modern transformer-based audio models

## Proposed Transformer Architecture

### Primary Model: Audio Spectrogram Transformer (AST) + Enhanced Features

**Core Architecture Components:**

1. **Spectrogram Preprocessing**
   - Log-mel spectrograms (128 mel bins, configurable time frames)
   - Patch-based tokenization similar to Vision Transformer (ViT)
   - Positional encoding for time-frequency patches

2. **Transformer Backbone**
   - **Base Model**: Audio Spectrogram Transformer (AST) with 12 layers
   - **Attention**: Multi-head self-attention with 8 heads
   - **Hidden Size**: 768 dimensions
   - **Enhanced Features**: 
     - Hierarchical attention for multi-scale temporal patterns
     - Cross-attention between time and frequency domains
     - Conformer blocks for hybrid CNN-Transformer approach

3. **Output Head**
   - Multi-label classification for drum positions (0-68)
   - Sigmoid activation for independent drum hit probabilities
   - Temporal consistency loss for smooth predictions

### Alternative Architecture: MT3-Inspired Approach

**For Advanced Implementation:**

1. **Encoder-Decoder Architecture**
   - Spectrogram encoder with transformer layers
   - Autoregressive decoder for sequential drum event prediction
   - Token-based output (similar to MT3) for drum events with timing

2. **Advantages**
   - Better temporal modeling of drum sequences
   - Supports variable-length outputs
   - Can model complex polyrhythmic patterns

## Training Infrastructure Design

### Local Training (M1-Max MacBook Pro)

**Hardware Optimization:**
- **Memory Management**: Utilize 64GB RAM for large batch processing
- **Storage Strategy**: Efficient data loading from 1TB SSD
- **Metal Performance Shaders**: Leverage Apple Silicon optimization
- **Mixed Precision**: Use AMP for memory efficiency

**Configuration:**
```python
# Local training config
LOCAL_CONFIG = {
    'batch_size': 32,  # Optimized for 64GB RAM
    'sequence_length': 1024,  # 10-second audio segments
    'learning_rate': 1e-4,
    'warmup_steps': 1000,
    'gradient_checkpointing': True,
    'mixed_precision': True,
    'device': 'mps'  # Metal Performance Shaders
}
```

### Cloud Training (Google Colab)

**GPU Optimization:**
- **T4/V100/A100**: Automatic detection and configuration
- **Distributed Training**: DataParallel for multi-GPU setup
- **Checkpoint Management**: Automatic saving to Google Drive
- **Memory Optimization**: Gradient accumulation for large effective batch sizes

**Configuration:**
```python
# Cloud training config
CLOUD_CONFIG = {
    'batch_size': 64,  # Higher for GPU memory
    'sequence_length': 2048,  # Longer sequences on GPU
    'learning_rate': 2e-4,
    'warmup_steps': 2000,
    'gradient_accumulation_steps': 4,
    'mixed_precision': True,
    'device': 'cuda'
}
```

## Implementation Roadmap

### Phase 1: Foundation Setup (Weeks 1-2) ✅ COMPLETED

1. **Environment Configuration** ✅
   - ✅ Created transformer training environment
   - ✅ Installed PyTorch with MPS (local) and CUDA (cloud) support
   - ✅ Set up transformers library and audio processing dependencies

2. **Data Pipeline Modernization** ✅
   - ✅ Refactored `data_preparation.py` for transformer input format
   - ✅ Implemented patch-based spectrogram tokenization
   - ✅ Created new DataLoader with transformer-compatible processing
   - ✅ Added memory-efficient processing with parallel controls
   - ✅ Fixed audio loading to use pre-processed data

3. **Model Architecture Implementation** ✅
   - ✅ Implemented AST base model with PyTorch (85M+ parameters)
   - ✅ Added drum-specific output heads for multi-label classification
   - ✅ Created configuration classes for local/cloud variants
   - ✅ Implemented patch-based spectrogram processing

### Phase 2: Training Infrastructure (Weeks 3-4) ✅ COMPLETED

1. **Training Loop Refactoring** ✅
   - ✅ Replaced Keras training with PyTorch Lightning
   - ✅ Implemented mixed precision training (16-bit AMP)
   - ✅ Added advanced optimizers (AdamW with weight decay)
   - ✅ Implemented learning rate scheduling (cosine annealing with warmup)
   - ✅ Added gradient clipping and accumulation

2. **Evaluation Metrics** ✅
   - ✅ Implemented multi-label F1-score metrics
   - ✅ Added multi-label accuracy evaluation
   - ✅ Integrated torchmetrics for comprehensive evaluation
   - ✅ Added per-drum-type evaluation capabilities

3. **Checkpoint and Model Management** ✅
   - ✅ Implemented automatic checkpoint saving
   - ✅ Added early stopping and best model selection
   - ✅ Created model saving and loading infrastructure
   - ✅ Added training resume functionality

### Phase 3: Advanced Features (Weeks 5-6)

1. **Self-Supervised Pre-training**
   - Implement masked spectrogram modeling
   - Pre-train on large unlabeled audio datasets
   - Fine-tune on EGMD with drum-specific objectives

2. **Multi-Scale Training**
   - Variable-length sequence training
   - Hierarchical attention for different time scales
   - Data augmentation with time-stretching and pitch-shifting

3. **Model Ensemble and Distillation**
   - Train multiple model variants
   - Knowledge distillation for efficiency
   - Model averaging for improved accuracy

### Phase 4: Integration and Optimization (Weeks 7-8)

1. **Inference Pipeline Update**
   - Update `prediction.py` with transformer inference
   - Optimize for real-time processing
   - Add sliding window inference for long audio

2. **Chart Generation Enhancement**
   - Improve temporal alignment with transformer outputs
   - Add confidence-based filtering
   - Enhanced drum mapping logic

3. **Performance Optimization**
   - Model quantization for deployment
   - ONNX export for cross-platform inference
   - Benchmark against CNN baseline

## Technical Specifications

### Model Architecture Details

```python
class DrumTranscriptionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Spectrogram preprocessing
        self.patch_embed = PatchEmbedding(
            patch_size=config.patch_size,
            embed_dim=config.hidden_size
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(
            config.hidden_size, 
            config.max_seq_len
        )
        
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout,
                activation='gelu'
            ),
            num_layers=config.num_layers
        )
        
        # Drum classification heads
        self.drum_classifier = nn.Linear(
            config.hidden_size, 
            config.num_drum_classes
        )
        
    def forward(self, spectrograms, attention_mask=None):
        # Convert spectrograms to patches
        patches = self.patch_embed(spectrograms)
        
        # Add positional encoding
        patches = self.pos_encoding(patches)
        
        # Transformer processing
        hidden_states = self.transformer(
            patches, 
            src_key_padding_mask=attention_mask
        )
        
        # Classification
        logits = self.drum_classifier(hidden_states)
        
        return logits
```

### Dataset Adaptations

**Enhanced EGMD Processing:**
- Convert audio to log-mel spectrograms with overlap
- Create patch-based representations
- Implement data augmentation pipeline
- Add temporal consistency labels

### Training Optimizations

**Advanced Training Techniques:**
- **Curriculum Learning**: Start with simple patterns, progress to complex
- **Multi-Task Learning**: Joint training on onset detection + drum classification
- **Contrastive Learning**: Learn better drum pattern representations
- **Temporal Consistency Loss**: Ensure smooth predictions over time

## Performance Expectations

### Expected Improvements

1. **Accuracy**: 15-20% improvement in F1-score over CNN baseline
2. **Temporal Consistency**: Better handling of rapid drum sequences
3. **Generalization**: Improved performance on unseen music styles
4. **Context Understanding**: Better modeling of drum pattern relationships

### Benchmarks

**Target Metrics:**
- **Onset F1-Score**: >0.85 (vs current ~0.75)
- **Multi-Label F1**: >0.80 per drum type
- **Inference Speed**: <100ms per 10-second segment
- **Memory Usage**: <8GB for inference

## Risk Mitigation

### Technical Risks

1. **Memory Constraints**: 
   - Mitigation: Gradient checkpointing, optimized batch sizes
   
2. **Training Time**: 
   - Mitigation: Mixed precision, efficient data loading, distributed training
   
3. **Model Complexity**: 
   - Mitigation: Start with smaller models, gradual scaling

### Compatibility Risks

1. **Integration Issues**: 
   - Mitigation: Maintain backward compatibility in inference API
   
2. **Performance Regression**: 
   - Mitigation: Comprehensive benchmarking, A/B testing

## Success Metrics

### Phase 1-2 Achievements ✅
- ✅ **Complete Transformer Implementation**: 85M+ parameter model working
- ✅ **Memory-Efficient Data Pipeline**: Handles large datasets without crashes
- ✅ **Functional Training Pipeline**: PyTorch Lightning + mixed precision
- ✅ **Multi-Environment Support**: Local (M1-Max) and cloud (Colab) configs
- ✅ **Advanced Training Features**: Checkpointing, early stopping, resume

### Quantitative Goals (In Progress/Future)
- [ ] Achieve >85% onset detection F1-score
- [ ] Reduce inference latency by 30%
- [ ] Improve drum type classification accuracy by 20%
- [ ] Maintain <100MB model size for deployment

### Qualitative Goals (In Progress/Future)
- [ ] Better handling of complex polyrhythmic patterns
- [ ] Improved transcription of fast drum fills
- [ ] Enhanced temporal consistency in long tracks
- [ ] Better generalization across music genres

## Resource Requirements

### Development Resources
- **Time**: 8 weeks for complete implementation
- **Compute**: 
  - Local: M1-Max MacBook Pro for development/testing
  - Cloud: Google Colab Pro+ for large-scale training
- **Storage**: ~500GB for datasets and model checkpoints

### Dependencies
- PyTorch 2.0+ with MPS/CUDA support
- transformers library
- torchaudio for audio processing
- PyTorch Lightning for training framework
- wandb for experiment tracking

## Current Status & Next Steps

### Completed Work (Phase 1-2) ✅
The transformer refactoring has successfully completed Phases 1-2, delivering a fully functional training pipeline:

**✅ Core Infrastructure:**
- Complete Audio Spectrogram Transformer implementation (85M+ parameters)
- PyTorch Lightning training framework with mixed precision
- Memory-efficient data preparation with parallel processing controls
- Dual-environment support (M1-Max local + Google Colab cloud)

**✅ Key Innovations:**
- Patch-based spectrogram tokenization for transformer input
- Advanced memory management preventing system crashes
- Pre-processed audio pipeline for efficient training
- Comprehensive evaluation metrics and checkpointing

### Ready for Phase 3: Model Training & Evaluation

**Immediate Next Steps:**
1. **Full Model Training**: Run complete training cycles on processed EGMD data
2. **Performance Evaluation**: Benchmark against CNN baseline
3. **Hyperparameter Optimization**: Fine-tune learning rates, architectures
4. **Advanced Features**: Implement Phase 3 enhancements (self-supervised pre-training, multi-scale training)

### Available Commands for Training:
```bash
# Data preparation
python prepare_egmd_data.py --sample-ratio 0.1 --memory-limit-gb 32 --high-performance

# Full training
python model_training/train_transformer.py --config auto --data-dir datasets/processed --audio-dir datasets/e-gmd-v1.0.0

# Quick validation
python model_training/train_transformer.py --config auto --data-dir datasets/processed --audio-dir datasets/e-gmd-v1.0.0 --quick-test
```

## Conclusion

The transformer modernization has successfully established a robust foundation with working training infrastructure. Phase 1-2 objectives have been fully achieved, positioning the project for advanced model training and evaluation in subsequent phases. The implemented system provides significant architectural improvements while maintaining practical deployment considerations.