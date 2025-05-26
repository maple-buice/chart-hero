# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chart-Hero is a machine learning project that automatically transcribes drum parts from audio files and creates charts for Clone Hero (a Guitar Hero-like rhythm game). The system uses deep learning models to detect drum hits and converts them into playable game charts.

## Key Architecture Components

### Main Pipeline (main.py)
The entry point accepts either YouTube links or local audio files and processes them through:
1. Song identification using audio fingerprinting
2. Drum track extraction and separation 
3. ML-based drum hit prediction
4. Chart generation for Clone Hero

### Core Modules
- `inference/`: Contains the main inference pipeline
  - `song_identifier.py`: Audio fingerprinting and metadata extraction
  - `input_transform.py`: Audio preprocessing and drum extraction
  - `prediction.py`: ML model inference for drum hit detection
  - `charter.py`: Converts predictions to musical notation/charts
- `model_training/`: Complete training pipeline for the ML models
  - `data_preparation.py`: Prepares training datasets
  - `create_model.py`: Defines neural network architecture
  - `train_model.py`: Main training script with batch processing
- `utils/`: Shared utilities for file operations

### Model Training Flow
The training process uses:
1. MIDI drum data from Google's Groove dataset
2. Audio augmentation techniques for data diversity
3. Mel-spectrogram features for audio representation
4. Multi-class classification for different drum types (kick, snare, hi-hat, etc.)

## Development Commands

### Running the Main Application
```bash
python main.py -l "https://youtube.com/watch?v=..." -km performance
python main.py -p "path/to/audio.wav" -km speed
```

### Model Training

**Legacy CNN Training:**
```bash
python model_training/train_model.py
```

**Data Preparation for Transformer Training:**
```bash
# Prepare E-GMD dataset with memory-efficient processing
python prepare_egmd_data.py --sample-ratio 0.1 --memory-limit-gb 32 --high-performance

# Conservative mode for low-memory systems
python prepare_egmd_data.py --sample-ratio 0.1 --conservative

# Test data preparation
python prepare_egmd_data.py --sample-ratio 0.01 --memory-limit-gb 8 --high-performance --n-jobs 4
```

**Transformer Training (Phase 1 Complete & Functional):**
```bash
# Setup transformer environment
python setup_transformer.py

# Test installation
python model_training/test_transformer_setup.py

# Full training (auto-detects local M1-Max or cloud config)
python model_training/train_transformer.py --config auto --data-dir datasets/processed --audio-dir datasets/e-gmd-v1.0.0

# Quick test run (1 epoch, small batches)
python model_training/train_transformer.py --config auto --data-dir datasets/processed --audio-dir datasets/e-gmd-v1.0.0 --quick-test

# With W&B logging (requires wandb.login())
python model_training/train_transformer.py --config auto --data-dir datasets/processed --audio-dir datasets/e-gmd-v1.0.0 --use-wandb
```

### Dependencies
Install requirements:
```bash
pip install -r requirements.txt
```

## Important Implementation Details

### Drum Mapping
The system maps MIDI drum notes to Clone Hero chart positions using a comprehensive mapping defined in `model_training/README.md`. Key mappings:
- Kick (MIDI 36) → Chart position 0
- Snare (MIDI 38) → Chart position 1
- Hi-hat/Cymbals → Chart positions 66-68
- Toms → Chart positions 2-4

### Model Architecture
**Current (Legacy)**: Models are stored in `model_training/model/` and use Keras/TensorFlow CNN architecture that processes mel-spectrograms to predict drum hit probabilities across multiple drum types simultaneously.

**Transformer Implementation (Phase 1 Complete & Functional)**: The project has been modernized with transformer-based architectures following the comprehensive plan in `TRANSFORMER_REFACTORING_PLAN.md`. The new system includes:
- ✅ Audio Spectrogram Transformer (AST) with patch-based tokenization implemented
- ✅ PyTorch Lightning training framework with mixed precision support
- ✅ Configuration classes for both local training (M1-Max MacBook Pro) and cloud training (Google Colab)
- ✅ Enhanced data pipeline with transformer-compatible spectrogram processing
- ✅ Complete model architecture with 85M+ parameters
- ✅ Memory-efficient data preparation with parallel processing controls
- ✅ Fixed audio loading pipeline to use pre-processed data
- ✅ Working training pipeline with optional W&B logging
- 🔄 Expected 15-20% improvement in F1-score over CNN baseline (to be validated in Phase 2)

### File Structure Patterns
- **Legacy**: Training data organized in batches with naming pattern: `{batch_num}_{mode}_{type}.npy`
- **Transformer**: Training data organized as `{batch_num}_{mode}.pkl` files in `datasets/processed/`
- Model checkpoints use backup and restore functionality
- Audio processing uses librosa for feature extraction and pre-processing

## Notes for Development

When working with this codebase:
- The main pipeline in `main.py` has most processing steps commented out - uncomment as needed
- **Legacy**: Model training uses efficient batch loading via Keras Sequence classes
- **Transformer**: Uses PyTorch DataLoaders with patch-based spectrograms and pre-processed audio data
- Audio fingerprinting integrates with AcousticBrainz and Audd APIs
- The system is designed to work with both local audio files and YouTube videos

### Memory Management & Performance
- **Data Preparation**: Uses aggressive memory monitoring and limits to prevent system crashes
- **Parallel Processing**: Automatically switches to sequential mode for memory limits ≤48GB due to librosa memory explosion issues
- **Batch Processing**: Respects specified batch limits and stops processing at target counts
- **Training**: Supports both CPU and MPS (Apple Silicon) acceleration with mixed precision

### Training Environments
- **Local (M1-Max)**: Optimized for Apple Silicon with MPS acceleration
- **Cloud (Google Colab)**: Configured for GPU training with appropriate batch sizes
- **Auto-detection**: Automatically selects optimal configuration based on available hardware