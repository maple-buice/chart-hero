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
```bash
python model_training/train_model.py
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
Models are stored in `model_training/model/` and use Keras/TensorFlow. The current architecture processes mel-spectrograms to predict drum hit probabilities across multiple drum types simultaneously.

### File Structure Patterns
- Training data is organized in batches with naming pattern: `{batch_num}_{mode}_{type}.npy`
- Model checkpoints use backup and restore functionality
- Audio processing uses librosa for feature extraction

## Notes for Development

When working with this codebase:
- The main pipeline in `main.py` has most processing steps commented out - uncomment as needed
- Model training uses efficient batch loading via Keras Sequence classes
- Audio fingerprinting integrates with AcousticBrainz and Audd APIs
- The system is designed to work with both local audio files and YouTube videos