# chart-hero
Project to use ML and AI techniques to automatically chart songs to play in Clone Hero.

## Project Status

This project has been significantly updated to use a modern transformer-based architecture for drum transcription. Key features include:

- **Audio Spectrogram Transformer (AST)**: Utilizes an 85M+ parameter model for improved accuracy.
- **PyTorch Lightning**: Employs a robust training pipeline with mixed precision support.
- **Memory-Efficient Data Processing**: Capable of handling large datasets.
- **Multi-Environment Support**: Configured for both local (M1-Max) and cloud (Google Colab) training.

For detailed information on model training, see `model_training/README.md`.
For the refactoring plan and current progress, see `TRANSFORMER_REFACTORING_PLAN.md`.

## Quick Start

Refer to `model_training/README.md` for data preparation and training commands.
