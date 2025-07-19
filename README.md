# chart-hero
Project to use ML and AI techniques to automatically chart songs to play in Clone Hero.

## Project Status

This project has been significantly updated to use a modern transformer-based architecture for drum transcription. Key features include:

- **Audio Spectrogram Transformer (AST)**: Utilizes an 85M+ parameter model for improved accuracy.
- **PyTorch Lightning**: Employs a robust training pipeline with mixed precision support.
- **Memory-Efficient Data Processing**: Capable of handling large datasets.
- **Multi-Environment Support**: Configured for both local (M1-Max) and cloud (Google Colab) training.

For detailed information on model training, see `model_training/README.md`.

## Setup

This project uses [Conda](https://docs.conda.io/en/latest/) to manage dependencies. To get started, create the conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

This will create a new conda environment named `chart-hero`. To activate the environment, run:

```bash
conda activate chart-hero
```

After creating the environment, install the `chart-hero` package in editable mode:

```bash
pip install -e .
```

## Quick Start

Refer to `model_training/README.md` for data preparation and training commands.

## Configuration

### AudD API Key

This project uses the AudD API for music identification. To use this feature, you must set the `AUDD_API_TOKEN` environment variable to your AudD API token.

You can get a free API token by signing up at [audd.io](https://audd.io).

**Example:**

```bash
export AUDD_API_TOKEN="your_api_token_here"
```

**Testing the pre-commit hooks.**
