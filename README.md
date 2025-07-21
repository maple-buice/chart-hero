# chart-hero
Project to use ML and AI techniques to automatically chart songs to play in Clone Hero.

## Project Status

This project has been significantly updated to use a modern transformer-based architecture for drum transcription. Key features include:

- **Audio Spectrogram Transformer (AST)**: Utilizes an 85M+ parameter model for improved accuracy.
- **PyTorch Lightning**: Employs a robust training pipeline with mixed precision support.
- **Memory-Efficient Data Processing**: Capable of handling large datasets.
- **Multi-Environment Support**: Configured for both local (M1-Max) and cloud (Google Colab) training.

For detailed information on model training, see `model_training/README.md`.

## Never delete

If you're having trouble with getting all the packages and MPS to get along, try `pip3 install numpy --pre torch torchvision torchaudio numba --extra-index-url https://download.pytorch.org/whl/nightly/cpu --force-reinstall`!

## Setup

This project requires an `arm64` build of Python 3.11 on Apple Silicon machines to use the MPS accelerator.

### 1. Install Python 3.11 with Homebrew

If you don't have an `arm64` build of Python 3.11, you can install it with Homebrew:

```bash
brew install python@3.11
```

### 2. Create and activate a virtual environment

Create a new virtual environment using your `arm64` Python installation:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv chart-hero-venv
```

Activate the virtual environment:

```bash
source chart-hero-venv/bin/activate
```

### 3. Install dependencies

Install the required dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Install the `chart-hero` package

Finally, install the `chart-hero` package in editable mode:

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

## Troubleshooting

### ModuleNotFoundError: No module named 'typing_extensions'

If you encounter this error, it may be due to a version conflict with `pytorch-lightning`. To resolve this, force a reinstallation of `pytorch-lightning` and `lightning-utilities`:

```bash
pip install --force-reinstall --no-deps pytorch-lightning lightning-utilities
```
