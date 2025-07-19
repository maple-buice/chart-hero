# Guide: Migrating to Poetry

This document outlines the steps to migrate this project's dependency management from `requirements.txt` to Poetry. This process was attempted and reverted due to a complex build dependency issue with `llvmlite`. This guide documents the correct path forward for a future attempt.

## Prerequisites

1.  **Install Poetry:**
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```
    Ensure the Poetry `bin` directory is added to your shell's `PATH`.

2.  **Install LLVM:** The `llvmlite` package, a dependency of `librosa`, requires the LLVM compiler toolchain to be installed.
    ```bash
    brew install llvm
    ```

## Migration Steps

1.  **Initialize the Project:**
    This command will create the `pyproject.toml` file with the correct initial settings for Python 3.12.
    ```bash
    poetry init --name chart-hero --version 0.1.0 --description "Automatic drum chart generation from audio files." --author "Maple Buice <maple.buice@gmail.com>" --python "~3.12" --license MIT -n
    ```

2.  **Add Dependencies:**
    This is the most critical step. The `llvmlite` package needs a specific environment variable to find the `llvm-config` executable provided by Homebrew. We will prepend this variable to the `poetry add` command.

    *   **Add Main Dependencies:**
        ```bash
        LLVM_CONFIG=$(brew --prefix llvm)/bin/llvm-config poetry add \
          torch torchaudio pytorch-lightning transformers timm wandb \
          librosa scipy numpy pandas scikit-learn tqdm matplotlib \
          seaborn psutil music21 soundfile joblib audiomentations six
        ```

    *   **Add Development Dependencies:**
        ```bash
        poetry add --group dev \
          jupyter ipywidgets black isort ruff mypy pyfakefs
        ```

3.  **Verify Installation:**
    After the dependencies are added, you can verify that the environment is correctly installed.
    ```bash
    poetry install
    poetry run pytest
    ```

4.  **Remove `requirements.txt`:**
    Once the `poetry.lock` file is generated and the installation is successful, the `requirements.txt` file can be removed.
    ```bash
    git rm requirements.txt
    ```

By following these steps, the migration should proceed smoothly, correctly handling the `llvmlite` build issue from the start.
