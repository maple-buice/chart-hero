**TASK DESCRIPTION:**
The primary goal is to successfully run hyperparameter tuning experiments using the `model_training/run_experiments.py` script. This script, in turn, calls `model_training/train_transformer.py` for each experiment. The user wants to iterate on fixing errors until the experiments run.

**COMPLETED:**
1.  Initial check confirmed `train_transformer.py` supports the `--hidden-size` argument.
2.  Multiple attempts were made to run `run_experiments.py`, each revealing new errors that were subsequently addressed:
    *   **Argument Mismatch:** Corrected `run_experiments.py` to pass command-line arguments to `train_transformer.py` in kebab-case (e.g., `--batch-size`) instead of snake_case (`--batch_size`).
    *   **TypeError in path handling:** Fixed `TypeError: 'property' object is not callable` in `utils/file_utils.py` by changing `Path.parent(os.getcwd())` to `Path(os.getcwd()).parent`.
    *   **Audio Directory Logic:** Modified `train_transformer.py` to more robustly determine `config.audio_dir`, prioritizing CLI arguments, then loaded config values, before attempting to derive it, to prevent incorrect overrides.
    *   **Config File Issues (`transformer_config.py`):**
        *   Addressed `ImportError: No module named 'google.colab'` by adding a `# type: ignore` comment.
        *   Adjusted `validate_config` to correctly handle model save paths for different configs, ensuring `LocalConfig` uses `model_dir`.
        *   Ensured `LocalConfig.audio_dir` is set to `"datasets/processed/"`.
3.  Log files were reviewed after each failed run to diagnose issues. The latest one read was `/Users/maple/Repos/chart-hero/logs/experiment_runner_20250526_232635.log`.

**PENDING:**
1.  **Resolve Current Error:** The immediate next step for the new session is to review the latest log file: `/Users/maple/Repos/chart-hero/logs/experiment_runner_20250526_232635.log`.
    *   The last observed error in this log is: `FileNotFoundError: No training data found in datasets/`. This error originates from `model_training/transformer_data.py` (line 291) during the call to `create_data_loaders(config=config, data_dir=config.data_dir, audio_dir=config.audio_dir)`.
2.  **Successfully run `model_training/run_experiments.py`**.

**CURRENT_STATE:**
*   The `run_experiments.py` script is still failing.
*   The `LocalConfig` in `transformer_config.py` uses `data_dir = "datasets/"` and `audio_dir = "datasets/processed/"`.
*   The error `FileNotFoundError: No training data found in datasets/` suggests that the `create_data_loaders` function in `transformer_data.py` expects `config.data_dir` (which is currently `"datasets/"`) to point to a directory containing specific training files or a certain structure, which it's not finding directly within the `"datasets/"` root. The actual audio files are in `"datasets/processed/"` as confirmed by the user.

**CODE STATE (files discussed or modified):**
*   `/Users/maple/Repos/chart-hero/model_training/run_experiments.py`
*   `/Users/maple/Repos/chart-hero/model_training/train_transformer.py`
*   `/Users/maple/Repos/chart-hero/utils/file_utils.py`
*   `/Users/maple/Repos/chart-hero/model_training/transformer_config.py`
*   `/Users/maple/Repos/chart-hero/model_training/transformer_data.py` (error source)
*   `/Users/maple/Repos/chart-hero/datasets/` (directory structure confirmed)
*   `/Users/maple/Repos/chart-hero/datasets/processed/` (confirmed location of audio data)
*   Log files in `/Users/maple/Repos/chart-hero/logs/`, specifically `experiment_runner_20250526_232635.log` is the most recent.

**CHANGES (Key code edits):**
*   **`run_experiments.py`**: Argument passing to `train_transformer.py` changed from snake_case to kebab-case.
    ```python
    # In run_experiment function
    # ...existing code...
    for key, value in params.items():
        cli_key = key.replace("_", "-") # Corrected line
        cmd.extend([f"--{cli_key}", str(value)])
    # ...existing code...
    ```
*   **`utils/file_utils.py`**: Corrected `Path.parent` usage.
    ```python
    # In get_training_data_dir and get_model_training_dir
    # ...existing code...
    return get_dir(Path(os.getcwd()).parent, dir_name, False) # Corrected usage
    # ...existing code...
    ```
*   **`train_transformer.py`**: Refined `audio_dir` determination logic.
    ```python
    # In main function
    # ...existing code...
    if args.audio_dir:
        config.audio_dir = args.audio_dir
    elif not getattr(config, 'audio_dir', None):
        # ... try to derive ...
    else:
        # use config.audio_dir from loaded config
    # ...existing code...
    ```
*   **`transformer_config.py`**:
    *   Added `# type: ignore` to `import google.colab`.
    *   `validate_config`: Modified to correctly find model save paths for different configs.
        ```python
        # In validate_config function
            # ...existing code...
            if hasattr(config, 'model_save_path') and isinstance(config.model_save_path, str): # For LocalConfig before change
                os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
            elif hasattr(config, 'model_dir') and isinstance(config.model_dir, str): # For CloudConfig and expected by train_transformer.py
                os.makedirs(config.model_dir, exist_ok=True)
            # ...existing code...
        ```
    *   `LocalConfig`: Ensured `audio_dir` is `"datasets/processed/"` and uses `model_dir`.
        ```python
        @dataclass
        class LocalConfig(BaseConfig):
            # ...existing code...
            audio_dir: str = "datasets/processed/"
            # model_save_path: str = "models/transformer_model.pt" # Original line
            model_dir: str = "model_training/transformer_models/" # Changed for consistency
            # ...existing code...
        ```

The immediate focus for a new session should be on the `FileNotFoundError` in the latest logs and why `transformer_data.py` isn't finding what it needs in `config.data_dir` (currently `"datasets/"`).
