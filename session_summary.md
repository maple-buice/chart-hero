# Chart-Hero Session Summary

## Overall Goal
Refactor the data pipeline to be frame-based and deterministic, and then verify the new architecture by training a toy model.

## Key Knowledge
- The data preparation pipeline was architecturally flawed, using a "one-label-per-clip" approach that is unsuitable for transcribing fast, complex drum patterns.
- The new architecture uses a frame-based approach, where the model predicts a label for each time frame of the spectrogram.
- The data pipeline has multiple sources of non-determinism (e.g., `torch.utils.data.random_split`, `DataLoader` shuffling, `torch.randn`) that need to be controlled to create a stable, reproducible process.
- The `replace` and `sed` tools have been unreliable and have led to repeated errors. `write_file` is a more robust alternative for file modifications, but requires careful application.
- The `torch.load` function in PyTorch 2.6 defaults to `weights_only=True`, which can cause errors when loading non-weight tensors.
- The model's expected input shape is `[batch_size, channels, time, freq]`. The repeated errors have been caused by the data pipeline producing tensors with an incorrect number of dimensions (either 3D or 5D instead of 4D).

## File System State
- **CWD**: `/Users/maple/Repos/chart-hero`
- **MODIFIED**: `src/chart_hero/model_training/data_preparation.py` - In the process of removing the `collate_fn` and moving segmentation logic to `NpyDrumDataset`.
- **MODIFIED**: `src/chart_hero/model_training/transformer_data.py` - Will be modified to include segmentation logic in `NpyDrumDataset` and remove `collate_fn` from `create_data_loaders`.
- **MODIFIED**: `src/chart_hero/model_training/train_transformer.py` - Modified to handle the new data shapes and downsample labels before loss calculation.
- **CREATED**: `tests/model_training/test_data_integrity.py` - A new test to ensure the data pipeline is deterministic.
- **CREATED**: `tests/assets/golden_input/` and `tests/assets/golden_data/` - New directories for the data pipeline integrity test.

## Recent Actions & Failures
- Multiple failed attempts to modify `src/chart_hero/model_training/data_preparation.py` and `src/chart_hero/model_training/train_transformer.py` using `replace`, `write_file`, and `sed`.
- The core issue has been a persistent inability to correctly modify the data pipeline, leading to tensors with incorrect dimensions being passed to the model. The `collate_fn` in `data_preparation.py` was identified as a primary source of these issues, as it was incorrectly adding a batch dimension before saving files, and subsequent attempts to fix the issue in the training script were treating the symptom rather than the cause.

## Current Plan
1.  **[IN PROGRESS]** Refactor the data pipeline to be frame-based and deterministic.
    -   [DONE] Refactor `EGMDRawDataset` to produce full-length spectrograms and label matrices.
    -   [DONE] Refactor the model to produce frame-by-frame predictions.
    -   [IN PROGRESS] Remove the `collate_fn` from `data_preparation.py`.
    -   [TODO] Move the segmentation logic to `NpyDrumDataset` in `transformer_data.py`.
    -   [TODO] Remove the `collate_fn` from the `DataLoader`s in `create_data_loaders`.
    -   [TODO] Ensure the data is saved and loaded with the correct dimensions (3D for spectrograms, 2D for labels) at all stages.
2.  **[TODO]** Run the training script to verify the new architecture.
3.  **[TODO]** Commit the architectural rework.
