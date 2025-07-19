The `model_training` directory contains all the scripts and modules related to training the drum chart generation model. This includes data preparation, model definition, training loops, and experiment management.

- **`train_transformer.py`**: The main orchestration script for training the transformer model.
- **`lightning_module.py`**: Contains the core `DrumTranscriptionModule`, the PyTorch Lightning module for the model.
- **`training_setup.py`**: Provides helper functions for setting up the training environment, including argument parsing and configuration.
- **`transformer_model.py`**: Defines and implements the core transformer model architecture.
- **`transformer_data.py`**: Contains the PyTorch Lightning DataModule for the transformer model.
- **`transformer_config.py`**: Configuration file for the transformer model.
- **`data_preparation.py`**: Scripts for preparing the dataset for training.
- **`augment_audio.py`**: A script for augmenting audio data.
- **`run_experiments.py`**: A script for running a series of training experiments.
