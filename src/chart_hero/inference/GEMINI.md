The `inference` directory contains the necessary modules and scripts to run the trained drum chart generation model on new audio files. It includes components for transforming input audio, identifying songs, and predicting drum patterns to create a final chart.

- **`charter.py`**: Orchestrates the chart generation process.
- **`input_transform.py`**: Handles the transformation of input audio into a format suitable for the model.
- **`song_identifier.py`**: Identifies song information using external APIs.
- **`classes/`**: Contains helper classes for inference tasks.
  - **`audd.py`**: A class for interacting with the AudD music identification API.
- **`pretrained_models/`**: Directory for storing pretrained models used in inference.
