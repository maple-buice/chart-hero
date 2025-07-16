The `inference` directory contains the necessary modules and scripts to run the trained drum chart generation model on new audio files. It includes components for transforming input audio, identifying songs, and predicting drum patterns to create a final chart.

- **charter.py:** Orchestrates the chart generation process.
- **input_transform.py:** Handles the transformation of input audio into a format suitable for the model.
- **prediction.py:** Uses the trained model to predict drum patterns.
- **song_identifier.py:** Identifies song information using external APIs.
- **classes/audd.py:** A class for interacting with the AudD music identification API.
