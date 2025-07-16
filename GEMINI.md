This project is designed to automatically generate drum charts from audio files. It uses a transformer-based deep learning model to analyze the audio, identify drum patterns, and create corresponding MIDI charts. The system is built to handle large datasets and includes tools for data preparation, model training, inference, and performance analysis.

The project is organized into the following main directories:

- **`src/chart_hero`**: The core Python source code for the project.
- **`model_training`**: Contains trained transformer models. The source code for training is in `src/chart_hero/model_training`.
- **`scripts`**: A collection of utility scripts for tasks like running experiments and analyzing logs.
- **`tests`**: Contains all the tests for the project.
- **`colab`**: Jupyter notebooks for training and experimentation in Google Colab.
- **`datasets`**: Directory for storing and processing datasets.
- **`lightning_logs` / `wandb`**: Directories for experiment tracking and logging.
