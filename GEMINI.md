This project uses a transformer-based deep learning model to automatically generate drum charts from audio files. It analyzes audio to identify drum patterns and creates corresponding MIDI charts for use in games like Clone Hero.

The project is organized into the following main directories:

- **`src/chart_hero`**: The core Python source code for the project, including model training and inference logic.
- **`models`**: Contains trained transformer models.
- **`scripts`**: A collection of utility scripts for tasks like running experiments and analyzing logs.
- **`tests`**: Contains all the tests for the project.
- **`colab`**: Jupyter notebooks for training and experimentation in Google Colab.
- **`datasets`**: Directory for storing and processing datasets.
- **`lightning_logs` / `wandb`**: Directories for experiment tracking and logging from PyTorch Lightning and Weights & Biases.
