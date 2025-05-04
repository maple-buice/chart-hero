import gc
from math import floor
import os
from array import array
from pathlib import Path
import random

import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.utils import Sequence # Use tf.keras

from keras.api.models import Sequential, load_model
from keras.api.callbacks import BackupAndRestore, EarlyStopping, History, ModelCheckpoint # Add ModelCheckpoint

from matplotlib import pyplot as plt

from model_training.create_model import create_model, get_batch_training_labels_array, get_data_array, get_features_shape, get_first_match_training_features_file, get_labels_shape
from model_training.data_preparation import get_number_of_audio_set_batches
from model_training.label_data import get_drum_hits_as_strings
from utils.file_utils import get_labeled_audio_set_dir, get_model_backup_dir, get_model_file

# --- Keras Sequence for Efficient Batch Loading ---
class BatchSequence(Sequence):
    """Generates batches of data from .npy files."""
    def __init__(self, mode: str, batch_size: int, shuffle=True):
        self.mode = mode
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.labeled_audio_set_dir = get_labeled_audio_set_dir()
        self.batch_files = self._get_batch_files()
        self.features_shape, self.labels_shape = self._get_shapes()
        self.indices = np.arange(len(self.batch_files)) # Indices for shuffling batches
        self.on_epoch_end() # Initial shuffle

    def _get_batch_files(self) -> list[tuple[str, str]]:
        """Gets pairs of (feature_file, label_file) paths."""
        files = {} # {batch_num: {'mel': path, 'label': path}}
        for file in os.listdir(self.labeled_audio_set_dir):
            if not file.endswith('.npy') or self.mode not in file:
                continue
            
            parts = Path(file).stem.split('_')
            if len(parts) < 3:
                continue # Skip improperly named files
                
            batch_num = int(parts[0])
            file_type = parts[-1] # 'mel' or 'label'
            
            if batch_num not in files:
                files[batch_num] = {}
            
            files[batch_num][file_type] = os.path.join(self.labeled_audio_set_dir, file)

        # Create list of pairs, ensuring both mel and label exist
        file_pairs = []
        for batch_num in sorted(files.keys()):
            if 'mel' in files[batch_num] and 'label' in files[batch_num]:
                file_pairs.append((files[batch_num]['mel'], files[batch_num]['label']))
            else:
                print(f"Warning: Missing mel or label file for batch {batch_num} in mode '{self.mode}'. Skipping batch.")
        
        if not file_pairs:
             raise FileNotFoundError(f"No complete batch file pairs found for mode '{self.mode}' in {self.labeled_audio_set_dir}")
             
        return file_pairs

    def _get_shapes(self):
        """Infers feature and label shapes from the first batch file."""
        try:
            features = np.load(self.batch_files[0][0])
            labels = np.load(self.batch_files[0][1])
            # Shape: (samples_in_batch, height, width, channels), (samples_in_batch, num_classes)
            return features.shape[1:], labels.shape[1:] 
        except Exception as e:
            print(f"Error loading shapes from {self.batch_files[0]}: {e}")
            raise

    def __len__(self):
        """Denotes the number of batches per epoch."""
        # Each file pair represents a pre-defined batch of samples
        # The generator yields one sample at a time from these files
        # Keras' fit handles the batch_size grouping
        # So, length is the total number of samples / batch_size
        # Let's estimate total samples (can be slow if files are huge)
        # A better way might be to store total samples per mode elsewhere
        # For now, approximate based on first batch file size * num files
        # TODO: Find a more accurate way to get total samples without loading all files
        samples_in_first_batch = np.load(self.batch_files[0][0]).shape[0]
        total_samples_estimate = samples_in_first_batch * len(self.batch_files)
        return int(np.ceil(total_samples_estimate / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Calculate start and end sample indices for the overall dataset
        start_sample_idx = index * self.batch_size
        end_sample_idx = (index + 1) * self.batch_size

        # Determine which files contain these samples
        # This requires knowing samples per file - assuming constant for now (simplification)
        # TODO: Handle variable samples per file if necessary
        samples_per_file = np.load(self.batch_files[0][0]).shape[0]
        
        start_file_idx = start_sample_idx // samples_per_file
        end_file_idx = (end_sample_idx - 1) // samples_per_file

        # Adjust indices to be relative within the files being loaded
        start_sample_in_file = start_sample_idx % samples_per_file
        end_sample_in_file = (end_sample_idx - 1) % samples_per_file

        batch_features_list = []
        batch_labels_list = []

        for i in range(start_file_idx, end_file_idx + 1):
            file_pair_index = self.indices[i % len(self.indices)] # Use shuffled index
            feature_file, label_file = self.batch_files[file_pair_index]
            
            features_data = np.load(feature_file)
            labels_data = np.load(label_file)

            # Determine slice indices for this file
            slice_start = start_sample_in_file if i == start_file_idx else 0
            slice_end = end_sample_in_file + 1 if i == end_file_idx else samples_per_file
            
            batch_features_list.append(features_data[slice_start:slice_end])
            batch_labels_list.append(labels_data[slice_start:slice_end])

        # Concatenate data from potentially multiple files
        batch_features = np.concatenate(batch_features_list, axis=0)
        batch_labels = np.concatenate(batch_labels_list, axis=0)
        
        # Ensure the batch size is correct (might be smaller for the last batch)
        # Keras handles this, but good practice
        actual_batch_size = batch_features.shape[0]
        if actual_batch_size != self.batch_size and end_sample_idx < self.__len__() * self.batch_size:
             print(f"Warning: Batch {index} size mismatch. Expected {self.batch_size}, got {actual_batch_size}")

        return batch_features, batch_labels

    def on_epoch_end(self):
        """Updates indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)
# --- End Keras Sequence ---

def train(model: Sequential) -> History:
    batch_size = 128 # Adjusted batch size - tune based on GPU memory
    number_of_epochs = 25
    
    print(f'Training model.\n\t' +
          f'Num GPUs Available: {len(tf.config.list_physical_devices("GPU"))}\n\t' +
          f'Batch Size: {batch_size}')

    # Create Sequence generators
    train_generator = BatchSequence(mode='train', batch_size=batch_size, shuffle=True)
    val_generator = BatchSequence(mode='val', batch_size=batch_size, shuffle=False) # Use 'val' set, no shuffle

    # Define callbacks
    early_stopping = EarlyStopping(patience=5, monitor='val_accuracy', verbose=1) # Monitor val_accuracy
    backup_restore = BackupAndRestore(get_model_backup_dir(), save_freq='epoch')
    model_checkpoint = ModelCheckpoint(
        filepath=get_model_file(), # Save best model to the final path
        monitor='val_accuracy', 
        save_best_only=True, 
        verbose=1
    )
    
    history: History = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=number_of_epochs,
        callbacks=[early_stopping, backup_restore, model_checkpoint],
        workers=4, # Use multiple workers for data loading (adjust based on CPU cores)
        use_multiprocessing=True
    )
    
    return history

def plot_history(history: History):
    if history == None or len(history.history.keys()) > 0:
        print('No history to plot')
        return
    
    # %%
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Training - accuracy history')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # %%
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training - loss history')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def evaluate_network(model: Sequential):
    # Evaluate on the test set using the Sequence
    print("\nEvaluating model on test set...")
    batch_size = 128 # Use same batch size or adjust as needed
    test_generator = BatchSequence(mode='test', batch_size=batch_size, shuffle=False)

    if len(test_generator) == 0:
        print("No test data found to evaluate.")
        return

    # Get all predictions and true labels from the test generator
    # This might be memory intensive for large test sets
    # Alternative: Evaluate in batches and aggregate metrics
    try:
        y_pred_raw = model.predict(test_generator, verbose=1)
        y_pred = np.round(y_pred_raw)
        
        # Get true labels (requires iterating through generator)
        y_true = []
        for i in range(len(test_generator)):
            _, labels_batch = test_generator[i]
            y_true.append(labels_batch)
        y_true = np.concatenate(y_true, axis=0)

        # Adjust predictions where sum is 0 (optional, same logic as before)
        y_pred_adjusted = []
        for i in range(y_pred_raw.shape[0]):
            prediction = y_pred[i]
            if sum(prediction) == 0 and i < y_pred_raw.shape[0]: # Check index boundary
                new = np.zeros_like(prediction)
                new[y_pred_raw[i].argmax()] = 1
                y_pred_adjusted.append(new)
            else:
                y_pred_adjusted.append(prediction)
        y_pred_adjusted = np.array(y_pred_adjusted)
        
        # Ensure shapes match before classification report
        if y_true.shape[0] != y_pred_adjusted.shape[0]:
             print(f"Warning: Mismatch in number of samples between true labels ({y_true.shape[0]}) and predictions ({y_pred_adjusted.shape[0]}). Truncating to minimum.")
             min_samples = min(y_true.shape[0], y_pred_adjusted.shape[0])
             y_true = y_true[:min_samples]
             y_pred_adjusted = y_pred_adjusted[:min_samples]

        print(classification_report(y_true, y_pred_adjusted, target_names=get_drum_hits_as_strings(), zero_division=0))

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

def get_model() -> Sequential:
    model: Sequential = None
    model_file_path = get_model_file()
    
    if os.path.exists(model_file_path):
        model = load_model(model_file_path)
    else:
        model = create_model()
    
    return model

def train_model():
    # %%
    # Get model
    model = get_model()
    model.summary()

    # %%
    # Train model
    history = train(model)
    print(history.history)

    # %%
    # If any training happened, save model and plot history
    if history != None and len(history.history.keys()) > 0:
        plot_history(history)
    
    # %%
    # Network evaluation
    evaluate_network(model)
