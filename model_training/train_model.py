import os
from array import array
from pathlib import Path
import random

import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf

from keras.api.models import Sequential, load_model
from keras.api.callbacks import BackupAndRestore, EarlyStopping, History

from matplotlib import pyplot as plt

from model_training.create_model import create_model, get_batch_training_labels_array, get_data_array, get_first_match_training_features_file
from model_training.label_data import get_drum_hits_as_strings
from utils.file_utils import get_labeled_audio_set_dir, get_model_backup_dir, get_model_file

def get_batch_files(mode: str) -> list:
    # print(f'get_batch_files(\'{mode}\')')
    
    batch_files = []
    labeled_audio_set_dir = get_labeled_audio_set_dir()
    
    for file in os.listdir(labeled_audio_set_dir):
        if not file.endswith('.npy'): # ignore directories and non-numpy files
            continue
        if not mode in file:
            continue
        
        batch_file = os.path.join(labeled_audio_set_dir, file)
        
        # print(f'get_batch_files(\'{mode}\') appending \'{batch_file}\'')
        batch_files.append(batch_file)
    
    return batch_files

def get_batch_number(file: str) -> int:
    return int(Path(file).stem.split('_')[0])

def get_batch_numbers(mode: str) -> array[int]:
    batch_number_set = set()
    
    for file in get_batch_files(mode):
        batch_number_set.add(get_batch_number(file))
    
    return array('i', batch_number_set)

def get_total_rows(mode: str) -> int:
    total_rows = 0
    batch_files = get_batch_files(mode)
    
    if batch_files is None or len(batch_files) == 0:
        raise FileNotFoundError(f'No batch files found for mode {mode}')
    
    for file in get_batch_files(mode):
        total_rows += len(np.load(file))
    
    return total_rows

# making this a util function in case I decide to swap the order again...
def get_batch_file_name(batch_numbers: array[int], batch_number: int, mode: str, file_suffix: str) -> str:
    return f'{batch_numbers[batch_number]}_{mode}_{file_suffix}.npy'

def get_features(batch_numbers: array[int], batch_number: int, mode: str) -> np.ndarray:
    return np.load(
        os.path.join(
            get_labeled_audio_set_dir(), 
            get_batch_file_name(batch_numbers, batch_number, mode, 'mel')
            )
        )

def get_labels(batch_numbers: array[int], batch_number: int, mode: str) -> np.ndarray:
    return np.load(
        os.path.join(
            get_labeled_audio_set_dir(), 
            get_batch_file_name(batch_numbers, batch_number, mode, 'label')
            )
        )

def get_generator(batch_size: int, mode: str):
    batch_numbers = get_batch_numbers(mode)
    
    while True:
        batch_index = random.randint(0, len(batch_numbers)-1)
        features = get_features(batch_numbers, batch_index, mode)
        labels = get_labels(batch_numbers, batch_index, mode)
        
        # Create empty arrays to contain batch of features and labels
        batch_features = np.zeros((batch_size, features.shape[1], features.shape[2], 1))
        batch_labels = np.zeros((batch_size, labels.shape[1]))
        
        for i in range(batch_size):
            # Choose random index in features
            index = random.randint(0, len(features)-1)
            batch_features[i] = features[index] if 0 <= index < len(features) else None
            batch_labels[i] = labels[index] if 0 <= index < len(labels) else None
        yield batch_features, batch_labels

def train(model: Sequential) -> History:
    batch_size = 1024
    portion_of_dataset_per_epoch = 1
    
    print(f'Training model.\n\t' +
          f'Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}\n\t' +
          f'Portion of dataset per epoch: 1/{portion_of_dataset_per_epoch}')
    
    history: History = model.fit(
        get_generator(batch_size, 'train'),
        steps_per_epoch = get_total_rows('train') // portion_of_dataset_per_epoch // batch_size,
        validation_data = get_generator(batch_size, 'test'),
	    validation_steps = get_total_rows('test') // portion_of_dataset_per_epoch // batch_size,
        epochs = 25,
        batch_size=batch_size,
        callbacks=[
            EarlyStopping(patience=5, monitor = 'val_accuracy'), # to prevent overfitting
            BackupAndRestore(get_model_backup_dir(), 'epoch', True, False), # to restore interrupted training sessions
            ]
        )
    
    if len(history.history.keys()) > 0:
        model.save("my_model.keras")

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
    features_file = get_first_match_training_features_file()
    features_array = get_data_array(features_file)
    labels_array = get_batch_training_labels_array(get_batch_number(features_file))

    result = []
    pred_raw: np.ndarray = model.predict(features_array)
    print(pred_raw.shape)
    pred = np.round(pred_raw)

    for i in range(pred_raw.shape[0]):
        prediction = pred[i]
        if sum(prediction) == 0:
            new = np.zeros(pred_raw.shape[1])
            new[pred_raw[i].argmax()] = 1
            result.append(new)
        else:
            result.append(prediction)

    print(classification_report(labels_array, np.array(result), target_names=get_drum_hits_as_strings()))

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
        model.save(get_model_file())
        plot_history(history)
    
    # %%
    # Network evaluation
    evaluate_network(model)
