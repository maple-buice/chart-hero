import gc
from math import floor
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

from model_training.create_model import create_model, get_batch_training_labels_array, get_data_array, get_features_shape, get_first_match_training_features_file, get_labels_shape
from model_training.data_preparation import get_number_of_audio_set_batches
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
        if not '_mel' in file: # only count feature files so we don't double count
            continue
        
        batch_file = os.path.join(labeled_audio_set_dir, file)
        
        # print(f'get_batch_files(\'{mode}\') appending \'{batch_file}\'')
        batch_files.append(batch_file)
    
    return batch_files

def get_batch_number(file: str) -> int:
    return int(Path(file).stem.split('_')[0])

def get_batch_numbers(mode: str) -> list[int]:
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

def get_total_rows_in_batches(batch_numbers: list[int], mode: str) -> int:
    total_rows = 0
    batch_files = get_batch_files(mode)
    
    if batch_files is None or len(batch_files) == 0:
        raise FileNotFoundError(f'No batch files found for mode {mode}')
    
    for file in batch_files:
        if get_batch_number(file) in batch_numbers:
            total_rows += len(np.load(file))
    
    return total_rows

# making this a util function in case I decide to swap the order again...
def get_batch_file_name(batch_number: int, mode: str, file_suffix: str) -> str:
    return f'{batch_number}_{mode}_{file_suffix}.npy'

def get_batch_file_name_from_list(batch_numbers: list[int], batch_number: int, mode: str, file_suffix: str) -> str:
    return f'{batch_numbers[batch_number]}_{mode}_{file_suffix}.npy'

def get_features_from_list(batch_numbers: list[int], batch_number: int, mode: str) -> np.ndarray:
    return np.load(
        os.path.join(
            get_labeled_audio_set_dir(), 
            get_batch_file_name_from_list(batch_numbers, batch_number, mode, 'mel')
            )
        )

def get_features(batch_number: int, mode: str) -> np.ndarray:
    return np.load(
        os.path.join(
            get_labeled_audio_set_dir(), 
            get_batch_file_name(batch_number, mode, 'mel')
            )
        )

def get_labels_from_list(batch_numbers: list[int], batch_number: int, mode: str) -> np.ndarray:
    return np.load(
        os.path.join(
            get_labeled_audio_set_dir(), 
            get_batch_file_name_from_list(batch_numbers, batch_number, mode, 'label')
            )
        )

def get_labels(batch_number: int, mode: str) -> np.ndarray:
    return np.load(
        os.path.join(
            get_labeled_audio_set_dir(), 
            get_batch_file_name(batch_number, mode, 'label')
            )
        )
    
def get_all_features(batch_numbers: list[int], mode: str) -> np.ndarray:
    total_rows_in_batches = get_total_rows_in_batches(batch_numbers, mode)
    features_shape = get_features_shape()
    features: np.ndarray = np.zeros((
        total_rows_in_batches,
        features_shape[1],
        features_shape[2],
        1))
    
    i = 0
    for batch_number in batch_numbers:
        for feature in get_features(batch_number, mode):
            features[i] = feature
            i += 1
            
    # print(f'get_all_features! Total rows in batches: {total_rows_in_batches}, i: {i}, features: {features}')
    
    return features
    
def get_all_labels(batch_numbers: list[int], mode: str) -> np.ndarray:
    total_rows_in_batches = get_total_rows_in_batches(batch_numbers, mode)
    labels_shape = get_labels(batch_numbers[0], mode).shape
    labels: np.ndarray = np.zeros((
        total_rows_in_batches,
        labels_shape[1]))
    
    i = 0
    for batch_number in batch_numbers:
        for label in get_labels(batch_number, mode):
            labels[i] = label
            i += 1
    
    # print(f'get_all_labels! Total rows in batches: {total_rows_in_batches}, i: {i}, labels: {labels}')
    
    return labels

def get_generator_large_dataset(in_memory_batches: int, batch_size: int, mode: str):
    while True:
        batches_to_load = get_batch_numbers_to_load(mode, in_memory_batches)
        total_rows_in_audio_set_batches = get_total_rows_in_batches(batches_to_load, mode)
        print(
            f'\n\nLoading a new set of batches into memory!' +
            f'\n\tMode: {mode}' + 
            f'\n\tBatches being loaded: {batches_to_load}' + 
            f'\n\tTotal rows: {total_rows_in_audio_set_batches} rows\n')
        
        features = get_all_features(batches_to_load, mode)
        labels = get_all_labels(batches_to_load, mode)
        
        gc.collect()
        
        # Load a new set of random batches when we've processed the count of this one
        in_memory_row_index = 0
        while in_memory_row_index < total_rows_in_audio_set_batches * 1.5:
            
            # Create empty arrays to contain batch of features and labels
            batch_features = np.zeros((batch_size, features.shape[1], features.shape[2], 1))
            batch_labels = np.zeros((batch_size, labels.shape[1]))
            
            for batch_index in range(batch_size):
                # Choose random index in features
                index = random.randint(0, len(features)-1)
                
                # print(f'loop {i} of {batch_size}, index: {index}')
                
                batch_features[batch_index] = features[index] if 0 <= index < len(features) else None
                batch_labels[batch_index] = labels[index] if 0 <= index < len(labels) else None
                
                in_memory_row_index += 1
            
            yield batch_features, batch_labels

def get_generator(batch_size: int, mode: str):
    shuffled_batch_numbers = get_batch_numbers(mode)
        
    while True:
        random.shuffle(shuffled_batch_numbers)
        
        for batch_number in shuffled_batch_numbers:
            features = get_features_from_list(shuffled_batch_numbers, batch_number, mode)
            labels = get_labels_from_list(shuffled_batch_numbers, batch_number, mode)
            
            # Create empty arrays to contain batch of features and labels
            batch_features = np.zeros((batch_size, features.shape[1], features.shape[2], 1))
            batch_labels = np.zeros((batch_size, labels.shape[1]))
            
            for i in range(batch_size):
                # Choose random index in features
                index = random.randint(0, len(features)-1)
                batch_features[i] = features[index] if 0 <= index < len(features) else None
                batch_labels[i] = labels[index] if 0 <= index < len(labels) else None
            
            yield batch_features, batch_labels
            
def get_batch_numbers_to_load(mode: str, audio_set_batches_per_epoch: int) -> list[int]:
    all_batch_numbers = get_batch_numbers(mode)
    random.shuffle(all_batch_numbers)
    return all_batch_numbers[:audio_set_batches_per_epoch]

def train(model: Sequential) -> History:
    batch_size = 2048
    number_of_audio_set_batches = get_number_of_audio_set_batches()
    number_of_epochs = 25
    in_memory_batches = 12
    total_training_rows = get_total_rows('train')
    total_test_rows = get_total_rows('test')
    
    assert in_memory_batches > 0 and in_memory_batches <= number_of_audio_set_batches
    
    print(f'Training model.\n\t' +
          f'Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}\n\t' +
          f'Batches of dataset per epoch: {in_memory_batches}')
    
    history: History = model.fit(
        get_generator_large_dataset(in_memory_batches, batch_size, 'train'),
        steps_per_epoch = total_training_rows // batch_size,
        validation_data = get_generator_large_dataset(in_memory_batches, batch_size, 'test'),
	    validation_steps = total_test_rows // batch_size,
        epochs = number_of_epochs,
        batch_size=batch_size,
        callbacks=[
            EarlyStopping(patience=5, monitor = 'val_accuracy'), # to prevent overfitting
            BackupAndRestore(get_model_backup_dir(), 'epoch', True), # to restore interrupted training sessions
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
