import os
from array import array
from pathlib import Path
import random

import numpy as np
import tensorflow as tf

from keras.api.models import Sequential
from keras.api.callbacks import BackupAndRestore, EarlyStopping, History

from model_training.create_model import create_model

def train_model(npy_data_path: str, model: Sequential, batch_size: int) -> History:
    # add callback setting to model fitting process, to prevent overfitting

    batch_number_set = set()
    total_train_rows = 0
    total_test_rows = 0
    
    for file in os.listdir(npy_data_path):
        if not file.endswith('.npy'):
            continue
        batchNumber = Path(file).stem.split('_')[0]
        batch_number_set.add(int(batchNumber))
        
        loaded = np.load(os.path.join(npy_data_path, file))
        if 'train' in file:
            total_train_rows += len(loaded)
        elif 'test' in file:
            total_test_rows += len(loaded)
        
    batch_numbers = array('i', batch_number_set)
    
    def get_features(batch_number: int, mode: str):
        return np.load(os.path.join(npy_data_path, mode + f'_{batch_numbers[batch_number]}_mel.npy'))
    
    def get_labels(batch_number: int, mode: str):
        return np.load(os.path.join(npy_data_path, mode + f'_{batch_numbers[batch_number]}_label.npy'))
    
    def generator(batch_size: int, mode: str):
        # Create empty arrays to contain batch of features and labels#
        batch_features = np.zeros((batch_size, 128, 18, 1))
        batch_labels = np.zeros((batch_size, 8))
        
        while True:
            batch_index = random.randint(0, len(batch_numbers)-1)
            features = get_features(batch_index, mode)
            labels = get_labels(batch_index, mode)
            
            for i in range(batch_size):
                # choose random index in features
                index = random.randint(0, len(features)-1)
                batch_features[i] = features[index] if 0 <= index < len(features) else None
                batch_labels[i] = labels[index] if 0 <= index < len(labels) else None
            yield batch_features, batch_labels
        
    callback = [
        EarlyStopping(patience=5, monitor = 'val_accuracy'),
        BackupAndRestore('backup/', 'epoch', True, False),
    ]
    
    return model.fit(
        generator(batch_size, 'train'),
        steps_per_epoch = total_train_rows // batch_size,
        validation_data = generator(batch_size, 'test'),
	    validation_steps = total_test_rows // batch_size,
        epochs = 25,
        batch_size=batch_size,
        callbacks=callback
        )
    
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# training_data_path = os.path.join(os.getcwd(), 'training_data')
# npy_data_path = os.path.join(training_data_path, 'npy_data')
# model = create_model(128, 18)

# history = train_model(npy_data_path, model)
