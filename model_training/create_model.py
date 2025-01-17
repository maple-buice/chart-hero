import numpy as np
from numpy import ndarray

from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.api.optimizers import Adam

from utils.file_utils import get_first_match, get_labeled_audio_set_dir

#region file util functions
def get_first_match_data_file(mode: str, file_suffix: str) -> str:
    return get_first_match(get_labeled_audio_set_dir(), list([mode, file_suffix]))
def get_batch_data_file(mode: str, file_suffix: str, batch_number: int) -> str:
    return get_first_match(get_labeled_audio_set_dir(), list([mode, file_suffix, f'{batch_number}_']))
#endregion

#region data array util functions
def get_data_array(file: str) -> ndarray:
    return np.load(file)
def get_first_match_data_array(mode: str, file_suffix: str) -> ndarray:
    return get_data_array(get_first_match_data_file(mode, file_suffix))
def get_batch_data_array(mode: str, file_suffix: str, batch_number: int) -> ndarray:
    return get_data_array(get_batch_data_file(mode, file_suffix, batch_number))
#endregion

#region features util functions
def get_first_match_training_features_file() -> str:
    return get_first_match_data_file('train', '_mel')
def get_first_match_training_features_array() -> ndarray:
    return get_data_array(get_first_match_training_features_file())
def get_features_shape() -> tuple[int, ...]:
    return get_first_match_training_features_array().shape
#endregion

#region labels util functions
#region first_match
def get_first_match_training_labels_file() -> str:
    return get_first_match_data_file('train', '_mel')
def get_first_match_training_labels_array() -> ndarray:
    return get_data_array(get_first_match_training_labels_file())
#endregion
#region batch
def get_batch_training_labels_file(batch_number: int) -> ndarray:
    return get_batch_data_file('train', '_label', batch_number)
def get_batch_training_labels_array(batch_number: int) -> ndarray:
    return get_data_array(get_batch_training_labels_file(batch_number))
#endregion

def get_labels_shape() -> tuple[int, ...]:
    return get_first_match_training_labels_array().shape
#endregion

def create_model() -> Sequential:
    # Initialize a Keras CNN network
    model = Sequential()
    
    x_shape = get_features_shape()
    print(x_shape)
    model.add(
        Conv2D(32, (3,3), 1,
        input_shape = (x_shape[1], x_shape[2], 1),
        activation = 'relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3), 1, activation = 'relu', padding='same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(64, (3,3), 1, activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Flatten())
    model.add(Dense(100, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dropout(0.3))

    # Since we're doing a multi-label classification task, 
    # activation function set as sigmoid for the output layer
    
    model.add(Dense(get_labels_shape()[1], activation = 'sigmoid'))

    model.compile(optimizer = Adam(learning_rate=0.001), 
                loss = 'binary_crossentropy', 
                metrics = ['accuracy'])
    
    return model