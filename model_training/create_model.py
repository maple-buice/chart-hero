from tensorflow import keras
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.api.optimizers import Adam

def create_model(shape_1, shape_2) -> Sequential:
    # Initialize a Keras CNN network
    model = Sequential()  

    model.add(Conv2D(32, (3,3), 1, input_shape = (shape_1, shape_2, 1),  activation = 'relu', padding='same'))
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

    # Since we're doing a multi-label classification task, activation function set as sigmoid for the output layer

    model.add(Dense(8, activation = 'sigmoid'))

    model.compile(optimizer = Adam(learning_rate=0.001), 
                loss = 'binary_crossentropy', 
                metrics = ['accuracy'])
    
    return model