# %%
# Import packages and declare variables
import os
    
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report

from model_training.train_model import train_model
from model_training.label_data import label_data
from model_training.create_model import create_model

training_data_path = os.path.join(os.getcwd(), 'training_data')
e_gmd_path = os.path.join(training_data_path, 'e-gmd-v1.0.0')
audio_set_path = os.path.join(training_data_path, 'audio_set')
npy_data_path = os.path.join(training_data_path, 'npy_data')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %%
# create the audio set from the EGMD dataset
# create_audio_set(e_gmd_path, audio_set_path)

# %%
# label the data
label_data(audio_set_path, npy_data_path)

# %%
# load the train data and labels 
# for detailed info about how the data is generated, see the "Data_Transformation_Demo" notebook
# these two files are generated from the whole EGMD dataset and are too big to store in Github.

x_train = np.load(os.path.join(npy_data_path, 'train_0_mel.npy'))
y_train = np.load(os.path.join(npy_data_path, 'train_0_label.npy'))

x_test = np.load(os.path.join(npy_data_path, 'test_0_mel.npy'))
y_test = np.load(os.path.join(npy_data_path, 'test_0_label.npy'))

x_train.shape
y_train.shape

# %%
# Create model
model = create_model(128, 18)
model.summary()

# %%
# Train model

history = train_model(npy_data_path, model, 4096)
model.save("my_model.keras")

# %%
from matplotlib import pyplot as plt

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

# %%
# Network evaluation

drum_hits = ['0','1','2','3','4','66','67','68']

result = []
pred_raw = model.predict(x_test)
pred = np.round(pred_raw)

for i in range(pred_raw.shape[0]):
  prediction = pred[i]
  if sum(prediction) == 0:
    raw = pred_raw[i]
    new = np.zeros(6)
    ind = raw.argmax()
    new[ind] = 1
    result.append(new)
  else:
    result.append(prediction)

print(classification_report(y_test, np.array(result), target_names=drum_hits))

# %%
# Finish.


# %%
