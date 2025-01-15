import os
from pathlib import Path
    
import pandas as pd
import numpy as np

import librosa

class label_data:
    def __init__(self, audio_set_path, npy_data_path):
        for file in os.listdir(audio_set_path):
            
            if not file.endswith('.pkl'):
                continue
            
            print(file)
            
            result_mel_file = os.path.join(npy_data_path, Path(file).stem + '_mel.npy')
            result_label_file = os.path.join(npy_data_path, Path(file).stem + '_label.npy')
            
            if os.path.exists(result_mel_file) and os.path.exists(result_label_file):
                print(result_mel_file + ' and ' + result_label_file + ' exist')
                continue
            
            print(file)
            
            df = pd.read_pickle(os.path.join(audio_set_path, file))
            # df.head()
            
            # eight unique drum hit types
            drum_hits = ['0','1','2','3','4','66','67','68']

            # label encoding            
            # some instances there are multiple positive labels

            for label in drum_hits:
                df[label] = df.apply(lambda row: label in row['label'],axis=1)
                df[label] = df[label].astype(int)

            y = df[drum_hits]
            # y.shape
            # y.head()
            
            mel_train = []

            # Mel-spec representation of drum_hit instances, each has the shape (128,18), has 1 channel
            # (normal image classification tasks the instances will have multiple channels)
            # the task we're doing is audio classification, which will only have 1 channel for all representation formats

            for i in range(df.shape[0]):
                mel_train.append(librosa.feature.melspectrogram(y=df.audio_wav.iloc[i], 
                                        sr=df.sampling_rate.iloc[i], n_mels=128, fmax=8000))

            X = np.array(mel_train)
            X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
            
            # this is in the format of 
            # a 4d array of (number of instances, y axis shape, x axis shape, 1 channel)
            # X.shape

            # save the numpy ndarray data
            # which can be directly fed to Keras network for training
            # can use np.save(filename, array) to save the ndarray data as npy files

            y = np.array(y)
            np.save(result_mel_file, X)
            np.save(result_label_file, y)

            # use np.load(filename) to load whenever we need it
