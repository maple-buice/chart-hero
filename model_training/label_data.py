import os
from pathlib import Path
    
import pandas as pd
import numpy as np

import librosa

from utils.file_utils import get_audio_set_files, get_labeled_audio_set_dir

def get_drum_hits() -> list[int]:
    return [
        22, # Hi-hat Closed (Edge)
        26, # Hi-hat Open (Edge)
        35, # Acoustic Bass Drum
        36, # Kick / Bass Drum 1
        37, # Snare X-Stick / Side Stick
        38, # Snare (Head) / Acoustic Snare
        39, # Hand Clap	/ Cowbell
        40, # Snare (Rim) / Electric Snare
        41, # Low Floor Tom	-> LowTom
        42, # Hi-hat Closed (Bow) / Closed Hi-Hat
        43, # Tom 3 (Head) / High Floor Tom
        44, # Hi-hat Pedal / Pedal Hi-Hat
        45, # Tom 2 / Low Tom
        46, # Hi-hat Open (Bow) / Open Hi-Hat
        47, # Tom 2 (Rim) / Low-Mid Tom
        48, # Tom 1 / Hi-Mid Tom
        49, # Crash 1 (Bow) / Crash Cymbal 1
        50, # Tom 1 (Rim) / High Tom
        51, # Ride (Bow) / Ride Cymbal 1
        52, # Crash 2 (Edge) / Chinese Cymbal
        53, # Ride (Bell) / Ride Bell
        54, # Tambourine / Cowbell
        55, # Crash 1 (Edge) / Splash Cymbal
        56, # Cowbell
        57, # Crash 2 (Bow) / Crash Cymbal 2
        58, # Tom 3 (Rim) / Vibraslap
        59, # Ride (Edge) / Ride Cymbal 2
        60, # Hi Bongo
        61, # Low Bongo
        62, # Mute Hi Conga
        63, # Open Hi Conga
        64, # Low Conga
        65, # High Timbale
        66, # Low Timbale
        67, # High Agogo
        68, # Low Agogo
        69, # Cabasa
        70, # Maracas
        71, # Short Whistle
        72, # Long Whistle
        73, # Short Guiro
        74, # Long Guiro
        75, # Claves
        76, # Hi Wood Block
        77, # Low Wood Block
        78, # Mute Cuica
        79, # Open Cuica
        80, # Mute Triangle
        81, # Open Triangle
    ]
    
def get_drum_hits_as_strings() -> list[str]:
    drum_hits = []
    for drum_hit in get_drum_hits():
        drum_hits.append(str(drum_hit))
    return drum_hits

def label_data():
    npy_data_path = get_labeled_audio_set_dir()
        
    for file in get_audio_set_files():
        result_mel_file = os.path.join(npy_data_path, Path(file).stem + '_mel.npy')
        result_label_file = os.path.join(npy_data_path, Path(file).stem + '_label.npy')
        
        if os.path.exists(result_mel_file) and os.path.exists(result_label_file):
            print(f'\'{result_mel_file}\' and \'{result_label_file}\' exist. Cleaning up \'{file}\'')
            os.remove(file)
            continue
        
        print('labeling ' + file)
        
        df = pd.read_pickle(file)
        # df.head()
        
        # eight unique drum hit types
        # drum_hits = ['0','1','2','3','4','66','67','68']
        drum_hits = get_drum_hits()

        # label encoding            
        # some instances there are multiple positive labels

        for label in drum_hits:
            df[label] = df.apply(lambda row: label == row['label'], axis=1)
            df[label] = df[label].astype(int)

        y = df[drum_hits]
        # y.shape
        # y.head()
        
        mel_train = []

        # Mel-spec representation of drum_hit instances, each has the shape (128,18), has 1 channel
        # (normal image classification tasks the instances will have multiple channels)
        # the task we're doing is audio classification, which will only have 1 channel for all representation formats

        for i in range(df.shape[0]):
            mel_train.append(
                librosa.feature.melspectrogram(
                    y=df.audio_wav.iloc[i],
                    sr=df.sampling_rate.iloc[i],
                    n_mels=128,
                    fmax=8000))

        X = np.array(mel_train)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        
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
        
        # delete file to free up space and avoid reprocessing
        os.remove(file)