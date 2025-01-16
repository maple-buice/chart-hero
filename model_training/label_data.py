import os
from pathlib import Path
    
import pandas as pd
import numpy as np

import librosa

class label_data:
    def __init__(self, audio_set_path, npy_data_path):
        if not os.path.exists(npy_data_path):
            os.makedirs(npy_data_path)
            
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
            # drum_hits = ['0','1','2','3','4','66','67','68']
            drum_hits = [
                22, # Hi-hat Closed (Edge) -> HiHatCymbal
                26, # Hi-hat Open (Edge) -> HiHatCymbal
                35, # Acoustic Bass Drum -> Kick
                36, # Kick / Bass Drum 1 -> Kick
                37, # Snare X-Stick / Side Stick -> Snare
                38, # Snare (Head) / Acoustic Snare -> Snare
                39, # Hand Clap	/ Cowbell -> HiHatCymbal
                40, # Snare (Rim) / Electric Snare -> Snare
                41, # Low Floor Tom	-> LowTom
                42, # Hi-hat Closed (Bow) / Closed Hi-Hat -> HiHatCymbal
                43, # Tom 3 (Head) / High Floor Tom -> LowTom
                44, # Hi-hat Pedal / Pedal Hi-Hat -> HiHatCymbal
                45, # Tom 2 / Low Tom -> MiddleTom
                46, # Hi-hat Open (Bow) / Open Hi-Hat -> HiHatCymbal
                47, # Tom 2 (Rim) / Low-Mid Tom -> MiddleTom
                48, # Tom 1 / Hi-Mid Tom -> HighTom
                49, # Crash 1 (Bow) / Crash Cymbal 1 -> CrashCymbal
                50, # Tom 1 (Rim) / High Tom -> HighTom
                51, # Ride (Bow) / Ride Cymbal 1 -> RideCymbal
                52, # Crash 2 (Edge) / Chinese Cymbal -> CrashCymbal
                53, # Ride (Bell) / Ride Bell -> RideCymbal
                54, # Tambourine / Cowbell -> HiHatCymbal
                55, # Crash 1 (Edge) / Splash Cymbal -> CrashCymbal
                56, # Cowbell -> HiHatCymbal
                57, # Crash 2 (Bow) / Crash Cymbal 2 -> CrashCymbal
                58, # Tom 3 (Rim) / Vibraslap -> LowTom
                59, # Ride (Edge) / Ride Cymbal 2 -> RideCymbal
                60, # Hi Bongo -> HighTom
                61, # Low Bongo -> MiddleTom
                62, # Mute Hi Conga -> HighTom
                63, # Open Hi Conga -> MiddleTom
                64, # Low Conga -> LowTom
                65, # High Timbale -> HighTom
                66, # Low Timbale -> MiddleTom
                67, # High Agogo -> HighTom
                68, # Low Agogo -> MiddleTom
                69, # Cabasa -> HiHatCymbal
                70, # Maracas -> HiHatCymbal
                71, # Short Whistle -> RideCymbal
                72, # Long Whistle -> CrashCymbal
                73, # Short Guiro -> RideCymbal
                74, # Long Guiro -> CrashCymbal
                75, # Claves -> HiHatCymbal
                76, # Hi Wood Block -> HighTom
                77, # Low Wood Block -> MiddleTom
                78, # Mute Cuica -> HighTom
                79, # Open Cuica -> MiddleTom
                80, # Mute Triangle -> RideCymbal
                81, # Open Triangle -> CrashCymbal
            ]

            # label encoding            
            # some instances there are multiple positive labels

            for label in drum_hits:
                df[label] = df.apply(lambda row: label == row['label'],axis=1)
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
            
            # delete file to free up space and avoid reprocessing
            os.remove(os.path.join(audio_set_path, file))