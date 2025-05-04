import os
import itertools
import math

import numpy as np
import pandas as pd

import mido
from mido import MidiFile

import librosa
import librosa.display

from tqdm.notebook import tqdm

from model_training.augment_audio import apply_augmentations

import soundfile as sf
from joblib import Parallel, delayed
import math
from functools import partial

def get_number_of_audio_set_batches() -> int:
    return 50

class data_preparation():

    tqdm.pandas()

    """
    This is a class for creating training dataset for model training.
    
    :attr directory_path: the directory path specified when initiaed the class
    :attr dataset_type: the type of dataset used 
    :attr dataset: the resampled dataset in pandas dataframe format
    :attr midi_wav_map: the mapping of each midi file and wav file pair
    :attr notes_collection: the training dataset that ready to use in downstrem tasks
    
    :param directory_path (string): The path to the root directory of the dataset. This class assume the use of GMD / E-GMD dataset
    :param dataset (string): either "gmd" or "egmd" is acceptable for now
    :param sample_ratio (float): the fraction of dataset want to be used in creating the training/test/eval dataset
    :param diff_threshold (float): filter out the midi/audio pair that > the duration difference threshold value 

    :raise NameError: the use of other dataset is not supported
    """
    
    def __init__(self, directory_path, dataset, sample_ratio=1, diff_threshold=1, n_jobs=-1):
        if dataset in ['gmd', 'egmd']:
            self.directory_path = directory_path
            self.dataset_type = dataset
            self.batch_tracking = 0
            self.n_jobs = n_jobs # Store number of jobs for parallel processing
            
            csv_path = [f for f in os.listdir(directory_path) if '.csv' in f][0]
            
            self.dataset = pd.read_csv(
                os.path.join(directory_path, csv_path)).dropna().sample(frac=sample_ratio).reset_index()
            
            df = self.dataset[['index', 'midi_filename', 'audio_filename', 'duration']].copy()
            df.columns = ['track_id', 'midi_filename', 'audio_filename', 'duration']
            
            print(f'Filtering out the midi/audio pair that has a duration difference > {diff_threshold} second using soundfile')

            # --- Optimization: Use soundfile and parallel processing for duration check ---
            def get_wav_duration(filepath):
                try:
                    info = sf.info(os.path.join(self.directory_path, filepath))
                    return info.duration
                except Exception as e:
                    print(f"Error getting duration for {filepath}: {e}")
                    return None # Return None or np.nan for problematic files

            # Use joblib for parallel execution
            durations = Parallel(n_jobs=self.n_jobs)(delayed(get_wav_duration)(f) for f in tqdm(df['audio_filename'], desc="Calculating audio durations"))
            df['wav_length'] = durations
            df.dropna(subset=['wav_length'], inplace=True) # Drop rows where duration couldn't be read
            # --- End Optimization ---

            df['diff'] = np.abs(df['duration'] - df['wav_length'])
            df = df[df['diff'].le(diff_threshold)]
            
            self.midi_wav_map = df.copy()
            self.notes_collection = pd.DataFrame()
            
            # the midi note mapping is copied from the Google project page. note 39,54,56 are new in 
            # EGMD dataset and Google never assigned it to a code. From initial listening test, these
            # are electronic pad / cowbell sound, temporaily assigned it to CB, CowBell for now
            # 
            # Updated for Clone Hero use case. See README for methodology.

            # self.midi_note_map={
            #     22: '67', # Hi-hat Closed (Edge) -> HiHatCymbal
            #     26: '67', # Hi-hat Open (Edge) -> HiHatCymbal
            #     35: '0', # Acoustic Bass Drum -> Kick
            #     36: '0', # Kick / Bass Drum 1 -> Kick
            #     37: '1', # Snare X-Stick / Side Stick -> Snare
            #     38: '1', # Snare (Head) / Acoustic Snare -> Snare
            #     39: '67', # Hand Clap	/ Cowbell -> HiHatCymbal
            #     40: '1', # Snare (Rim) / Electric Snare -> Snare
            #     41: '4', # Low Floor Tom	-> LowTom
            #     42: '67', # Hi-hat Closed (Bow) / Closed Hi-Hat -> HiHatCymbal
            #     43: '4', # Tom 3 (Head) / High Floor Tom -> LowTom
            #     44: '67', # Hi-hat Pedal / Pedal Hi-Hat -> HiHatCymbal
            #     45: '3', # Tom 2 / Low Tom -> MiddleTom
            #     46: '67', # Hi-hat Open (Bow) / Open Hi-Hat -> HiHatCymbal
            #     47: '3', # Tom 2 (Rim) / Low-Mid Tom -> MiddleTom
            #     48: '2', # Tom 1 / Hi-Mid Tom -> HighTom
            #     49: '66', # Crash 1 (Bow) / Crash Cymbal 1 -> CrashCymbal
            #     50: '2', # Tom 1 (Rim) / High Tom -> HighTom
            #     51: '68', # Ride (Bow) / Ride Cymbal 1 -> RideCymbal
            #     52: '66', # Crash 2 (Edge) / Chinese Cymbal -> CrashCymbal
            #     53: '68', # Ride (Bell) / Ride Bell -> RideCymbal
            #     54: '67', # Tambourine / Cowbell -> HiHatCymbal
            #     55: '66', # Crash 1 (Edge) / Splash Cymbal -> CrashCymbal
            #     56: '67', # Cowbell -> HiHatCymbal
            #     57: '66', # Crash 2 (Bow) / Crash Cymbal 2 -> CrashCymbal
            #     58: '4', # Tom 3 (Rim) / Vibraslap -> LowTom
            #     59: '68', # Ride (Edge) / Ride Cymbal 2 -> RideCymbal
            #     60: '2', # Hi Bongo -> HighTom
            #     61: '3', # Low Bongo -> MiddleTom
            #     62: '2', # Mute Hi Conga -> HighTom
            #     63: '3', # Open Hi Conga -> MiddleTom
            #     64: '4', # Low Conga -> LowTom
            #     65: '2', # High Timbale -> HighTom
            #     66: '3', # Low Timbale -> MiddleTom
            #     67: '2', # High Agogo -> HighTom
            #     68: '3', # Low Agogo -> MiddleTom
            #     69: '67', # Cabasa -> HiHatCymbal
            #     70: '67', # Maracas -> HiHatCymbal
            #     71: '68', # Short Whistle -> RideCymbal
            #     72: '66', # Long Whistle -> CrashCymbal
            #     73: '68', # Short Guiro -> RideCymbal
            #     74: '66', # Long Guiro -> CrashCymbal
            #     75: '67', # Claves -> HiHatCymbal
            #     76: '2', # Hi Wood Block -> HighTom
            #     77: '3', # Low Wood Block -> MiddleTom
            #     78: '2', # Mute Cuica -> HighTom
            #     79: '3', # Open Cuica -> MiddleTom
            #     80: '68', # Mute Triangle -> RideCymbal
            #     81: '66', # Open Triangle -> CrashCymbal
            # }

            id_len = self.midi_wav_map[['track_id','wav_length']]
            id_len.set_index('track_id', inplace=True)
            self.id_len_dict = id_len.to_dict()['wav_length']
        else:
            raise NameError('dataset not supported')

    # Keep get_length in case it's used elsewhere, but __init__ now uses soundfile
    def get_length(self,x):
        # This is now less efficient than sf.info(path).duration
        try:
            wav, sr = librosa.load(os.path.join(self.directory_path, x), sr=None, mono=True)
            return librosa.get_duration(y=wav, sr=sr)
        except Exception as e:
            print(f"Librosa error loading {x}: {e}")
            return None

    def notes_extraction(self, midi_file):
        """
        A function to extract notes from the miditrack. A miditrack is composed by a series of MidiMessage.
        Each MidiMessage contains information like channel, note, velocity, time etc.
        This function extract each note presented in the midi track and the associated start and stop playing time of each note
        
        :param midi_file (str):     The midi file path
        :return (list):             a list of 3-element lists, each 3-element list consist of [note label, associated start time in the track, associated end time in the track]  
        """
        
        # the time value stored in the MidiMessage is DELTATIME in ticks unit instead of exact time in second unit.
        # The delta time refer to the time difference between the MidiMessage and the previous MidiMessage
        
        # The midi note key map can be found in here:
        # https://rolandus.zendesk.com/hc/en-us/articles/360005173411-TD-17-Default-Factory-MIDI-Note-Map
        
        self.time_log = 0
        notes_collection = []
        temp_dict = {}
        
        self.midi_track = MidiFile(os.path.join(self.directory_path, midi_file))
        
        for msg in self.midi_track.tracks[-1]:
            try:
                self.time_log = self.time_log + msg.time
            except:
                continue

            if msg.type == 'note_on' and msg.velocity > 0:
                start = self.time_log
                if msg.note in temp_dict.keys():
                    try:
                        temp_dict[msg.note].append(start)
                    except:
                        start_list = [start]
                        start_list.append(temp_dict[msg.note])
                        temp_dict[msg.note] = start_list
                else:
                    temp_dict[msg.note] = start

            elif (msg.type == 'note_on' and msg.velocity == 0) or msg.type == 'note_off':
                end = self.time_log
                if type(temp_dict[msg.note]) == list:
                    notes_collection.append([
                        msg.note,
                        math.floor(temp_dict[msg.note][0] * 100) / 100,
                        math.ceil(end * 100) / 100])
                    del temp_dict[msg.note][0]
                    if len(temp_dict[msg.note]) == 0:
                        del temp_dict[msg.note]
                else:
                    notes_collection.append([
                        msg.note,
                        math.floor(temp_dict[msg.note] * 100) / 100,
                        math.ceil(end*100)/100])
                    del temp_dict[msg.note]

            else:
                continue
        return [[n[0], n[1], n[2]] for n in notes_collection]
    
    def time_meta_extraction(self):
        """
        A helper function to extract ticks_per_beat and tempo information from the meta messages in the midi file. 
        These information are required to convert midi time into seconds.
        """
        ticks_per_beat = self.midi_track.ticks_per_beat
        
        for msg in self.midi_track.tracks[0]:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
            else:
                pass
                    
        return (ticks_per_beat, tempo)
    
    def merge_note_label(self, track_id, notes_collection):
        """
        Merge the notes if they share the same start time, which means these notes were start playing at the same time.
        
        :param track_id (int):          the unique id of the midi_track
        :param notes_collection (list): the list of 3-element lists, each 3-element list consist of
                                        [note label, associated start time in the track, associated end time in the track]

        :return: Pandas DataFrame
        
        """

        merged_note_collection=[]

        key_func = lambda x: x[1]

        for key, group in itertools.groupby(notes_collection, key_func):
            group_ = list(group)
            if len(group_) > 1:
                merged_note = [x[0] for x in group_]
                start = min([x[1] for x in group_])
                end = max([x[2] for x in group_])
                merged_note_collection.append([merged_note, start, end])
            else:
                merged_note_collection.append(group_[0])

        output_df = pd.DataFrame(merged_note_collection)
        output_df.columns = ['label', 'start', 'end']
        output_df['track_id'] = track_id
        return output_df

    def ticks_to_second(self, notes_collection):
        """
        A helper function to convert midi time value to seconds. The time value stored in the MidiMessages is in "ticks" unit, this need to be converted
        to "second" unit for audio slicing tasks
        
        :param notes_collection (list):     the list of 3-element lists, each 3-element list consist of
                                            [note label, associated start time in the track, associated end time in the track]
    
        :return: a list of 3-element lists with start and end time rounded to 2 decimal places in seconds
        """
        if type(self.time_log) == float:
            return [[
                    note[0],
                    round(note[1], 2),
                    round(note[2], 2)]
                for note in notes_collection]
        else:
            ticks_per_beat, tempo = self.time_meta_extraction()

            return [[
                    note[0],
                    round(mido.tick2second(note[1], ticks_per_beat,tempo), 2),
                    round(mido.tick2second(note[2], ticks_per_beat,tempo), 2)]
                for note in notes_collection]
    
    # --- Optimization: Helper function for parallel processing of one file pair ---
    def _process_file_pair(self, row, pad_before, pad_after, fix_length):
        if row['midi_filename'] == 'drummer1/session1/78_jazz-fast_290_beat_4-4.mid':
            return None # Skip problematic file

        try:
            # MIDI Processing
            notes_collection = self.notes_extraction(row['midi_filename'])
            converted_notes_collection = self.ticks_to_second(notes_collection)
            track_notes = self.merge_note_label(row['track_id'], converted_notes_collection)

            if track_notes.empty:
                return None

            audio_file_path = os.path.join(self.directory_path, row['audio_filename'])
            sr = sf.info(audio_file_path).samplerate # Get sample rate efficiently
            audio_duration_samples = sf.info(audio_file_path).frames

            # --- Optimization: Segmented Audio Loading & Slicing ---
            def optimized_audio_slicing(note_row):
                start_time = note_row['start']
                end_time = note_row['end']
                max_len_seconds = audio_duration_samples / sr

                # Calculate start and end samples, applying padding
                slice_start_time = max(0.0, start_time - pad_before)
                slice_start_sample = librosa.time_to_samples(slice_start_time, sr=sr)

                if fix_length is not None:
                    slice_duration = fix_length
                    slice_end_time = slice_start_time + slice_duration
                    slice_end_sample = slice_start_sample + librosa.time_to_samples(slice_duration, sr=sr)
                else:
                    # This path is less efficient and avoided by the calling script
                    slice_end_time = min(max_len_seconds, end_time + pad_after)
                    slice_duration = slice_end_time - slice_start_time
                    slice_end_sample = min(audio_duration_samples, librosa.time_to_samples(end_time, sr=sr) + librosa.time_to_samples(pad_after, sr=sr))

                if slice_start_time >= max_len_seconds or slice_duration <= 0:
                    # If the start time is beyond the audio or duration is zero/negative, return silence
                    if fix_length is not None:
                        return np.zeros(librosa.time_to_samples(fix_length, sr=sr), dtype=np.float32)
                    else:
                        return np.array([], dtype=np.float32) # Or handle as appropriate

                # Load only the required segment
                try:
                    # Ensure we don't read past the end of the file
                    read_duration = min(slice_duration, max_len_seconds - slice_start_time)
                    if read_duration <= 0:
                         if fix_length is not None:
                            return np.zeros(librosa.time_to_samples(fix_length, sr=sr), dtype=np.float32)
                         else:
                            return np.array([], dtype=np.float32)

                    sliced_wav, _ = librosa.load(
                        audio_file_path, sr=sr, mono=True,
                        offset=slice_start_time,
                        duration=read_duration
                    )
                except Exception as load_err:
                     print(f"Error loading segment for {row['audio_filename']} at offset {slice_start_time}, duration {read_duration}: {load_err}")
                     if fix_length is not None:
                        return np.zeros(librosa.time_to_samples(fix_length, sr=sr), dtype=np.float32)
                     else:
                        return np.array([], dtype=np.float32)

                # Pad if necessary (especially for fix_length)
                if fix_length is not None:
                    target_samples = librosa.time_to_samples(fix_length, sr=sr)
                    if len(sliced_wav) < target_samples:
                        padding = target_samples - len(sliced_wav)
                        sliced_wav = np.pad(sliced_wav, (0, padding), mode='constant', constant_values=0)
                    elif len(sliced_wav) > target_samples:
                        sliced_wav = sliced_wav[:target_samples]

                return sliced_wav
            # --- End Segmented Loading ---

            track_notes['audio_wav'] = track_notes.apply(optimized_audio_slicing, axis=1)
            track_notes['sampling_rate'] = sr
            # Filter out rows where slicing might have failed or resulted in empty arrays
            track_notes = track_notes[track_notes['audio_wav'].apply(lambda x: isinstance(x, np.ndarray) and x.size > 0)]

            return track_notes

        except Exception as e:
            print(f"Error processing pair {row['midi_filename']} / {row['audio_filename']}: {e}")
            import traceback
            traceback.print_exc()
            return None
    # --- End Helper Function ---

    def create_audio_set(self, pad_before=0.02, pad_after=0.02, fix_length=None, batching=False, dir_path='', num_batches=50):
        """
        main function to create training/test/eval dataset from dataset
        
        :param  pad_before (float):     padding (in seconds) add to the begining of the sliced audio. default 0.02 seconds
        :param  pad_after (float):      padding (in seconds) add to the end of the sliced audio. default 0.02 seconds
        :param  fix_length (float):     in seconds, setting this length will force the sound clip to have exact same length in seconds. suggest value is 0.1~0.2
        :param  batching (bool):        apply batching to avoid memory issues. Suggest to turn this on if processing the full dataset.
        :param  dir_path (str):         The path to the directory to store .pkl files
        :param  num_batches (int):      The target number of batches to divide the dataset into.
        :return None
        """
        if batching and not dir_path:
            raise ValueError('Please specify directory path for saving pickle files when batching=True')

        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.batch_tracking = 0
        # tqdm.pandas() # Not needed directly here anymore

        # --- Removed old audio_slicing, resampling, check_length functions as logic is moved/optimized ---

        def save_batch(df_list, batch_idx):
            if not df_list:
                print(f"Batch {batch_idx} is empty, skipping save.")
                return
            
            notes_collection_batch = pd.concat(df_list, ignore_index=True)
            
            # Filter out problematic tracks where start time might exceed audio length (less likely now but good check)
            problematic_tracks = []
            for _, r in notes_collection_batch.iterrows():
                if r.track_id not in self.id_len_dict:
                     print(f"Warning: track_id {r.track_id} not found in id_len_dict. Skipping duration check for this row.")
                     continue
                if r.start > self.id_len_dict[r.track_id]:
                    problematic_tracks.append(r.track_id)
            
            notes_collection_batch = notes_collection_batch[~notes_collection_batch.track_id.isin(problematic_tracks)]

            if notes_collection_batch.empty:
                print(f"Batch {batch_idx} became empty after filtering problematic tracks, skipping save.")
                return

            # Train/Val/Test Split
            train_frac, val_frac = 0.6, 0.2
            train_df, val_df, test_df = np.split(
                notes_collection_batch.sample(frac=1, random_state=42),
                [int(train_frac * len(notes_collection_batch)),
                 int((train_frac + val_frac) * len(notes_collection_batch))])

            # Save
            train_df.to_pickle(os.path.join(dir_path, f"{batch_idx}_train.pkl"))
            val_df.to_pickle(os.path.join(dir_path, f"{batch_idx}_val.pkl"))
            test_df.to_pickle(os.path.join(dir_path, f"{batch_idx}_test.pkl"))

            print(f'Saved batch {batch_idx} data at {dir_path}')
            # Explicitly delete to free memory
            del notes_collection_batch, train_df, val_df, test_df

        print('Generating Dataset using Parallel Processing')

        # --- Optimization: Parallel Processing Loop ---
        process_func = partial(self._process_file_pair, pad_before=pad_before, pad_after=pad_after, fix_length=fix_length)
        results = Parallel(n_jobs=self.n_jobs, backend='loky')( # 'loky' is often more robust
            delayed(process_func)(row)
            for _, row in tqdm(self.midi_wav_map.iterrows(), total=self.midi_wav_map.shape[0], desc="Processing file pairs")
        )
        # --- End Parallel Processing Loop ---

        # Filter out None results (from errors or skipped files)
        valid_results = [df for df in results if df is not None and not df.empty]
        del results # Free memory

        if not valid_results:
             print("No valid dataframes were generated. Check for errors during processing.")
             return

        if batching:
            print(f"Processing {len(valid_results)} results into {num_batches} batches.")
            # Calculate approximate number of dataframes per batch
            num_results = len(valid_results)
            results_per_batch = math.ceil(num_results / num_batches)

            current_batch_list = []
            for i, df in enumerate(tqdm(valid_results, desc="Saving batches")):
                current_batch_list.append(df)
                # Save when batch is full or it's the last result
                if len(current_batch_list) >= results_per_batch or i == num_results - 1:
                    save_batch(current_batch_list, self.batch_tracking)
                    self.batch_tracking += 1
                    current_batch_list = [] # Reset for next batch
                    import gc
                    gc.collect() # Force garbage collection

        else: # No batching (likely for small datasets)
            print("Concatenating all results (no batching)")
            self.notes_collection = pd.concat(valid_results, ignore_index=True)
             # Filter out problematic tracks (redundant check, but safe)
            problematic_tracks=[]
            for r in tqdm(self.notes_collection.iterrows(), total=self.notes_collection.shape[0], desc="Final check"):
                if r[1].start>self.id_len_dict[r[1].track_id]:
                    problematic_tracks.append(r[1].track_id)
            self.notes_collection = self.notes_collection[~self.notes_collection.track_id.isin(problematic_tracks)]
            # Save the single file
            self.notes_collection.to_pickle(os.path.join(dir_path, f"dataset_full.pkl"))

        print('Done!')
        
    def augment_audio(self, audio_col='audio_wav', aug_col_names=None, aug_param_dict={}, train_only=False):
        """
        Apply audio augmentations to the training or full portion of a prepared audio dataset. The original dataset is modified to contain columns containing the augmented audio.
        
        Parameters:
            audio_col: String specifying the name of source audio column.
            aug_col_names: Names to use for augmented columns. Defaults to using the augmentation functions
                as column names.
            aug_param_dict: Dictionary of function names and associated parameters.
            train_only: Boolean for whether to augment the training set, or the data in its entirety.
        
        Example usage:
            data_container = data_preparation.data_preparation(gmd_path, dataset='gmd', sample_ratio=sample_ratio)
            data_container.augment_audio()
        """
        if not aug_param_dict:
            aug_param_dict = {
                'add_white_noise': {'snr': 20, 'random_state': 42},
                'augment_pitch': {'n_steps': 2, 'step_var': range(-1, 2, 1)},
                'add_pedalboard_effects': {}
            }
        if train_only:
            self.train = apply_augmentations(self.train, audio_col, aug_col_names, **aug_param_dict)
        else:
            self.notes_collection = apply_augmentations(self.notes_collection, audio_col, aug_col_names, **aug_param_dict)
