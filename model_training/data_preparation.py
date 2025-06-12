import sys
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
from joblib import Parallel, delayed, parallel_backend
import math
from functools import partial
import logging
import time

# Configure logging at the module level
logger = logging.getLogger(__name__)

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
        init_start_time = time.perf_counter()
        logger.info(f"Initializing data_preparation for dataset: {dataset} at {directory_path} with sample_ratio={sample_ratio}, diff_threshold={diff_threshold}")
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

            logger.info("Calculating audio durations using soundfile...")
            duration_calc_start_time = time.perf_counter()

            def get_wav_duration(filepath, base_dir): # Pass base_dir explicitly
                try:
                    info = sf.info(os.path.join(base_dir, filepath))
                    return info.duration
                except Exception as e:
                    logger.warning(f"Error getting duration for {filepath}: {e}") # Use warning
                    return None

            # Parallelize duration calculation
            if self.n_jobs == 1:
                logger.info("Using sequential processing for duration calculation (n_jobs=1)")
                durations = []
                for i, filename in enumerate(tqdm(df['audio_filename'], desc="Calculating audio durations sequentially")):
                    duration = get_wav_duration(filename, self.directory_path)
                    durations.append(duration)
                    if i % 200 == 0: # Less frequent GC for sequential
                        import gc
                        gc.collect()
            else:
                logger.info(f"Using parallel processing for duration calculation with n_jobs={self.n_jobs}")
                num_parallel_jobs = self.n_jobs if self.n_jobs > 0 else os.cpu_count()
                logger.info(f"Effective number of parallel jobs for duration calculation: {num_parallel_jobs}")
                
                # tasks = [delayed(get_wav_duration)(filename, self.directory_path) for filename in df['audio_filename']]
                # Use a wrapper function for pickling compatibility
                tasks = [delayed(self._get_wav_duration_wrapper)(filename) for filename in df['audio_filename']]

                with parallel_backend('loky', n_jobs=num_parallel_jobs):
                    durations = Parallel(verbose=10)(tasks)

            df['wav_length'] = durations
            duration_calc_end_time = time.perf_counter()
            logger.info(f"Audio duration calculation finished in {duration_calc_end_time - duration_calc_start_time:.2f} seconds.")

            df.dropna(subset=['wav_length'], inplace=True) # Drop rows where duration couldn't be read
            initial_pairs = len(df)
            logger.info(f"Found {initial_pairs} pairs before duration difference filtering.")

            df['diff'] = np.abs(df['duration'] - df['wav_length'])
            df = df[df['diff'].le(diff_threshold)]
            final_pairs = len(df)
            logger.info(f"Filtered pairs by duration difference (>{diff_threshold}s). Kept {final_pairs} pairs (removed {initial_pairs - final_pairs}).")

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
        init_end_time = time.perf_counter()
        logger.info(f"data_preparation initialization finished in {init_end_time - init_start_time:.2f} seconds.")

    # Keep get_length in case it's used elsewhere, but __init__ now uses soundfile
    def get_length(self,x):
        # This is now less efficient than sf.info(path).duration
        try:
            wav, sr = librosa.load(os.path.join(self.directory_path, x), sr=None, mono=True)
            return librosa.get_duration(y=wav, sr=sr)
        except Exception as e:
            print(f"Librosa error loading {x}: {e}")
            return None

    # Wrapper function for get_wav_duration to ensure it's picklable by joblib
    def _get_wav_duration_wrapper(self, filepath):
        # This method is part of the class, so self.directory_path is accessible
        try:
            info = sf.info(os.path.join(self.directory_path, filepath))
            return info.duration
        except Exception as e:
            logger.warning(f"Error getting duration for {filepath}: {e}")
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
    def _process_file_pair(self, row, pad_before, pad_after, fix_length, memory_limit_gb=None):
        # Ensure model_training module can be found by loky workers in Colab
        import sys
        import os
        colab_project_path = '/content/chart-hero'  # Path used in the notebook
        if colab_project_path not in sys.path:
            sys.path.insert(0, colab_project_path) # Insert at beginning to be safe

        pair_start_time = time.perf_counter()
        midi_file = row['midi_filename']
        audio_file = row['audio_filename']
        
        # Add memory monitoring if limit is provided
        if memory_limit_gb:
            import psutil
            process = psutil.Process()
            mem_before = process.memory_info().rss / (1024**3)
            if mem_before > memory_limit_gb * 0.8:  # 80% threshold
                logger.warning(f"Worker memory usage ({mem_before:.1f}GB) approaching limit ({memory_limit_gb:.1f}GB), skipping file {midi_file}")
                return None
        
        # logger.debug(f"Processing pair: {midi_file}") # Debug level for per-pair start

        if midi_file == 'drummer1/session1/78_jazz-fast_290_beat_4-4.mid':
             logger.warning(f"Skipping known problematic file: {midi_file}")
             return None # Skip problematic file

        try:
            # MIDI Processing
            notes_collection = self.notes_extraction(midi_file)
            if not notes_collection:
                 logger.warning(f"No notes extracted from MIDI: {midi_file}")
                 return None
            converted_notes_collection = self.ticks_to_second(notes_collection)
            track_notes = self.merge_note_label(row['track_id'], converted_notes_collection)

            if track_notes.empty:
                logger.warning(f"No notes after merging for MIDI: {midi_file}")
                return None

            audio_file_path = os.path.join(self.directory_path, audio_file)
            try:
                sf_info = sf.info(audio_file_path)
                sr = sf_info.samplerate
                audio_duration_samples = sf_info.frames
            except Exception as sf_err:
                 logger.error(f"Soundfile error reading info for {audio_file_path}: {sf_err}")
                 return None

            # --- Optimization: Segmented Audio Loading & Slicing ---
            def optimized_audio_slicing(note_row):
                # Memory monitoring before processing each note
                if memory_limit_gb:
                    import psutil
                    import gc
                    current_memory = psutil.Process().memory_info().rss / (1024**3)
                    if current_memory > memory_limit_gb * 0.9:  # 90% threshold
                        logger.warning(f"Memory usage ({current_memory:.1f}GB) near limit ({memory_limit_gb:.1f}GB), forcing garbage collection")
                        gc.collect()
                        # Re-check after cleanup
                        current_memory = psutil.Process().memory_info().rss / (1024**3)
                        if current_memory > memory_limit_gb * 0.95:  # 95% threshold
                            logger.error(f"Memory usage ({current_memory:.1f}GB) critically high, skipping note processing")
                            if fix_length is not None:
                                return np.zeros(librosa.time_to_samples(fix_length, sr=sr), dtype=np.float32)
                            else:
                                return np.array([], dtype=np.float32)
                
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

                # Load only the required segment with memory optimization
                try:
                    # Ensure we don't read past the end of the file
                    read_duration = min(slice_duration, max_len_seconds - slice_start_time)
                    if read_duration <= 0:
                         if fix_length is not None:
                            return np.zeros(librosa.time_to_samples(fix_length, sr=sr), dtype=np.float32)
                         else:
                            return np.array([], dtype=np.float32)

                    # Use lower-precision float32 and minimal memory allocation
                    sliced_wav, _ = librosa.load(
                        audio_file_path, sr=sr, mono=True,
                        offset=slice_start_time,
                        duration=read_duration,
                        dtype=np.float32  # Explicit float32 for memory efficiency
                    )
                except Exception as load_err:
                     # Corrected f-string syntax
                     logger.error(f"Error loading segment for {audio_file} at offset {slice_start_time}, duration {read_duration}: {load_err}")
                     # Ensure proper handling for both fix_length cases
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

            slicing_start_time = time.perf_counter()
            track_notes['audio_wav'] = track_notes.apply(optimized_audio_slicing, axis=1)
            slicing_end_time = time.perf_counter()
            # logger.debug(f"Audio slicing for {audio_file} took {slicing_end_time - slicing_start_time:.3f}s")

            track_notes['sampling_rate'] = sr
            # Filter out rows where slicing might have failed or resulted in empty arrays
            initial_notes = len(track_notes)
            track_notes = track_notes[track_notes['audio_wav'].apply(lambda x: isinstance(x, np.ndarray) and x.size > 0)]
            final_notes = len(track_notes)
            if initial_notes != final_notes:
                 logger.warning(f"Filtered {initial_notes - final_notes} notes with empty audio for {audio_file}")

            if track_notes.empty:
                 logger.warning(f"Track notes became empty after audio slicing for {midi_file}")
                 return None

            # Aggressive memory cleanup before returning
            import gc
            
            # Force deletion of large objects
            del sf_info
            if 'sliced_wav' in locals():
                del sliced_wav
            
            # Multiple garbage collection passes to ensure cleanup
            gc.collect()
            gc.collect()
            
            # Final memory check
            if memory_limit_gb:
                import psutil
                final_memory = psutil.Process().memory_info().rss / (1024**3)
                if final_memory > memory_limit_gb * 0.8:
                    logger.warning(f"Memory usage after processing ({final_memory:.1f}GB) still high")
            
            pair_end_time = time.perf_counter()
            logger.debug(f"Processed pair {midi_file} in {pair_end_time - pair_start_time:.3f} seconds, resulting in {final_notes} notes.")
            return track_notes

        except Exception as e:
            # Log error with traceback information
            logger.error(f"Error processing pair {midi_file} / {audio_file}: {e}", exc_info=True)
            return None
    # --- End Helper Function ---

    def create_audio_set(self, pad_before=0.02, pad_after=0.02, fix_length=None, batching=False, dir_path='', num_batches=50, memory_limit_gb=8):
        create_set_start_time = time.perf_counter()
        logger.info(f"Starting create_audio_set: pad_before={pad_before}, pad_after={pad_after}, fix_length={fix_length}, batching={batching}, num_batches={num_batches}, memory_limit={memory_limit_gb}GB")
        
        # Memory monitoring
        import psutil
        process = psutil.Process()
        
        def check_memory():
            mem_gb = process.memory_info().rss / (1024**3)
            if mem_gb > memory_limit_gb:
                logger.warning(f"Memory usage ({mem_gb:.1f}GB) exceeds limit ({memory_limit_gb}GB)")
            return mem_gb

        if batching and not dir_path:
            logger.error('Directory path must be specified for saving pickle files when batching=True')
            raise ValueError('Please specify directory path for saving pickle files when batching=True')

        if dir_path and not os.path.exists(dir_path):
            logger.info(f"Creating output directory: {dir_path}")
            os.makedirs(dir_path)

        self.batch_tracking = 0

        # Legacy save_batch function (now handled by _save_batch_sequential)
        def save_batch(df_list, batch_idx):
            # This is kept for backwards compatibility but should not be called
            self._save_batch_sequential(df_list, batch_idx, dir_path)
            df_list.clear()  # Clear input list

        # Create processing function with memory limit
        # For sequential processing, use full memory limit; for parallel, divide by cores
        if memory_limit_gb <= 48:  # Sequential mode (matches the processing decision threshold)
            per_process_memory_limit = memory_limit_gb
        else:  # Parallel mode - calculate based on cores that will actually be used
            n_cores_for_calc = min(os.cpu_count(), 6)  # Match the max cores from prepare_egmd_data.py
            per_process_memory_limit = memory_limit_gb / n_cores_for_calc
        
        process_func = partial(self._process_file_pair, pad_before=pad_before, pad_after=pad_after, fix_length=fix_length, memory_limit_gb=per_process_memory_limit)
        
        # Adaptive processing mode based on memory limits  
        # Force sequential for most cases since parallel audio processing causes memory explosion in librosa
        if memory_limit_gb <= 48:  # Conservative mode - parallel only for very high memory (48GB+)
            logger.info(f'Processing {len(self.midi_wav_map)} file pairs sequentially (low-memory mode)...')
            total_valid = self._process_sequential(process_func, batching, dir_path, num_batches, check_memory, memory_limit_gb)
        else:  # High-performance mode
            logger.info(f'Processing {len(self.midi_wav_map)} file pairs with optimized parallel processing (high-performance mode)...')
            total_valid = self._process_parallel_optimized(process_func, batching, dir_path, num_batches, check_memory, memory_limit_gb)

        if total_valid == 0:
             logger.warning("No valid dataframes were generated. Check logs for errors during processing.")
             return

        if batching:
            logger.info(f"Batching complete: saved {self.batch_tracking} batches")

        else: # No batching - combine individual files
            logger.info("Non-batching mode: combining individual files")
            individual_files = [f for f in os.listdir(dir_path) if f.startswith('individual_') and f.endswith('.pkl')]
            
            if individual_files:
                logger.info(f"Combining {len(individual_files)} individual files")
                combined_dfs = []
                for file in tqdm(individual_files, desc="Loading individual files"):
                    df = pd.read_pickle(os.path.join(dir_path, file))
                    combined_dfs.append(df)
                    # Clean up file after loading
                    os.remove(os.path.join(dir_path, file))
                
                self.notes_collection = pd.concat(combined_dfs, ignore_index=True)
                del combined_dfs
                import gc
                gc.collect()
                
                # Filter out problematic tracks
                problematic_tracks=[]
                for r in tqdm(self.notes_collection.iterrows(), total=self.notes_collection.shape[0], desc="Final check"):
                    if r[1].start>self.id_len_dict[r[1].track_id]:
                        problematic_tracks.append(r[1].track_id)
                self.notes_collection = self.notes_collection[~self.notes_collection.track_id.isin(problematic_tracks)]
                
                output_path = os.path.join(dir_path, "dataset_full.pkl")
                self.notes_collection.to_pickle(output_path)
                logger.info(f"Saved full dataset ({len(self.notes_collection)} notes) to {output_path}")
            else:
                logger.warning("No individual files found to combine")

        create_set_end_time = time.perf_counter()
        logger.info(f'create_audio_set finished in {create_set_end_time - create_set_start_time:.2f} seconds.')
    
    def _process_sequential(self, process_func, batching, dir_path, num_batches, check_memory, memory_limit_gb):
        """Sequential processing for memory-constrained environments"""
        total_valid = 0
        total_failed = 0
        current_batch_data = []
        batch_sample_count = 0
        target_samples_per_batch = 200  # Very small batches for memory efficiency
        
        for idx, (_, row) in enumerate(tqdm(self.midi_wav_map.iterrows(), total=len(self.midi_wav_map), desc="Processing files sequentially")):
            # Aggressive memory check before each file
            mem_before = check_memory()
            if mem_before > memory_limit_gb * 0.8:  # 80% threshold
                logger.warning(f"Memory usage ({mem_before:.1f}GB) high before processing file {idx}, forcing cleanup")
                import gc
                gc.collect()
                gc.collect()  # Double collection
                mem_after_cleanup = check_memory()
                logger.info(f"Memory after cleanup: {mem_after_cleanup:.1f}GB")
                
                # If still too high, skip processing
                if mem_after_cleanup > memory_limit_gb * 0.9:
                    logger.error(f"Memory usage still too high ({mem_after_cleanup:.1f}GB), skipping file")
                    total_failed += 1
                    continue
            
            result = process_func(row)
            
            if result is not None and not result.empty:
                total_valid += 1
                
                if batching:
                    current_batch_data.append(result)
                    batch_sample_count += len(result)
                    
                    # Save when batch reaches target size, but only if we haven't reached max batches
                    if batch_sample_count >= target_samples_per_batch and self.batch_tracking < num_batches:
                        # Find next available batch number (no overwriting)
                        while os.path.exists(os.path.join(dir_path, f"{self.batch_tracking}_train.pkl")):
                            self.batch_tracking += 1
                        
                        self._save_batch_sequential(current_batch_data, self.batch_tracking, dir_path)
                        logger.info(f"Saved batch {self.batch_tracking} ({batch_sample_count} samples)")
                        
                        # Reset for next batch
                        current_batch_data = []
                        batch_sample_count = 0
                        self.batch_tracking += 1
                        
                        import gc
                        gc.collect()
                    elif self.batch_tracking >= num_batches:
                        # Stop processing once we reach the desired number of batches
                        logger.info(f"Reached target number of batches ({num_batches}), stopping processing")
                        break
                else:
                    # Save individual files
                    individual_file = os.path.join(dir_path, f"individual_{idx}.pkl")
                    result.to_pickle(individual_file)
                
                del result
            else:
                total_failed += 1
            
            # More frequent garbage collection for memory management
            if idx % 10 == 0:  # Every 10 files instead of 50
                import gc
                gc.collect()
                gc.collect()  # Double collection
                current_mem = check_memory()
                logger.debug(f"Memory after processing {idx} files: {current_mem:.1f}GB")
        
        # Save remaining data only if we haven't reached the batch limit
        if batching and current_batch_data and self.batch_tracking < num_batches:
            while os.path.exists(os.path.join(dir_path, f"{self.batch_tracking}_train.pkl")):
                self.batch_tracking += 1
            self._save_batch_sequential(current_batch_data, self.batch_tracking, dir_path)
            logger.info(f"Saved final batch {self.batch_tracking} ({batch_sample_count} samples)")
        
        logger.info(f"Sequential processing complete: {total_valid} valid, {total_failed} failed")
        return total_valid
    
    def _process_parallel_optimized(self, process_func, batching, dir_path, num_batches, check_memory, memory_limit_gb):
        """Optimized parallel processing for high-resource environments"""
        # Calculate optimal cores and chunk size based on memory constraints
        available_cores = self.n_jobs if self.n_jobs > 0 else os.cpu_count()
        
        # Much more aggressive memory-based core limiting
        # Each audio processing worker can use 10-20GB easily, so be very conservative
        if memory_limit_gb <= 16:
            max_cores_by_memory = 1  # Force sequential for low memory
        elif memory_limit_gb <= 32:
            max_cores_by_memory = max(1, int(memory_limit_gb / 8))  # 8GB per core minimum
        else:
            max_cores_by_memory = max(1, int(memory_limit_gb / 12))  # 12GB per core for safety
        
        n_cores = min(available_cores, max_cores_by_memory, 16)  # Cap at 16 cores for memory safety (increased from 4)
        
        # Divide total memory limit across processes
        memory_per_process_gb = memory_limit_gb / n_cores
        logger.info(f"Total memory limit: {memory_limit_gb}GB, using {n_cores} cores = {memory_per_process_gb:.2f}GB per process")
        
        # Aggressively scale chunk size based on memory constraints
        base_chunk_size = max(5, min(50, len(self.midi_wav_map) // (n_cores * 4)))
        
        # Much more aggressive memory-based scaling
        if memory_per_process_gb <= 2:
            chunk_size = 1  # Process one file at a time
            logger.warning(f"Very low memory per process ({memory_per_process_gb:.2f}GB), using chunk_size=1")
        elif memory_per_process_gb <= 4:
            chunk_size = max(2, base_chunk_size // 8)  # Very small chunks
            logger.info(f"Low memory per process ({memory_per_process_gb:.2f}GB), using chunk_size={chunk_size}")
        elif memory_per_process_gb <= 8:
            chunk_size = max(5, base_chunk_size // 4)  # Small chunks
            logger.info(f"Medium memory per process ({memory_per_process_gb:.2f}GB), using chunk_size={chunk_size}")
        else:
            chunk_size = max(10, base_chunk_size // 2)  # Moderate chunks
            logger.info(f"Good memory per process ({memory_per_process_gb:.2f}GB), using chunk_size={chunk_size}")
        
        # Create per-process memory check function
        def check_memory_per_process():
            import psutil
            process = psutil.Process()
            mem_gb = process.memory_info().rss / (1024**3)
            if mem_gb > memory_per_process_gb:
                logger.warning(f"Process memory usage ({mem_gb:.1f}GB) exceeds per-process limit ({memory_per_process_gb:.2f}GB)")
            return mem_gb
        
        logger.info(f"Using {n_cores} cores with chunk size {chunk_size}")
        
        total_valid = 0
        total_failed = 0
        batch_data_accumulator = []
        batch_sample_count = 0
        
        # Much more conservative batch sizing based on memory
        if memory_per_process_gb <= 2:
            target_samples_per_batch = 100  # Very small batches
        elif memory_per_process_gb <= 4:
            target_samples_per_batch = 250  # Small batches
        elif memory_per_process_gb <= 8:
            target_samples_per_batch = 500  # Medium batches
        else:
            target_samples_per_batch = min(1000, max(500, int(memory_per_process_gb * 50)))  # Larger batches
        
        logger.info(f"Target samples per batch: {target_samples_per_batch} (based on {memory_per_process_gb:.2f}GB per process)")
        
        # Process in optimized chunks
        for chunk_start in tqdm(range(0, len(self.midi_wav_map), chunk_size), desc="Processing chunks"):
            chunk_end = min(chunk_start + chunk_size, len(self.midi_wav_map))
            chunk_df = self.midi_wav_map.iloc[chunk_start:chunk_end]
            
            mem_before = check_memory_per_process()
            
            # Parallel processing of chunk
            with parallel_backend('loky', n_jobs=n_cores):
                chunk_results = Parallel()(
                    delayed(process_func)(row)
                    for _, row in chunk_df.iterrows()
                )
            
            # Process results
            valid_chunk_results = [df for df in chunk_results if df is not None and not df.empty]
            chunk_valid = len(valid_chunk_results)
            chunk_failed = len(chunk_df) - chunk_valid
            total_valid += chunk_valid
            total_failed += chunk_failed
            
            if batching and valid_chunk_results:
                batch_data_accumulator.extend(valid_chunk_results)
                batch_sample_count += sum(len(df) for df in valid_chunk_results)
                
                # Save batches when they reach target size, but only if we haven't reached max batches
                while batch_sample_count >= target_samples_per_batch and self.batch_tracking < num_batches:
                    # Extract one batch worth of data
                    current_samples = 0
                    batch_data = []
                    remaining_data = []
                    
                    for df in batch_data_accumulator:
                        if current_samples + len(df) <= target_samples_per_batch:
                            batch_data.append(df)
                            current_samples += len(df)
                        else:
                            remaining_data.append(df)
                    
                    if not batch_data:
                        # If single dataframe is larger than target, save it anyway
                        batch_data = batch_data_accumulator[:1]
                        remaining_data = batch_data_accumulator[1:]
                        current_samples = len(batch_data[0]) if batch_data else 0
                    
                    # Find next available batch number
                    while os.path.exists(os.path.join(dir_path, f"{self.batch_tracking}_train.pkl")):
                        self.batch_tracking += 1
                    
                    self._save_batch_sequential(batch_data, self.batch_tracking, dir_path)
                    logger.info(f"Saved batch {self.batch_tracking} ({current_samples} samples)")
                    
                    # Update accumulators
                    batch_data_accumulator = remaining_data
                    batch_sample_count -= current_samples
                    self.batch_tracking += 1
                    
                    import gc
                    gc.collect()
            
            # Check if we've reached target number of batches
            if batching and self.batch_tracking >= num_batches:
                logger.info(f"Reached target number of batches ({num_batches}), stopping chunk processing")
                break
            
            # Memory cleanup
            del chunk_results, valid_chunk_results
            import gc
            gc.collect()
            
            mem_after = check_memory_per_process()
            logger.debug(f"Chunk {chunk_start//chunk_size + 1}: {chunk_valid} valid, {chunk_failed} failed, memory: {mem_before:.1f}GB -> {mem_after:.1f}GB")
        
        # Save remaining data only if we haven't reached the batch limit
        if batching and batch_data_accumulator and self.batch_tracking < num_batches:
            while os.path.exists(os.path.join(dir_path, f"{self.batch_tracking}_train.pkl")):
                self.batch_tracking += 1
            self._save_batch_sequential(batch_data_accumulator, self.batch_tracking, dir_path)
            logger.info(f"Saved final batch {self.batch_tracking} ({batch_sample_count} samples)")
        
        logger.info(f"Parallel processing complete: {total_valid} valid, {total_failed} failed")
        return total_valid
    
    def _save_batch_sequential(self, df_list, batch_idx, dir_path):
        """Save batch without modifying the input list"""
        if not df_list:
            return
        
        notes_collection_batch = pd.concat(df_list, ignore_index=True)
        
        # Train/Val/Test Split
        train_frac, val_frac = 0.6, 0.2
        shuffled_batch = notes_collection_batch.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n_total = len(shuffled_batch)
        n_train = int(train_frac * n_total)
        n_val = int(val_frac * n_total)
        
        train_df = shuffled_batch.iloc[:n_train].copy()
        val_df = shuffled_batch.iloc[n_train:n_train + n_val].copy()
        test_df = shuffled_batch.iloc[n_train + n_val:].copy()
        
        # Save files
        train_df.to_pickle(os.path.join(dir_path, f"{batch_idx}_train.pkl"))
        val_df.to_pickle(os.path.join(dir_path, f"{batch_idx}_val.pkl"))
        test_df.to_pickle(os.path.join(dir_path, f"{batch_idx}_test.pkl"))
        
        # Cleanup
        del notes_collection_batch, shuffled_batch, train_df, val_df, test_df
        import gc
        gc.collect()

    def augment_audio(self, audio_col='audio_wav', aug_col_names=None, aug_param_dict={}, train_only=False):
        aug_start_time = time.perf_counter()
        target_df_name = 'train' if train_only else 'notes_collection'
        logger.info(f"Starting audio augmentation on '{target_df_name}' (train_only={train_only}). Using params: {aug_param_dict or 'default'}")

        if not aug_param_dict:
            # --- Updated Default Augmentation Parameters --- 
            print("Using default augmentation parameters with randomization.")
            aug_param_dict = {
                'add_white_noise': {'snr_db_range': (10, 30)}, # Use randomized SNR
                'augment_pitch': {'n_steps_range': (-2, 2)}, # Use randomized pitch steps
                'add_pedalboard_effects': { # Use randomized reverb/lowpass
                    'room_size_range': (0.1, 0.8),
                    'cutoff_freq_hz_range': (1000, 6000)
                },
                # Example: Add lowpass filter separately sometimes
                # 'add_lowpass_filter': {'cutoff_freq_hz_range': (1500, 7000)}
            }
            # --- End Updated Defaults --- 
            
        # The rest of the function remains the same, calling the updated apply_augmentations
        if train_only:
             if hasattr(self, 'train') and not self.train.empty:
                 logger.info(f"Applying augmentations to training set ({len(self.train)} samples)...")
                 self.train = apply_augmentations(self.train, audio_col, aug_col_names, **aug_param_dict)
             else:
                 logger.warning("Skipping augmentation: 'train_only' is True, but self.train is not available or empty.")
        else:
             if not self.notes_collection.empty:
                 logger.info(f"Applying augmentations to entire dataset ({len(self.notes_collection)} samples)...")
                 self.notes_collection = apply_augmentations(self.notes_collection, audio_col, aug_col_names, **aug_param_dict)
             else:
                 logger.warning("Skipping augmentation: self.notes_collection is empty.")

        aug_end_time = time.perf_counter()
        logger.info(f"Audio augmentation finished in {aug_end_time - aug_start_time:.2f} seconds.")
