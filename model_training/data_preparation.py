import sys
import os

# Explicitly add project root to sys.path for Colab and local
# This needs to run successfully in the main process AND in each joblib worker process
# when they import this module.

_DRIVE_PATH_CHART_HERO = '/content/drive/MyDrive/chart-hero' # Standard Colab path for this project

def _add_project_root_to_path():
    # Try to determine project root and add to sys.path
    # This function will be called when the module is imported.
    project_root_to_add = None
    try:
        if 'google.colab' in sys.modules:
            project_root_to_add = _DRIVE_PATH_CHART_HERO
        else: # Local execution
            # Assumes data_preparation.py is in chart-hero/model_training/
            # __file__ should be defined when the module is imported.
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root_to_add = os.path.abspath(os.path.join(current_file_dir, '..'))

        if project_root_to_add and project_root_to_add not in sys.path:
            sys.path.insert(0, project_root_to_add)
            # print(f"[data_preparation module] Added to sys.path: {project_root_to_add}", flush=True) # Debug
        # else:
            # print(f"[data_preparation module] Project root {project_root_to_add} already in sys.path or not determined.", flush=True) # Debug
            
    except Exception as e:
        # print(f"[data_preparation module] Error in _add_project_root_to_path: {e}", flush=True) # Debug
        pass # Avoid crashing import if path logic fails

_add_project_root_to_path()

# Remove the previous sys.path modification block if it exists below to avoid redundancy.
# The old block looked like:
# _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if _PROJECT_ROOT not in sys.path:
#    sys.path.insert(0, _PROJECT_ROOT)

# Now proceed with other imports
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
from functools import partial
import logging
import time

# Configure logging at the module level
logger = logging.getLogger(__name__)

# Aliased imports for top-level helper
import soundfile as sf_top
import os as os_top

# NEW Top-level worker function for create_audio_set parallel processing
def process_file_pair_worker(row_dict, directory_path_arg, pad_before_arg, pad_after_arg, fix_length_arg, memory_limit_gb_worker, project_root_path_for_worker):
    # This function is called by joblib workers.
    
    # --- Explicitly add project root to sys.path IN THE WORKER ---
    if project_root_path_for_worker and project_root_path_for_worker not in sys.path:
        sys.path.insert(0, project_root_path_for_worker)
        # Optional: log from worker to confirm path addition
        # import logging
        # worker_logger = logging.getLogger(f"worker_{os.getpid()}") # Basic config for worker log
        # worker_logger.info(f"[Worker] Added to sys.path: {project_root_path_for_worker}")
        # worker_logger.info(f"[Worker] sys.path is now: {sys.path}")

    # It needs to be self-contained or import modules that are findable.
    # The sys.path modification at the top of the file should help.
    
    midi_file = row_dict['midi_filename']
    audio_file = row_dict['audio_filename']
    track_id = row_dict['track_id'] 

    pair_start_time = time.perf_counter()

    if memory_limit_gb_worker:
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024**3)
        if mem_before > memory_limit_gb_worker * 0.8:
            logger.warning(f"[Worker] Memory usage ({mem_before:.1f}GB) approaching limit ({memory_limit_gb_worker:.1f}GB), skipping file {midi_file}")
            return None

    if midi_file == 'drummer1/session1/78_jazz-fast_290_beat_4-4.mid':
        logger.warning(f"[Worker] Skipping known problematic file: {midi_file}")
        return None

    try:
        # --- MINIMAL MIDI MOCKUP TO TEST PICKLING AND MODULE FINDING --- 
        if not midi_file or not audio_file:
            logger.warning(f"[Worker] Missing midi_file or audio_file for row: {row_dict}")
            return None
        
        time.sleep(0.01) # Simulate work
        # Ensure pandas and numpy are imported if not already at top level of worker context
        import pandas as pd 
        import numpy as np
        track_notes = pd.DataFrame({
            'label': [[60]], 'start': [0.0], 'end': [0.5], 'track_id': [track_id]
        })
        # --- END MINIMAL MIDI MOCKUP ---

        if track_notes.empty:
            logger.warning(f"[Worker] No notes after merging for MIDI: {midi_file}")
            return None

        audio_file_path = os.path.join(directory_path_arg, audio_file)
        try:
            # Ensure soundfile (sf) and librosa are available to the worker
            # sf is imported as sf_top at module level, but direct sf might be an issue
            # For now, assume sf is available via global imports or direct import here
            import soundfile as sf_worker # Explicit import for worker
            import librosa as librosa_worker # Explicit import for worker
            sf_info = sf_worker.info(audio_file_path)
            sr = sf_info.samplerate
            # audio_duration_samples = sf_info.frames # Not used in mockup
        except Exception as sf_err:
            logger.error(f"[Worker] Soundfile error reading info for {audio_file_path}: {sf_err}")
            return None
        
        # --- MINIMAL AUDIO MOCKUP ---
        def optimized_audio_slicing_mock(note_row_dict_ignored):
            time.sleep(0.01) # Simulate work
            if fix_length_arg is not None:
                return np.zeros(librosa_worker.time_to_samples(fix_length_arg, sr=sr), dtype=np.float32)
            return np.array([1,2,3], dtype=np.float32)
        
        track_notes['audio_wav'] = track_notes.apply(lambda r: optimized_audio_slicing_mock(r.to_dict()), axis=1)
        # --- END MINIMAL AUDIO MOCKUP ---

        track_notes['sampling_rate'] = sr
        initial_notes = len(track_notes)
        track_notes = track_notes[track_notes['audio_wav'].apply(lambda x: isinstance(x, np.ndarray) and x.size > 0)]
        final_notes = len(track_notes)
        if initial_notes != final_notes:
            logger.warning(f"[Worker] Filtered {initial_notes - final_notes} notes with empty audio for {audio_file}")

        if track_notes.empty:
            logger.warning(f"[Worker] Track notes became empty after audio slicing for {midi_file}")
            return None

        import gc
        del sf_info
        gc.collect()
        
        pair_end_time = time.perf_counter()
        logger.debug(f"[Worker] Processed pair {midi_file} in {pair_end_time - pair_start_time:.3f} seconds, resulting in {final_notes} notes.")
        return track_notes

    except Exception as e:
        logger.error(f"[Worker] Error processing pair {midi_file} / {audio_file}: {e}", exc_info=True)
        return None

def get_duration_joblib_helper(filepath, base_dir):
    try:
        info = sf_top.info(os_top.path.join(base_dir, filepath))
        return info.duration
    except Exception as e:
        # Using logger, but ensure logger is configured to be multiprocess-safe if issues persist
        # For now, direct print might be more visible from workers in Colab if logger isn't showing up
        # print(f"[Worker] Error in get_duration_joblib_helper for {filepath}: {e}")
        logger.warning(f"Error getting duration for {filepath} in get_duration_joblib_helper: {e}")
        return None

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

            # Parallelize duration calculation
            if self.n_jobs == 1:
                logger.info("Using sequential processing for duration calculation (n_jobs=1)")
                durations = []
                for i, filename in enumerate(tqdm(df['audio_filename'], desc="Calculating audio durations sequentially")):
                    duration = get_duration_joblib_helper(filename, self.directory_path) # Use top-level helper
                    durations.append(duration)
                    if i % 200 == 0: # Less frequent GC for sequential
                        import gc
                        gc.collect()
            else:
                logger.info(f"Using joblib.Parallel for duration calculation with n_jobs={self.n_jobs}")
                num_parallel_jobs = self.n_jobs if self.n_jobs > 0 else os_top.cpu_count()
                logger.info(f"Effective number of parallel jobs for duration calculation: {num_parallel_jobs}")
                
                tasks = [delayed(get_duration_joblib_helper)(filename, self.directory_path) for filename in df['audio_filename']]

                try:
                    with parallel_backend('loky', n_jobs=num_parallel_jobs):
                        durations = Parallel(verbose=10)(tasks)
                except Exception as joblib_error:
                    logger.error(f"Error during joblib.Parallel execution for durations: {joblib_error}", exc_info=True)
                    # Attempt to diagnose if the module is importable in the main process context after failure
                    try:
                        import model_training.data_preparation
                        logger.info("DIAGNOSTIC: Successfully imported model_training.data_preparation in main process after joblib error.")
                    except ImportError as ie:
                        logger.error(f"DIAGNOSTIC: Failed to import model_training.data_preparation in main process after joblib error: {ie}")
                    raise


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

    # Keep get_length in case it\'s used elsewhere, but __init__ now uses soundfile
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
    def _process_file_pair(self, row, pad_before, pad_after, fix_length, memory_limit_gb=None):

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
        
        import psutil # Moved import here as it's used in check_memory
        process = psutil.Process() # Moved process definition here

        def check_memory(): # check_memory is now defined and can be passed
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
        if memory_limit_gb <= 48:  # Sequential mode
            per_process_memory_limit = memory_limit_gb
        else:  # Parallel mode
            n_cores_for_calc = min(os.cpu_count(), 6) 
            per_process_memory_limit = memory_limit_gb / n_cores_for_calc
        
        # Adaptive processing mode based on memory limits  
        if memory_limit_gb <= 48: 
            logger.info(f'Processing {len(self.midi_wav_map)} file pairs sequentially (low-memory mode)...')
            sequential_process_func = partial(self._process_file_pair, pad_before=pad_before, pad_after=pad_after, fix_length=fix_length, memory_limit_gb=per_process_memory_limit)
            total_valid = self._process_sequential(sequential_process_func, batching, dir_path, num_batches, check_memory, memory_limit_gb)
        else:
            logger.info(f'Processing {len(self.midi_wav_map)} file pairs in parallel using new top-level worker (high-memory mode)...')
            total_valid = self._process_parallel_optimized_new_worker(
                self.midi_wav_map, 
                self.directory_path, 
                pad_before, pad_after, fix_length, 
                batching, dir_path, num_batches, 
                check_memory, # Pass the defined check_memory function
                memory_limit_gb, self.n_jobs,
                per_process_memory_limit # Pass the calculated per_process_memory_limit for the worker
            )

        create_set_end_time = time.perf_counter()
        logger.info(f"create_audio_set completed: {total_valid} valid file pairs processed in {create_set_end_time - create_set_start_time:.2f} seconds.")
        return total_valid

    def _process_sequential(self, process_func, batching, dir_path, num_batches, check_memory, memory_limit_gb):
        """Sequential processing for memory-constrained environments"""
        logger.warning("Warning: _process_sequential is deprecated. Please use create_audio_set directly.")
        
        total_processed = 0
        df_list = []
        batch_idx = 0

        for i, row in enumerate(tqdm(self.midi_wav_map.itertuples(index=False), desc="Processing file pairs")):
            result = process_func(row)
            if result is not None:
                df_list.append(result)
                total_processed += 1

            # Batch saving logic
            if batching and len(df_list) >= 1000:
                self._save_batch_sequential(df_list, batch_idx, dir_path)
                batch_idx += 1
                df_list.clear()  # Clear list after saving

            # Memory check
            if i % 100 == 0:
                check_memory() # Corrected: use passed check_memory function

        # Final batch save
        if df_list:
            self._save_batch_sequential(df_list, batch_idx, dir_path)

        logger.info(f"Sequential processing complete: {total_processed} file pairs processed.")
        return total_processed

    # --- New parallel processing method using the top-level worker ---
    def _process_parallel_optimized_new_worker(self, midi_wav_df, directory_path_arg, pad_before_arg, pad_after_arg, fix_length_arg, batching_arg, dir_path_arg, num_batches_arg, check_memory_func_arg, memory_limit_gb_arg, n_jobs_arg, worker_memory_limit_arg):
        logger.info(f"Using _process_parallel_optimized_new_worker with n_jobs={n_jobs_arg}")
        available_cores = n_jobs_arg if n_jobs_arg > 0 else os.cpu_count()
        
        if memory_limit_gb_arg <= 16:
            max_cores_by_memory = 1
        elif memory_limit_gb_arg <= 32:
            max_cores_by_memory = max(1, int(memory_limit_gb_arg / 8))
        else:
            max_cores_by_memory = max(1, int(memory_limit_gb_arg / 12))
        
        n_cores = min(available_cores, max_cores_by_memory, 16)
        memory_per_process_gb = memory_limit_gb_arg / n_cores
        logger.info(f"Total memory limit: {memory_limit_gb_arg}GB, using {n_cores} cores = {memory_per_process_gb:.2f}GB per process for new worker")

        base_chunk_size = max(5, min(50, len(midi_wav_df) // (n_cores * 4)))
        if memory_per_process_gb <= 2: chunk_size = 1
        elif memory_per_process_gb <= 4: chunk_size = max(2, base_chunk_size // 8)
        elif memory_per_process_gb <= 8: chunk_size = max(5, base_chunk_size // 4)
        else: chunk_size = max(10, base_chunk_size // 2)
        logger.info(f"Using {n_cores} cores with chunk size {chunk_size} for new worker")

        total_valid = 0
        total_failed = 0
        batch_data_accumulator = []
        batch_sample_count = 0
        
        if memory_per_process_gb <= 2: target_samples_per_batch = 100
        elif memory_per_process_gb <= 4: target_samples_per_batch = 250
        elif memory_per_process_gb <= 8: target_samples_per_batch = 500
        else: target_samples_per_batch = min(1000, max(500, int(memory_per_process_gb * 50)))
        logger.info(f"Target samples per batch: {target_samples_per_batch} for new worker")

        # Prepare tasks: iterate over rows and create dictionaries for the worker
        # This is important: joblib works best with simple, picklable arguments.
        tasks = []
        for _, row in midi_wav_df.iterrows():
            row_data_dict = row.to_dict()
            tasks.append(delayed(process_file_pair_worker)(
                row_data_dict, 
                directory_path_arg, 
                pad_before_arg, 
                pad_after_arg, 
                fix_length_arg, 
                worker_memory_limit_arg # Pass per-process memory limit to worker
            ))

        logger.info(f"Submitting {len(tasks)} tasks to joblib.Parallel with {n_cores} cores.")
        with parallel_backend('loky', n_jobs=n_cores):
            all_results = Parallel(verbose=10)(tasks)
        logger.info(f"joblib.Parallel processing finished. Received {len(all_results)} results.")

        # Process results
        valid_results = [df for df in all_results if df is not None and not df.empty]
        total_valid = len(valid_results)
        total_failed = len(midi_wav_df) - total_valid
        logger.info(f"Processing results: {total_valid} valid, {total_failed} failed.")

        if batching_arg and valid_results:
            # Batching logic remains similar, but operates on `valid_results` list
            batch_data_accumulator.extend(valid_results)
            batch_sample_count = sum(len(df) for df in valid_results)
            logger.info(f"Accumulated {batch_sample_count} samples for batching.")
            
            # Reset batch_tracking from self, as it's an instance variable
            self.batch_tracking = 0 

            while batch_sample_count >= target_samples_per_batch and self.batch_tracking < num_batches_arg:
                current_samples_in_this_batch = 0
                current_batch_data_list = []
                remaining_data_accumulator = []
                
                for df_res in batch_data_accumulator:
                    if current_samples_in_this_batch + len(df_res) <= target_samples_per_batch and self.batch_tracking < num_batches_arg:
                        current_batch_data_list.append(df_res)
                        current_samples_in_this_batch += len(df_res)
                    else:
                        remaining_data_accumulator.append(df_res)
                
                if not current_batch_data_list and batch_data_accumulator and self.batch_tracking < num_batches_arg:
                    # If a single result is larger than batch, take it as one batch
                    current_batch_data_list = [batch_data_accumulator[0]]
                    current_samples_in_this_batch = len(current_batch_data_list[0])
                    remaining_data_accumulator = batch_data_accumulator[1:]

                if current_batch_data_list: # Ensure there's data to save
                    # Find next available batch number
                    while os.path.exists(os.path.join(dir_path_arg, f"{self.batch_tracking}_train.pkl")):
                        self.batch_tracking += 1
                    
                    if self.batch_tracking < num_batches_arg:
                        self._save_batch_sequential(current_batch_data_list, self.batch_tracking, dir_path_arg)
                        logger.info(f"Saved batch {self.batch_tracking} ({current_samples_in_this_batch} samples)")
                        self.batch_tracking += 1 # Increment after successful save
                    else:
                        logger.info("Reached num_batches_arg limit during batch formation.")
                        # Put data back if we can't save this batch
                        remaining_data_accumulator = current_batch_data_list + remaining_data_accumulator
                        current_samples_in_this_batch = 0 # Reset samples for this unsaved batch
                        break # Stop trying to form batches

                batch_data_accumulator = remaining_data_accumulator
                batch_sample_count -= current_samples_in_this_batch
                
                import gc
                gc.collect()
                if self.batch_tracking >= num_batches_arg:
                    logger.info(f"Reached target number of batches ({num_batches_arg}) during parallel result processing.")
                    break

            # Save any remaining data if batch limit not reached
            if batch_data_accumulator and self.batch_tracking < num_batches_arg:
                logger.info(f"Saving remaining {len(batch_data_accumulator)} dataframes ({batch_sample_count} samples) as final batch.")
                while os.path.exists(os.path.join(dir_path_arg, f"{self.batch_tracking}_train.pkl")):
                    self.batch_tracking += 1
                if self.batch_tracking < num_batches_arg:
                    self._save_batch_sequential(batch_data_accumulator, self.batch_tracking, dir_path_arg)
                    logger.info(f"Saved final batch {self.batch_tracking} ({batch_sample_count} samples)")
                    self.batch_tracking +=1
                else:
                    logger.info("Final batch not saved as num_batches_arg limit was reached.")
        
        elif not batching_arg and valid_results: # No batching - combine individual files (if this mode is still desired)
            logger.info("Non-batching mode: combining individual results from parallel processing.")
            # This part needs to be re-evaluated. If not batching, where do individual files come from?
            # The worker returns DataFrames. We can save them individually or concat them here.
            # For now, let's assume we concat them into self.notes_collection like the original sequential non-batching mode.
            
            # Clean up any old individual files if they exist from a previous run mode
            # for f_old in os.listdir(dir_path_arg):
            #     if f_old.startswith('individual_') and f_old.endswith('.pkl'):
            #         os.remove(os.path.join(dir_path_arg, f_old))

            self.notes_collection = pd.concat(valid_results, ignore_index=True) if valid_results else pd.DataFrame()
            del valid_results
            import gc
            gc.collect()

            if not self.notes_collection.empty:
                # Filter problematic tracks (needs self.id_len_dict)
                # Ensure id_len_dict is available or passed if this logic is kept
                problematic_tracks=[]
                for r_idx, r_val in tqdm(self.notes_collection.iterrows(), total=self.notes_collection.shape[0], desc="Final check for non-batching"):
                    if r_val.start > self.id_len_dict[r_val.track_id]: # Requires self.id_len_dict
                        problematic_tracks.append(r_val.track_id)
                self.notes_collection = self.notes_collection[~self.notes_collection.track_id.isin(problematic_tracks)]
                
                output_path = os.path.join(dir_path_arg, "dataset_full.pkl")
                self.notes_collection.to_pickle(output_path)
                logger.info(f"Saved full dataset ({len(self.notes_collection)} notes) to {output_path} (non-batching mode)")
            else:
                logger.warning("No valid results to combine for non-batching mode.")

        logger.info(f"_process_parallel_optimized_new_worker complete: {total_valid} valid, {total_failed} failed")
        return total_valid

    def _save_batch_sequential(self, df_list, batch_idx, dir_path):
        """
        Save a batch of DataFrames to disk as pickle files.
        
        :param df_list: List of DataFrames to save
        :param batch_idx: Index of the batch (used for naming the output file)
        :param dir_path: Directory path where the pickle files will be saved
        """
        if not df_list:
            return

        combined_df = pd.concat(df_list, ignore_index=True)
        output_file = os.path.join(dir_path, f"audio_set_batch_{batch_idx}.pkl")
        combined_df.to_pickle(output_file)

        logger.info(f"Saved batch {batch_idx}: {len(combined_df)} records to {output_file}")

    # --- Deprecated: Legacy parallel processing function ---
    def _process_parallel(self, process_func, batching, dir_path, num_batches, memory_check_func, memory_limit_gb):
        """
        Legacy parallel processing function. This is now handled by create_audio_set directly.
        Kept for backwards compatibility but not intended for direct use.
        """
        logger.warning("Warning: _process_parallel is deprecated. Please use create_audio_set directly.")
        
        total_processed = 0
        df_list = []
        batch_idx = 0

        num_parallel_jobs = min(os.cpu_count(), 6)  # Limit to 6 parallel jobs

        with parallel_backend('loky', n_jobs=num_parallel_jobs):
            for i, row in enumerate(tqdm(self.midi_wav_map.itertuples(index=False), desc="Processing file pairs")):
                result = process_func(row)
                if result is not None:
                    df_list.append(result)
                    total_processed += 1

                # Batch saving logic
                if batching and len(df_list) >= 1000:
                    self._save_batch_sequential(df_list, batch_idx, dir_path)
                    batch_idx += 1
                    df_list.clear()  # Clear list after saving

                # Memory check
                if i % 100 == 0:
                    memory_check_func()

        # Final batch save
        if df_list:
            self._save_batch_sequential(df_list, batch_idx, dir_path)

        logger.info(f"Parallel processing complete: {total_processed} file pairs processed.")
        return total_processed
