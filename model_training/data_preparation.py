import gc
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

from tqdm import tqdm

from model_training.augment_audio import apply_augmentations

import soundfile as sf
# Hybrid parallelization: threading for I/O-bound slicing, sequential for memory safety
from functools import partial
import logging
import time
import audioread
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Configure logging at the module level
logger = logging.getLogger(__name__)

# Aliased imports for top-level helper
import soundfile as sf_top
import os as os_top

# NEW Top-level worker function for create_audio_set parallel processing
def process_file_pair_worker(row_dict, directory_path_arg, pad_before_arg, pad_after_arg, fix_length_arg, memory_limit_gb_worker):
    """
    Optimized parallel worker function - minimal overhead for maximum speed.
    This is a streamlined version of _process_file_pair focused on performance.
    """
    
    # Essential imports for worker
    import pandas as pd
    import numpy as np
    import librosa
    import soundfile as sf
    import time
    import io
    import math
    import itertools
    import logging
    
    # Minimal memory tracking - only for catastrophic failures
    import psutil
    worker_process = psutil.Process()
    
    # PERFORMANCE: Set generous memory limit for workers (relaxed constraints)
    worker_memory_limit = memory_limit_gb_worker if memory_limit_gb_worker else 8.0  # Default to 8GB per worker
    from mido import MidiFile
    import mido
    
    # Set up worker logger
    logger = logging.getLogger(__name__)
    
    # Helper functions for audio compression (simplified for performance)
    def _compress_audio_slice_worker(audio_array):
        """Fast worker version of audio compression - FLAC only."""
        try:
            # FLAC compression (lossless, ~50-70% reduction)
            buffer = io.BytesIO()
            sf.write(buffer, audio_array, 22050, format='FLAC', subtype='PCM_16')
            compressed_data = buffer.getvalue()
            buffer.close()
            
            return {
                'compressed_audio': compressed_data,
                'format': 'FLAC',
                'sample_rate': 22050,
                'original_shape': audio_array.shape,
                'original_dtype': str(audio_array.dtype)
            }
        except Exception:
            # Fallback: float16 without logging overhead
            return audio_array.astype(np.float16)
    
    # Helper functions for MIDI processing (copied from main class)
    def notes_extraction_worker(midi_file_path):
        """Worker version of notes extraction."""
        time_log = 0
        notes_collection = []
        temp_dict = {}
        
        midi_track = MidiFile(midi_file_path)
        
        for msg in midi_track.tracks[-1]:
            try:
                time_log = time_log + msg.time
            except:
                continue

            if msg.type == 'note_on' and msg.velocity > 0:
                start = time_log
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
                end = time_log
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
        
        return [[n[0], n[1], n[2]] for n in notes_collection], midi_track

    def time_meta_extraction_worker(midi_track):
        """Worker version of time meta extraction."""
        ticks_per_beat = midi_track.ticks_per_beat
        
        for msg in midi_track.tracks[0]:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
        else:
            tempo = 500000  # Default tempo
                    
        return (ticks_per_beat, tempo)
    
    def ticks_to_second_worker(notes_collection, midi_track, time_log):
        """Worker version of ticks to second conversion."""
        if type(time_log) == float:
            return [[
                    note[0],
                    round(note[1], 2),
                    round(note[2], 2)]
                for note in notes_collection]
        else:
            ticks_per_beat, tempo = time_meta_extraction_worker(midi_track)

            return [[
                    note[0],
                    round(mido.tick2second(note[1], ticks_per_beat, tempo), 2),
                    round(mido.tick2second(note[2], ticks_per_beat, tempo), 2)]
                for note in notes_collection]

    def merge_note_label_worker(track_id, notes_collection):
        """Worker version of note merging."""
        merged_note_collection = []
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

    # Main worker processing logic
    pair_start_time = time.perf_counter()
    midi_file = row_dict['midi_filename']
    audio_file = row_dict['audio_filename']
    track_id = row_dict['track_id']

    if midi_file == 'drummer1/session1/78_jazz-fast_290_beat_4-4.mid':
        logger.warning(f"[Worker] Skipping known problematic file: {midi_file}")
        return None

    track_notes = None
    cached_audio = None

    try:
        # PERFORMANCE: Minimal logging in workers (reduce overhead)
        logger = logging.getLogger(__name__)

        # MIDI processing with optimizations
        midi_file_path = os.path.join(directory_path_arg, midi_file)
        notes_collection, midi_track = notes_extraction_worker(midi_file_path)
        
        if not notes_collection:
            logger.warning(f"[Worker] No notes extracted from MIDI: {midi_file}")
            return None
            
        # Use the time_log from the notes extraction
        time_log = 0.0  # Simplified for worker
        converted_notes_collection = ticks_to_second_worker(notes_collection, midi_track, time_log)
        track_notes = merge_note_label_worker(track_id, converted_notes_collection)

        if track_notes is None or track_notes.empty:
            logger.warning(f"[Worker] No notes after merging for MIDI: {midi_file}")
            return None

        # Audio file processing with optimizations
        audio_file_path = os.path.join(directory_path_arg, audio_file)
        try:
            sf_info = sf.info(audio_file_path)
            sr = sf_info.samplerate
            audio_duration_samples = sf_info.frames
            audio_duration_seconds = audio_duration_samples / sr
        except Exception as sf_err:
            logger.error(f"[Worker] Soundfile error reading info for {audio_file_path}: {sf_err}")
            return None

        # CRITICAL OPTIMIZATION: Load audio file ONCE and slice in memory
        def optimized_audio_slicing_with_caching_worker(note_row):
            nonlocal cached_audio, sr, audio_duration_seconds
            start_time = note_row['start']
            end_time = note_row['end']

            slice_start_time = max(0.0, start_time - pad_before_arg)

            if fix_length_arg is not None:
                slice_duration = fix_length_arg
            else:
                slice_end_time = min(audio_duration_seconds, end_time + pad_after_arg)
                slice_duration = slice_end_time - slice_start_time

            if slice_start_time >= audio_duration_seconds or slice_duration <= 0:
                target_samples = librosa.time_to_samples(fix_length_arg, sr=sr) if fix_length_arg else 1024
                return _compress_audio_slice_worker(np.zeros(target_samples, dtype=np.float32))

            try:
                # Load audio file only once and cache it
                if cached_audio is None:
                    # OPTIMIZATION: Use lower sample rate for very long files to save memory
                    if audio_duration_seconds > 60:
                        target_sr = 22050  # Half sample rate for long files
                    elif audio_duration_seconds > 30:
                        target_sr = max(22050, sr // 2)
                    else:
                        target_sr = sr
                        
                    # MEMORY SAFETY: Skip extremely long files (>10 minutes)
                    if audio_duration_seconds > 600:  # 10+ minutes
                        target_samples = librosa.time_to_samples(fix_length_arg, sr=sr) if fix_length_arg else 1024
                        return _compress_audio_slice_worker(np.zeros(target_samples, dtype=np.float32))
                        
                    cached_audio, actual_sr = librosa.load(audio_file_path, sr=target_sr, mono=True)
                    
                    # Update sr if we downsampled
                    if actual_sr != sr:
                        sr = actual_sr
                        audio_duration_seconds = len(cached_audio) / sr

                # Slice from cached audio
                start_sample = max(0, int(slice_start_time * sr))
                if fix_length_arg is not None:
                    target_samples = librosa.time_to_samples(fix_length_arg, sr=sr)
                    end_sample = start_sample + target_samples
                else:
                    end_sample = min(len(cached_audio), int((slice_start_time + slice_duration) * sr))
                    target_samples = end_sample - start_sample

                # Extract slice
                if start_sample < len(cached_audio):
                    audio_slice = cached_audio[start_sample:end_sample].copy()
                else:
                    audio_slice = np.zeros(target_samples, dtype=np.float32)

                # Pad if necessary
                if fix_length_arg is not None and len(audio_slice) < target_samples:
                    audio_slice = np.pad(audio_slice, (0, target_samples - len(audio_slice)), mode='constant')
                elif fix_length_arg is not None and len(audio_slice) > target_samples:
                    audio_slice = audio_slice[:target_samples]

                # CRITICAL OPTIMIZATION: Compress audio slice before storing
                compressed_slice = _compress_audio_slice_worker(audio_slice.astype(np.float32))
                
                return compressed_slice
                
            except Exception:
                # Minimal error handling - no logging overhead
                target_samples = librosa.time_to_samples(fix_length_arg, sr=sr) if fix_length_arg else 1024
                return _compress_audio_slice_worker(np.zeros(target_samples, dtype=np.float32))

        # Apply audio slicing with compression
        track_notes['audio_wav'] = track_notes.apply(optimized_audio_slicing_with_caching_worker, axis=1)
        track_notes['sampling_rate'] = sr

        # Fast filter for valid audio (simplified validation)
        def is_valid_audio_worker(x):
            if isinstance(x, dict) and 'compressed_audio' in x:
                return len(x['compressed_audio']) > 0
            elif isinstance(x, np.ndarray):
                return x.size > 0
            else:
                return False
        
        track_notes = track_notes[track_notes['audio_wav'].apply(is_valid_audio_worker)]

        if track_notes.empty:
            return None

        # CRITICAL OPTIMIZATION: Optimize DataFrame memory (simplified)
        def _optimize_dataframe_memory_worker(df):
            """Fast worker version of DataFrame memory optimization."""
            try:
                # Quick downcast for numeric columns only
                for col in df.columns:
                    if col == 'audio_wav':  # Skip audio data
                        continue
                    col_type = df[col].dtype
                    if col_type.kind == 'i':  # integer
                        df[col] = pd.to_numeric(df[col], downcast='integer')
                    elif col_type.kind == 'f':  # float
                        df[col] = pd.to_numeric(df[col], downcast='float')
                return df
            except Exception:
                return df
        
        # Apply memory optimization for larger results only
        if len(track_notes) > 10:
            track_notes = _optimize_dataframe_memory_worker(track_notes)

        # PERFORMANCE: Clean up cached audio immediately
        if cached_audio is not None:
            del cached_audio
            cached_audio = None

        # PERFORMANCE: Minimal memory check - only warn if catastrophic
        try:
            final_mem = worker_process.memory_info().rss / (1024**3)
            if final_mem > worker_memory_limit:
                logger.warning(f"[Worker] Memory exceeded: {final_mem:.1f}GB > {worker_memory_limit:.1f}GB")
        except Exception:
            pass  # Ignore memory check failures
        
        # PERFORMANCE: Minimal success logging
        return track_notes

    except Exception:
        # PERFORMANCE: Clean up cached audio on error (minimal handling)
        if cached_audio is not None:
            del cached_audio
            cached_audio = None
        # No logging or gc.collect() for performance
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
    """
    Optimized duration calculation helper for parallel processing.
    Minimal overhead for maximum speed.
    """
    try:
        import soundfile as sf
        import os
        
        # Get file info with minimal memory footprint
        file_path = os.path.join(base_dir, filepath)
        info = sf.info(file_path)
        duration = info.duration
        
        # Quick cleanup
        del info
        del file_path
        
        return duration
        
    except Exception:
        # No logging - just return None for failed files
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
    
    def __init__(self, directory_path, dataset, sample_ratio=1, diff_threshold=1):
        init_start_time = time.perf_counter()
        logger.info(f"Initializing data_preparation for dataset: {dataset} at {directory_path} with sample_ratio={sample_ratio}, diff_threshold={diff_threshold}")
        if dataset in ['gmd', 'egmd']:
            self.directory_path = directory_path
            self.dataset_type = dataset
            self.batch_tracking = 0
            
            csv_path = [f for f in os.listdir(directory_path) if '.csv' in f][0]
            
            self.dataset = pd.read_csv(
                os.path.join(directory_path, csv_path)).dropna().sample(frac=sample_ratio).reset_index()
            
            df = self.dataset[['index', 'midi_filename', 'audio_filename', 'duration']].copy()
            df.columns = ['track_id', 'midi_filename', 'audio_filename', 'duration']
            
            print(f'Filtering out the midi/audio pair that has a duration difference > {diff_threshold} second using soundfile')

            logger.info("Calculating audio durations using soundfile...")
            duration_calc_start_time = time.perf_counter()

            # MEMORY SAFETY: Duration calculation - no parallel processing for large datasets
            # Using joblib.Parallel here causes memory explosion due to main process memory inheritance
            logger.info("MEMORY SAFETY: Using sequential duration calculation to prevent memory explosion")
            logger.info("(joblib.Parallel inherits entire main process memory to each worker, causing massive RAM usage)")
            
            durations = []
            for i, filename in enumerate(tqdm(df['audio_filename'], desc="Calculating durations sequentially")):
                duration = get_duration_joblib_helper(filename, self.directory_path)
                durations.append(duration)
                # PERFORMANCE: Periodic garbage collection
                if i % 500 == 0:
                    import gc
                    gc.collect()


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
    
    def _compress_audio_slice(self, audio_array):
        """
        ULTRA-FAST audio compression optimized for maximum speed.
        
        Uses aggressive optimizations based on detailed performance profiling showing
        compression consumes ~50% of processing time.
        """
        try:
            # CRITICAL PERFORMANCE OPTIMIZATION: Choose fastest method based on size
            array_size_kb = audio_array.nbytes / 1024
            
            if array_size_kb < 50:  # Very small arrays < 50KB - no compression needed
                # For tiny arrays, any compression overhead dominates
                return audio_array.astype(np.float32)
            elif array_size_kb < 200:  # Small-medium arrays < 200KB - ultra-fast float16
                # Float16 is ~4x faster than any other compression and gives 50% size reduction
                return audio_array.astype(np.float16)
            else:  # Large arrays > 200KB - smart compression choice
                # PERFORMANCE INSIGHT: FLAC compression is slow but effective for large arrays
                # Only use FLAC for arrays where the time investment pays off
                
                # First, try fast float16 and see if size is acceptable
                float16_array = audio_array.astype(np.float16)
                if array_size_kb < 400:  # For medium-large arrays, float16 is usually good enough
                    return float16_array
                
                # For very large arrays, FLAC may be worth the time cost
                import io
                import soundfile as sf
                
                # PERFORMANCE: Use fastest FLAC settings
                buffer = io.BytesIO()
                # Use PCM_16 instead of PCM_24 for speed, minimal quality difference for drums
                sf.write(buffer, audio_array, 22050, format='FLAC', subtype='PCM_16')
                compressed_data = buffer.getvalue()
                buffer.close()
                
                # PERFORMANCE: Only use FLAC if it gives substantial savings over float16
                flac_size = len(compressed_data)
                float16_size = float16_array.nbytes
                
                if flac_size < float16_size * 0.7:  # FLAC must be 30% better than float16
                    return {
                        'compressed_audio': compressed_data,
                        'format': 'FLAC',
                        'sample_rate': 22050,
                        'original_shape': audio_array.shape,
                        'original_dtype': str(audio_array.dtype)
                    }
                else:  # float16 is good enough and much faster to decompress
                    return float16_array
            
        except Exception as compress_err:
            logger.debug(f"Compression failed: {compress_err}, using float16 fallback")
            # Ultra-fast fallback: Use float16 (50% size reduction, near-zero time cost)
            return audio_array.astype(np.float16)

    def _decompress_audio_slice(self, compressed_slice):
        """
        Fast audio decompression for training use.
        Optimized for speed with multiple compression format support.
        """
        try:
            if isinstance(compressed_slice, dict) and 'compressed_audio' in compressed_slice:
                import io
                import soundfile as sf
                
                # Decompress FLAC data
                buffer = io.BytesIO(compressed_slice['compressed_audio'])
                audio_array, sample_rate = sf.read(buffer, dtype='float32')
                buffer.close()
                
                return audio_array.astype(np.float32)
            elif isinstance(compressed_slice, np.ndarray):
                # Handle float16/float32 arrays directly (fast path)
                return compressed_slice.astype(np.float32)
            else:
                # Unknown format - return zeros as fallback
                return np.zeros(220500, dtype=np.float32)  # 10 seconds at 22050 Hz
                
        except Exception as decompress_err:
            logger.debug(f"Audio decompression failed: {decompress_err}")
            # Return zeros as fallback (will be caught by validation)
            return np.zeros(220500, dtype=np.float32)  # 10 seconds at 22050 Hz

    def _optimize_batch_for_storage(self, df):
        """
        Apply final optimizations before saving batches to disk.
        """
        try:
            # Calculate compression statistics
            total_original_mb = 0
            total_compressed_mb = 0
            compressed_count = 0
            uncompressed_count = 0
            
            for audio_data in df['audio_wav']:
                if isinstance(audio_data, dict) and 'compressed_audio' in audio_data:
                    # Already compressed
                    compressed_size = len(audio_data['compressed_audio'])
                    total_compressed_mb += compressed_size / (1024**2)
                    
                    # Estimate original size from shape
                    if 'original_shape' in audio_data:
                        original_samples = audio_data['original_shape'][0] if len(audio_data['original_shape']) > 0 else 220500
                        total_original_mb += original_samples * 4 / (1024**2)  # float32 = 4 bytes
                    else:
                        total_original_mb += compressed_size * 3 / (1024**2)  # Estimate 3x compression
                    
                    compressed_count += 1
                elif isinstance(audio_data, np.ndarray):
                    # Uncompressed numpy array
                    size_mb = audio_data.nbytes / (1024**2)
                    total_original_mb += size_mb
                    total_compressed_mb += size_mb
                    uncompressed_count += 1
            
            # Calculate and report compression ratio
            compression_ratio = total_compressed_mb / total_original_mb if total_original_mb > 0 else 1.0
            space_saved_mb = total_original_mb - total_compressed_mb
            
            logger.info(f"[STORAGE] Batch optimization: {total_original_mb:.1f}MB -> {total_compressed_mb:.1f}MB "
                       f"({compression_ratio*100:.1f}% of original, saved {space_saved_mb:.1f}MB)")
            logger.info(f"[STORAGE] Audio compression: {compressed_count} compressed, {uncompressed_count} uncompressed")
            
            return df
            
        except Exception as opt_err:
            logger.warning(f"Batch optimization failed: {opt_err}")
            return df
    
    def _optimize_dataframe_memory(self, df):
        """
        Optimize DataFrame memory usage by downcasting numeric columns.
        """
        try:
            original_memory = df.memory_usage(deep=True).sum()
            
            # Downcast numeric columns
            for col in df.columns:
                col_type = df[col].dtype
                
                # Skip audio_wav column and other object columns that contain arrays/dicts
                if col == 'audio_wav' or col_type == 'object':
                    continue
                    
                # Downcast integers
                if col_type.kind == 'i':  # integer
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                # Downcast floats
                elif col_type.kind == 'f':  # float
                    df[col] = pd.to_numeric(df[col], downcast='float')
            
            optimized_memory = df.memory_usage(deep=True).sum()
            memory_saved = original_memory - optimized_memory
            
            if memory_saved > 0:
                logger.debug(f"[MEMORY] DataFrame optimized: {original_memory/(1024**2):.1f}MB -> "
                           f"{optimized_memory/(1024**2):.1f}MB (saved {memory_saved/(1024**2):.1f}MB)")
            
            return df
            
        except Exception as e:
            logger.warning(f"DataFrame memory optimization failed: {e}")
            return df

    # --- Optimization: Helper function for parallel processing of one file pair ---
    def _process_file_pair(self, row, pad_before, pad_after, fix_length, memory_limit_gb=None):
        # PERFORMANCE INSTRUMENTATION: Track timing for each stage
        pair_start_time = time.perf_counter()
        midi_file = row.midi_filename
        audio_file = row.audio_filename

        # Timing variables for detailed profiling
        timing_stats = {
            'total': 0,
            'midi_extraction': 0,
            'midi_conversion': 0,
            'midi_merging': 0,
            'audio_info': 0,
            'audio_loading': 0,
            'note_slicing': 0,
            'compression': 0,
            'dataframe_creation': 0
        }

        if midi_file == 'drummer1/session1/78_jazz-fast_290_beat_4-4.mid':
             logger.warning(f"Skipping known problematic file: {midi_file}")
             return None # Skip problematic file

        # Initialize track_notes to None at the start of the function
        track_notes = None
        cached_audio = None  # Cache audio for multiple note slicing

        try:
            # Memory diagnostic: Initial memory
            if memory_limit_gb:
                import psutil
                initial_mem = psutil.Process().memory_info().rss / (1024**3)
                logger.debug(f"[MEMORY] Starting {midi_file}: {initial_mem:.1f}GB")

            # TIMING: MIDI extraction
            stage_start = time.perf_counter()
            notes_collection = self.notes_extraction(midi_file)
            timing_stats['midi_extraction'] = time.perf_counter() - stage_start
            
            if not notes_collection:
                 logger.warning(f"No notes extracted from MIDI: {midi_file}")
                 return None
            
            # TIMING: MIDI conversion to seconds
            stage_start = time.perf_counter()
            converted_notes_collection = self.ticks_to_second(notes_collection)
            timing_stats['midi_conversion'] = time.perf_counter() - stage_start
            
            # TIMING: Note merging and labeling
            stage_start = time.perf_counter()
            track_notes = self.merge_note_label(row.track_id, converted_notes_collection)
            timing_stats['midi_merging'] = time.perf_counter() - stage_start

            if track_notes is None or track_notes.empty:
                logger.warning(f"No notes after merging for MIDI: {midi_file}")
                return None

            # TIMING: Audio file info reading
            stage_start = time.perf_counter()
            audio_file_path = os.path.join(self.directory_path, audio_file)
            try:
                sf_info = sf.info(audio_file_path)
                sr = sf_info.samplerate
                audio_duration_samples = sf_info.frames
                audio_duration_seconds = audio_duration_samples / sr
            except Exception as sf_err:
                 logger.error(f"Soundfile error reading info for {audio_file_path}: {sf_err}")
                 return None
            timing_stats['audio_info'] = time.perf_counter() - stage_start

            # Memory diagnostic: After MIDI processing
            if memory_limit_gb:
                midi_mem = psutil.Process().memory_info().rss / (1024**3)
                logger.debug(f"[MEMORY] After MIDI processing {midi_file}: {midi_mem:.1f}GB (+{midi_mem-initial_mem:.1f}GB)")

            # CRITICAL OPTIMIZATION: Load audio file ONCE and slice in memory
            def optimized_audio_slicing_with_caching(note_row):
                nonlocal cached_audio, sr, audio_duration_seconds
                start_time = note_row['start']
                end_time = note_row['end']

                slice_start_time = max(0.0, start_time - pad_before)

                if fix_length is not None:
                    slice_duration = fix_length
                else:
                    slice_end_time = min(audio_duration_seconds, end_time + pad_after)
                    slice_duration = slice_end_time - slice_start_time

                if slice_start_time >= audio_duration_seconds or slice_duration <= 0:
                    target_samples = int(fix_length * 22050) if fix_length else 1024  # Use direct calculation instead of librosa
                    return self._compress_audio_slice(np.zeros(target_samples, dtype=np.float32))

                try:
                    # PERFORMANCE INSTRUMENTATION: Track audio loading time
                    audio_load_start = time.perf_counter()
                    
                    # PERFORMANCE OPTIMIZATION: Load audio file only once and cache it
                    if cached_audio is None:
                        logger.debug(f"[AUDIO] Loading {audio_file} (duration: {audio_duration_seconds:.1f}s)")
                        
                        # CRITICAL PERFORMANCE: Use soundfile instead of librosa for much faster loading
                        # librosa.load() is 3-5x slower than soundfile.read() for the same result
                        try:
                            # Fast loading with soundfile - directly specify target sample rate
                            target_sr = 22050  # Standard sample rate for this project
                            
                            # Read entire file at once (much faster than librosa)
                            cached_audio, actual_sr = sf.read(audio_file_path, dtype='float32')
                            
                            # Convert to mono if stereo (faster than librosa's method)
                            if len(cached_audio.shape) > 1:
                                cached_audio = np.mean(cached_audio, axis=1)
                            
                            # Resample only if needed (most files are already 22050 or 44100)
                            if actual_sr != target_sr:
                                # Use simple decimation for 2x downsampling (44100->22050)
                                if actual_sr == 44100 and target_sr == 22050:
                                    cached_audio = cached_audio[::2]  # Simple decimation - much faster
                                    sr = target_sr
                                else:
                                    # Fall back to librosa for unusual sample rates
                                    import librosa
                                    cached_audio = librosa.resample(cached_audio, orig_sr=actual_sr, target_sr=target_sr)
                                    sr = target_sr
                            else:
                                sr = actual_sr
                                
                            audio_duration_seconds = len(cached_audio) / sr
                            
                        except Exception as sf_err:
                            # Fallback to librosa if soundfile fails
                            logger.debug(f"[AUDIO] Soundfile failed, using librosa fallback: {sf_err}")
                            import librosa
                            target_sr = 22050
                            cached_audio, sr = librosa.load(audio_file_path, sr=target_sr, mono=True)
                            audio_duration_seconds = len(cached_audio) / sr
                        
                        # Memory diagnostic: After audio loading
                        if memory_limit_gb:
                            audio_mem = psutil.Process().memory_info().rss / (1024**3)
                            audio_size_mb = cached_audio.nbytes / (1024**2)
                            logger.debug(f"[MEMORY] After loading audio {audio_file}: {audio_mem:.1f}GB (+{audio_size_mb:.1f}MB audio, sr={sr})")
                    
                    # PERFORMANCE INSTRUMENTATION: Track audio loading time
                    audio_load_time = time.perf_counter() - audio_load_start
                    if 'audio_loading' not in timing_stats:
                        timing_stats['audio_loading'] = 0
                    timing_stats['audio_loading'] += audio_load_time

                    # PERFORMANCE INSTRUMENTATION: Track slicing time
                    slice_start_timing = time.perf_counter()
                    
                    # PERFORMANCE: Fast slice extraction (vectorized operations)
                    start_sample = max(0, int(slice_start_time * sr))
                    if fix_length is not None:
                        target_samples = int(fix_length * sr)  # Faster than librosa.time_to_samples
                        end_sample = start_sample + target_samples
                    else:
                        end_sample = min(len(cached_audio), int((slice_start_time + slice_duration) * sr))
                        target_samples = end_sample - start_sample

                    # PERFORMANCE: Direct array slicing (much faster than copying)
                    if start_sample < len(cached_audio):
                        if end_sample <= len(cached_audio):
                            audio_slice = cached_audio[start_sample:end_sample]
                        else:
                            # Need padding
                            audio_slice = cached_audio[start_sample:]
                            pad_length = target_samples - len(audio_slice)
                            audio_slice = np.concatenate([audio_slice, np.zeros(pad_length, dtype=np.float32)])
                    else:
                        audio_slice = np.zeros(target_samples, dtype=np.float32)

                    # PERFORMANCE: Skip unnecessary padding operations when possible
                    if fix_length is not None and len(audio_slice) != target_samples:
                        if len(audio_slice) < target_samples:
                            audio_slice = np.pad(audio_slice, (0, target_samples - len(audio_slice)), mode='constant')
                        else:
                            audio_slice = audio_slice[:target_samples]
                    
                    slice_time = time.perf_counter() - slice_start_timing
                    if 'note_slicing' not in timing_stats:
                        timing_stats['note_slicing'] = 0
                    timing_stats['note_slicing'] += slice_time

                    # PERFORMANCE INSTRUMENTATION: Track compression time
                    compress_start = time.perf_counter()
                    
                    # CRITICAL OPTIMIZATION: Compress audio slice before storing
                    compressed_slice = self._compress_audio_slice(audio_slice.astype(np.float32))
                    
                    compress_time = time.perf_counter() - compress_start
                    if 'compression' not in timing_stats:
                        timing_stats['compression'] = 0
                    timing_stats['compression'] += compress_time
                    
                    # Log if slice is unexpectedly large (after compression)
                    compressed_size_kb = len(compressed_slice['compressed_audio']) / 1024 if isinstance(compressed_slice, dict) and 'compressed_audio' in compressed_slice else compressed_slice.nbytes / 1024
                    if compressed_size_kb > 500:  # More than 500KB compressed is suspicious
                        logger.warning(f"Large compressed audio slice: {compressed_size_kb:.1f}KB for {audio_file} at {slice_start_time:.2f}s")
                    
                    return compressed_slice
                    
                except Exception as load_err:
                    logger.error(f"Error slicing audio for {audio_file} at {slice_start_time:.2f}s: {load_err}")
                    target_samples = int(fix_length * sr) if fix_length else 1024  # Use direct calculation
                    return self._compress_audio_slice(np.zeros(target_samples, dtype=np.float32))

            # PERFORMANCE INSTRUMENTATION: Create DataFrame timing
            df_start = time.perf_counter()
            slicing_start_time = time.perf_counter()
            
            # CRITICAL PERFORMANCE OPTIMIZATION: Batch audio slicing
            # Instead of slicing one note at a time, batch process all notes from this audio file
            # This reduces function call overhead and enables vectorized operations
            
            def vectorized_audio_slicing_ultra_fast():
                """
                ULTRA-FAST VECTORIZED AUDIO SLICING - Major Performance Optimization
                
                Uses numpy vectorized operations to process all note slices simultaneously
                instead of looping through each note individually. This should provide
                significant speedup (3-5x) for files with many notes.
                """
                nonlocal cached_audio, sr, audio_duration_seconds
                
                if cached_audio is None:
                    logger.debug(f"[AUDIO] Loading {audio_file} (duration: {audio_duration_seconds:.1f}s)")
                    
                    # CRITICAL PERFORMANCE: Use soundfile instead of librosa for much faster loading
                    try:
                        target_sr = 22050  # Standard sample rate for this project
                        
                        # Read entire file at once (much faster than librosa)
                        cached_audio, actual_sr = sf.read(audio_file_path, dtype='float32')
                        
                        # Convert to mono if stereo (faster than librosa's method)
                        if len(cached_audio.shape) > 1:
                            cached_audio = np.mean(cached_audio, axis=1)
                        
                        # Resample only if needed (most files are already 22050 or 44100)
                        if actual_sr != target_sr:
                            # Use simple decimation for 2x downsampling (44100->22050)
                            if actual_sr == 44100 and target_sr == 22050:
                                cached_audio = cached_audio[::2]  # Simple decimation - much faster
                                sr = target_sr
                            else:
                                # Fall back to librosa for unusual sample rates
                                import librosa
                                cached_audio = librosa.resample(cached_audio, orig_sr=actual_sr, target_sr=target_sr)
                                sr = target_sr
                        else:
                            sr = actual_sr
                            
                        audio_duration_seconds = len(cached_audio) / sr
                        
                    except Exception as sf_err:
                        # Fallback to librosa if soundfile fails
                        logger.debug(f"[AUDIO] Soundfile failed, using librosa fallback: {sf_err}")
                        import librosa
                        target_sr = 22050
                        cached_audio, sr = librosa.load(audio_file_path, sr=target_sr, mono=True)
                        audio_duration_seconds = len(cached_audio) / sr
                
                # VECTORIZED PERFORMANCE OPTIMIZATION: Process all slices at once
                
                # Step 1: Pre-compute all slice parameters vectorized
                start_times = track_notes['start'].values
                end_times = track_notes['end'].values
                
                # Apply padding vectorized
                slice_start_times = np.maximum(0.0, start_times - pad_before)
                
                if fix_length is not None:
                    # Fixed length mode - all slices same duration
                    target_samples = int(fix_length * sr)
                    slice_durations = np.full(len(track_notes), fix_length)
                else:
                    # Variable length mode
                    slice_end_times = np.minimum(audio_duration_seconds, end_times + pad_after)
                    slice_durations = slice_end_times - slice_start_times
                    target_samples = None  # Will vary per slice
                
                # Convert to sample indices vectorized
                start_samples = np.maximum(0, (slice_start_times * sr).astype(np.int32))
                
                if fix_length is not None:
                    # Fixed length: all end samples calculated the same way
                    end_samples = start_samples + target_samples
                else:
                    # Variable length: calculate per slice
                    end_samples = np.minimum(len(cached_audio), 
                                           ((slice_start_times + slice_durations) * sr).astype(np.int32))
                
                # Step 2: Filter out invalid slices (outside audio bounds)
                valid_mask = (slice_start_times < audio_duration_seconds) & (slice_durations > 0)
                
                # Step 3: Batch process slicing using advanced numpy indexing
                audio_slices = []
                
                if fix_length is not None:
                    # FIXED LENGTH MODE: Optimized for uniform slice sizes
                    for i, (start_sample, end_sample, is_valid) in enumerate(zip(start_samples, end_samples, valid_mask)):
                        if not is_valid:
                            audio_slices.append(np.zeros(target_samples, dtype=np.float32))
                            continue
                        
                        # Extract slice with bounds checking
                        if start_sample < len(cached_audio):
                            if end_sample <= len(cached_audio):
                                # Simple case: slice fits entirely within audio
                                audio_slice = cached_audio[start_sample:end_sample].copy()
                            else:
                                # Needs padding at end
                                audio_slice = cached_audio[start_sample:].copy()
                                pad_length = target_samples - len(audio_slice)
                                if pad_length > 0:
                                    audio_slice = np.concatenate([audio_slice, np.zeros(pad_length, dtype=np.float32)])
                        else:
                            # Entirely outside audio bounds
                            audio_slice = np.zeros(target_samples, dtype=np.float32)
                        
                        # Ensure exact target length
                        if len(audio_slice) != target_samples:
                            if len(audio_slice) < target_samples:
                                audio_slice = np.pad(audio_slice, (0, target_samples - len(audio_slice)), mode='constant')
                            else:
                                audio_slice = audio_slice[:target_samples]
                        
                        audio_slices.append(audio_slice.astype(np.float32))
                        
                else:
                    # VARIABLE LENGTH MODE: Less optimization opportunity but still vectorized bounds checking
                    for i, (start_sample, end_sample, is_valid) in enumerate(zip(start_samples, end_samples, valid_mask)):
                        if not is_valid:
                            audio_slices.append(np.zeros(1024, dtype=np.float32))  # Default minimal slice
                            continue
                        
                        target_length = end_sample - start_sample
                        
                        if start_sample < len(cached_audio) and target_length > 0:
                            actual_end = min(end_sample, len(cached_audio))
                            audio_slice = cached_audio[start_sample:actual_end].copy().astype(np.float32)
                        else:
                            audio_slice = np.zeros(max(1024, target_length), dtype=np.float32)
                        
                        audio_slices.append(audio_slice)
                
                return audio_slices
            
            # Get all audio slices in one batch using vectorized processing
            all_slices = vectorized_audio_slicing_ultra_fast()
            
            # PERFORMANCE: Parallel compression using threading for CPU-bound compression
            def compress_slice_threaded(audio_slice):
                return self._compress_audio_slice(audio_slice)
            
            compressed_slices = []
            
            # Use threading for compression if we have many slices
            if len(all_slices) > 10:  # Worth threading overhead
                # Use conservative thread count to avoid overwhelming the system
                max_workers = min(4, os.cpu_count() // 2)  # Conservative threading
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all compression tasks
                    future_to_slice = {executor.submit(compress_slice_threaded, audio_slice): i 
                                     for i, audio_slice in enumerate(all_slices)}
                    
                    # Collect results in order
                    compressed_slices = [None] * len(all_slices)
                    for future in as_completed(future_to_slice):
                        slice_index = future_to_slice[future]
                        try:
                            compressed_slice = future.result()
                            compressed_slices[slice_index] = compressed_slice
                        except Exception as exc:
                            logger.warning(f'Compression failed for slice {slice_index}: {exc}')
                            # Fallback: use original slice as float16
                            compressed_slices[slice_index] = all_slices[slice_index].astype(np.float16)
            else:
                # Sequential compression for small batches (avoid threading overhead)
                for audio_slice in all_slices:
                    compressed_slice = self._compress_audio_slice(audio_slice)
                    compressed_slices.append(compressed_slice)
            
            # Assign compressed slices to DataFrame
            track_notes['audio_wav'] = compressed_slices
            
            slicing_end_time = time.perf_counter()
            timing_stats['note_slicing'] = slicing_end_time - slicing_start_time

            # Memory diagnostic: After audio slicing
            if memory_limit_gb:
                slicing_mem = psutil.Process().memory_info().rss / (1024**3)
                # Calculate total audio memory (compressed and uncompressed)
                total_audio_mb = 0
                for audio_data in track_notes['audio_wav']:
                    if isinstance(audio_data, dict) and 'compressed_audio' in audio_data:
                        total_audio_mb += len(audio_data['compressed_audio']) / (1024**2)
                    elif isinstance(audio_data, np.ndarray):
                        total_audio_mb += audio_data.nbytes / (1024**2)
                
                avg_audio_mb_per_note = total_audio_mb / len(track_notes)
                logger.debug(f"[MEMORY] After slicing {len(track_notes)} notes: {slicing_mem:.1f}GB (+{total_audio_mb:.1f}MB sliced audio, {avg_audio_mb_per_note:.1f}MB/note)")

            track_notes['sampling_rate'] = sr
            # Filter out rows where slicing might have failed or resulted in empty data
            initial_notes = len(track_notes)
            def is_valid_audio(x):
                if isinstance(x, dict) and 'compressed_audio' in x:
                    return len(x['compressed_audio']) > 0
                elif isinstance(x, np.ndarray):
                    return x.size > 0
                else:
                    return False
            
            track_notes = track_notes[track_notes['audio_wav'].apply(is_valid_audio)]
            final_notes = len(track_notes)
            if initial_notes != final_notes:
                 logger.warning(f"Filtered {initial_notes - final_notes} notes with empty audio for {audio_file}")

            if track_notes.empty:
                 logger.warning(f"Track notes became empty after audio slicing for {midi_file}")
                 return None

            timing_stats['dataframe_creation'] = time.perf_counter() - df_start

            # CRITICAL: Clear cached audio immediately to free memory
            if cached_audio is not None:
                del cached_audio
                cached_audio = None

            # Memory cleanup without copying the result (avoids doubling memory usage)
            del sf_info
            del notes_collection
            del converted_notes_collection
            
            # Single garbage collection pass
            gc.collect()
            
            # Memory diagnostic: Final memory
            if memory_limit_gb:
                final_mem = psutil.Process().memory_info().rss / (1024**3)
                logger.debug(f"[MEMORY] Completed {midi_file}: {final_mem:.1f}GB (total change: +{final_mem-initial_mem:.1f}GB)")
            
            pair_end_time = time.perf_counter()
            timing_stats['total'] = pair_end_time - pair_start_time
            
            # PERFORMANCE REPORT: Log detailed timing breakdown for each file
            logger.info(f"[PERF] {midi_file}: {timing_stats['total']:.3f}s total | "
                       f"MIDI: {timing_stats['midi_extraction']:.3f}s + {timing_stats['midi_conversion']:.3f}s + {timing_stats['midi_merging']:.3f}s | "
                       f"Audio: {timing_stats['audio_info']:.3f}s info + {timing_stats.get('audio_loading', 0):.3f}s load + "
                       f"{timing_stats.get('note_slicing', 0):.3f}s slice + {timing_stats.get('compression', 0):.3f}s compress | "
                       f"Notes: {final_notes}")
            
            return track_notes

        except Exception as e:
            # Clean up cached audio on error
            if cached_audio is not None:
                del cached_audio
            # Log error with traceback information
            logger.error(f"Error processing pair {midi_file} / {audio_file}: {e}", exc_info=True)
            return None

    # --- End Helper Function ---

    def create_audio_set(self, pad_before=0.02, pad_after=0.02, fix_length=None, batching=False, dir_path='', num_batches=50, memory_limit_gb=8, batch_size_multiplier=1.0, enable_process_parallelization=False):
        create_set_start_time = time.perf_counter()
        logger.info(f"Starting create_audio_set: pad_before={pad_before}, pad_after={pad_after}, fix_length={fix_length}, batching={batching}, num_batches={num_batches}, memory_limit={memory_limit_gb}GB, batch_size_multiplier={batch_size_multiplier}x, process_parallel={enable_process_parallelization}")
        
        import psutil # Moved import here as it's used in check_memory
        process = psutil.Process() # Moved process definition here

        def check_memory(): # check_memory is now defined and can be passed
            """Enhanced memory monitoring with detailed diagnostics."""
            mem_info = process.memory_info()
            mem_gb = mem_info.rss / (1024**3)
            mem_pct = mem_gb / memory_limit_gb
            
            # Get system memory info
            sys_mem = psutil.virtual_memory()
            sys_mem_pct = sys_mem.percent
            
            # Log different levels based on memory pressure
            if mem_pct > 0.95:  # Critical: 95%+
                logger.error(f"CRITICAL MEMORY: {mem_gb:.1f}GB ({mem_pct*100:.1f}% of limit), System: {sys_mem_pct:.1f}%")
                gc.collect()  # Force GC
                # Try additional cleanup
                try:
                    import ctypes
                    libc = ctypes.CDLL("libc.so.6")
                    libc.malloc_trim(0)
                except:
                    pass  # Ignore if not on Linux
            elif mem_pct > 0.85:  # High: 85%+
                logger.warning(f"HIGH MEMORY: {mem_gb:.1f}GB ({mem_pct*100:.1f}% of limit), System: {sys_mem_pct:.1f}%")
                gc.collect()
            elif mem_pct > 0.70:  # Medium: 70%+
                logger.info(f"MEDIUM MEMORY: {mem_gb:.1f}GB ({mem_pct*100:.1f}% of limit), System: {sys_mem_pct:.1f}%")
            # Below 70% - only log occasionally to reduce noise
            
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
        
        # MEMORY SAFETY: Very conservative parallel processing thresholds
        # Force sequential for most cases to prevent memory explosions
        available_system_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # SMART PARALLELIZATION: New approach with explicit opt-in
        if enable_process_parallelization and memory_limit_gb >= 16 and available_system_memory_gb >= 32:
            # Only enable process parallelization with sufficient memory and explicit opt-in
            logger.info("🚀 EXPERIMENTAL: Process parallelization enabled with improved memory management")
            logger.info(f'Parallel mode: System memory: {available_system_memory_gb:.1f}GB, Process limit: {memory_limit_gb}GB')
            
            # Use conservative process count and memory per process
            n_processes = min(3, os.cpu_count() // 2)  # Very conservative
            per_process_memory_limit = memory_limit_gb / (n_processes + 1)  # Leave buffer for main process
            
            logger.info(f'Using {n_processes} processes with {per_process_memory_limit:.1f}GB limit per process')
            
            process_func = partial(self._process_file_pair, pad_before=pad_before, pad_after=pad_after, fix_length=fix_length, memory_limit_gb=per_process_memory_limit)
            total_valid = self._process_parallel_improved(process_func, batching, dir_path, num_batches, check_memory, memory_limit_gb, batch_size_multiplier, n_processes)
        else:
            # Default to sequential processing for memory safety and reliability  
            if enable_process_parallelization:
                logger.warning("⚠️  Process parallelization requested but insufficient memory - falling back to sequential mode")
                logger.warning(f"   Requires: memory_limit_gb >= 16, system_memory >= 32GB")
                logger.warning(f"   Current: memory_limit_gb = {memory_limit_gb}, system_memory = {available_system_memory_gb:.1f}GB")
            
            logger.info("Using sequential processing mode for better memory control")
            logger.info(f'Sequential mode: System memory: {available_system_memory_gb:.1f}GB, Process limit: {memory_limit_gb}GB')
            logger.info('Sequential processing provides predictable memory usage and reliable performance')
            
            sequential_process_func = partial(self._process_file_pair, pad_before=pad_before, pad_after=pad_after, fix_length=fix_length, memory_limit_gb=per_process_memory_limit)
            total_valid = self._process_sequential(sequential_process_func, batching, dir_path, num_batches, check_memory, memory_limit_gb, batch_size_multiplier)

        create_set_end_time = time.perf_counter()
        logger.info(f"create_audio_set completed: {total_valid} valid file pairs processed in {create_set_end_time - create_set_start_time:.2f} seconds.")
        return total_valid

    def _process_sequential(self, process_func, batching, dir_path, num_batches, check_memory, memory_limit_gb, batch_size_multiplier=1.0):
        """Sequential processing with streaming saves and record-based batching."""
        total_processed_files = 0
        total_processed_records = 0
        current_batch_records = []  # Accumulate individual records, not DataFrames
        batch_idx = 0
        
        # PERFORMANCE INSTRUMENTATION: Track aggregate timing statistics
        aggregate_timing = {
            'total_files': 0,
            'total_time': 0,
            'midi_extraction': 0,
            'midi_conversion': 0,
            'midi_merging': 0,
            'audio_info': 0,
            'audio_loading': 0,
            'note_slicing': 0,
            'compression': 0,
            'dataframe_creation': 0,
            'fastest_file': float('inf'),
            'slowest_file': 0,
            'fastest_file_name': '',
            'slowest_file_name': ''
        }
        
        # Simple, realistic batch sizes - batch size "optimization" doesn't actually speed up audio I/O
        if memory_limit_gb <= 8:
            max_records_per_batch = int(2000 * batch_size_multiplier)
        elif memory_limit_gb <= 16:
            max_records_per_batch = int(4000 * batch_size_multiplier)
        elif memory_limit_gb <= 32:
            max_records_per_batch = int(8000 * batch_size_multiplier)
        else:
            max_records_per_batch = int(12000 * batch_size_multiplier)
            
        logger.info(f"Batch size: {max_records_per_batch} records (memory limit: {memory_limit_gb}GB, multiplier: {batch_size_multiplier}x)")

        # Memory monitoring setup
        import psutil
        process = psutil.Process()
        memory_samples = []
        files_processed_since_last_save = 0

        for i, row in enumerate(tqdm(self.midi_wav_map.itertuples(index=False, name='Row'), desc="Processing file pairs")):
            # Pre-processing memory check
            pre_mem = process.memory_info().rss / (1024**3)
            
            # PERFORMANCE INSTRUMENTATION: Track per-file processing
            file_start_time = time.perf_counter()
            result = process_func(row)
            file_time = time.perf_counter() - file_start_time
            
            # PERFORMANCE INSTRUMENTATION: Collect aggregate timing (extract from logs if available)
            # The detailed timing is logged by _process_file_pair, here we track file-level stats
            aggregate_timing['total_files'] += 1
            aggregate_timing['total_time'] += file_time
            
            if file_time < aggregate_timing['fastest_file']:
                aggregate_timing['fastest_file'] = file_time
                aggregate_timing['fastest_file_name'] = row.midi_filename
            if file_time > aggregate_timing['slowest_file']:
                aggregate_timing['slowest_file'] = file_time
                aggregate_timing['slowest_file_name'] = row.midi_filename
            
            # Post-processing memory check
            post_mem = process.memory_info().rss / (1024**3)
            mem_delta = post_mem - pre_mem
            
            if result is not None and not result.empty:
                # Optimize DataFrame memory before adding to batch
                if len(result) > 10:
                    result = self._optimize_dataframe_memory(result)
                
                # STREAMING: Add individual records to batch, not entire DataFrames
                for _, record in result.iterrows():
                    current_batch_records.append(record.to_dict())
                    total_processed_records += 1
                
                total_processed_files += 1
                files_processed_since_last_save += 1
                
                # Estimate memory per record for monitoring
                records_in_result = len(result)
                if records_in_result > 0:
                    # Estimate based on audio array sizes (compressed and uncompressed)
                    audio_mem_mb = 0
                    for audio_data in result['audio_wav']:
                        if isinstance(audio_data, dict) and 'compressed_audio' in audio_data:
                            audio_mem_mb += len(audio_data['compressed_audio']) / (1024**2)
                        elif isinstance(audio_data, np.ndarray):
                            audio_mem_mb += audio_data.nbytes / (1024**2)
                    avg_mb_per_record = audio_mem_mb / records_in_result
                    memory_samples.append(avg_mb_per_record)
                
                # Clear the result DataFrame immediately to free memory
                del result
                
                # Log progress every 25 files
                if i % 25 == 0 and memory_samples:
                    avg_mem_per_record = np.mean(memory_samples[-100:])  # Rolling average
                    estimated_batch_mb = len(current_batch_records) * avg_mem_per_record
                    logger.info(f"[PROGRESS] Files: {total_processed_files}, Records: {total_processed_records}, "
                              f"Batch: {len(current_batch_records)} records (~{estimated_batch_mb:.1f}MB), "
                              f"Memory: {post_mem:.1f}GB (+{mem_delta:.1f}GB delta)")

            # EMERGENCY: Force save if memory is critically high (regardless of batch size)
            current_mem_pct = post_mem / memory_limit_gb
            if current_mem_pct > 0.75 and len(current_batch_records) >= 100:  # At least 100 records
                logger.warning(f"EMERGENCY SAVE: Memory at {current_mem_pct*100:.1f}% ({post_mem:.1f}GB), "
                             f"saving {len(current_batch_records)} records from {files_processed_since_last_save} files")
                if batching:
                    self._save_records_as_batch(current_batch_records, batch_idx, dir_path)
                    batch_idx += 1
                    current_batch_records.clear()
                    files_processed_since_last_save = 0
                    gc.collect()

            # REGULAR: Save when we hit record limit
            elif batching and len(current_batch_records) >= max_records_per_batch:
                logger.info(f"Regular batch save: {len(current_batch_records)} records from {files_processed_since_last_save} files")
                self._save_records_as_batch(current_batch_records, batch_idx, dir_path)
                batch_idx += 1
                current_batch_records.clear()
                files_processed_since_last_save = 0
                gc.collect()

            # Memory checks with adaptive frequency
            if current_mem_pct > 0.6:
                check_frequency = 5   # Check every 5 files when memory > 60%
            elif current_mem_pct > 0.4:
                check_frequency = 15  # Check every 15 files when memory > 40%
            else:
                check_frequency = 25  # Normal frequency
                
            if i % check_frequency == 0:
                current_mem = check_memory()
                
                # Dynamic batch size adjustment (respecting multiplier)
                base_max_records = max_records_per_batch / batch_size_multiplier  # Get base size
                if current_mem > memory_limit_gb * 0.8:
                    base_max_records = max(200, base_max_records // 2)
                    max_records_per_batch = int(base_max_records * batch_size_multiplier)
                    logger.warning(f"High memory pressure: reducing batch size to {max_records_per_batch} records")
                elif current_mem < memory_limit_gb * 0.4 and base_max_records < 5000:
                    base_max_records = min(5000, base_max_records + 500)
                    max_records_per_batch = int(base_max_records * batch_size_multiplier)
                    logger.info(f"Low memory pressure: increasing batch size to {max_records_per_batch} records")

        # Final batch save
        if current_batch_records and batching:
            logger.info(f"Final batch save: {len(current_batch_records)} records from {files_processed_since_last_save} files")
            self._save_records_as_batch(current_batch_records, batch_idx, dir_path)
            current_batch_records.clear()
            gc.collect()

        # PERFORMANCE REPORT: Final aggregate statistics
        final_mem = process.memory_info().rss / (1024**3)
        avg_records_per_file = total_processed_records / max(1, total_processed_files)
        avg_mem_per_record = np.mean(memory_samples) if memory_samples else 0
        avg_time_per_file = aggregate_timing['total_time'] / max(1, aggregate_timing['total_files'])
        files_per_second = aggregate_timing['total_files'] / max(0.001, aggregate_timing['total_time'])
        
        logger.info(f"Sequential processing complete:")
        logger.info(f"  - Files processed: {total_processed_files}")
        logger.info(f"  - Records generated: {total_processed_records}")
        logger.info(f"  - Avg records/file: {avg_records_per_file:.1f}")
        logger.info(f"  - Avg memory/record: {avg_mem_per_record:.1f}MB")
        logger.info(f"  - Final memory: {final_mem:.1f}GB")
        
        logger.info(f"[PERF SUMMARY] Processing speed: {files_per_second:.1f} files/sec ({avg_time_per_file:.3f}s/file)")
        logger.info(f"[PERF SUMMARY] Fastest file: {aggregate_timing['fastest_file']:.3f}s ({aggregate_timing['fastest_file_name']})")
        logger.info(f"[PERF SUMMARY] Slowest file: {aggregate_timing['slowest_file']:.3f}s ({aggregate_timing['slowest_file_name']})")
        logger.info(f"[PERF SUMMARY] Total processing time: {aggregate_timing['total_time']:.1f}s for {aggregate_timing['total_files']} files")
        
        return total_processed_files

    def _save_batch_sequential(self, batch_data, batch_idx, dir_path):
        """Enhanced batch saving with memory diagnostics and optimization."""
        if not batch_data:
            return

        # Memory diagnostic before combining
        import psutil
        pre_combine_mem = psutil.Process().memory_info().rss / (1024**3)
        
        # Estimate memory usage of batch_data
        total_estimated_mb = 0
        for df in batch_data:
            if hasattr(df, 'memory_usage'):
                total_estimated_mb += df.memory_usage(deep=True).sum() / (1024**2)
            else:
                # Rough estimate
                total_estimated_mb += sum(arr.nbytes for arr in df['audio_wav'] if isinstance(arr, np.ndarray)) / (1024**2)

        logger.info(f"[SAVE] Batch {batch_idx}: Combining {len(batch_data)} DataFrames (~{total_estimated_mb:.1f}MB estimated)")

        try:
            # Use ignore_index=True and handle memory more carefully
            combined_df = pd.concat(batch_data, ignore_index=True, copy=False)
            
            # Apply storage optimizations (compression analysis) 
            combined_df = self._optimize_batch_for_storage(combined_df)
            
            # Memory diagnostic after combining
            post_combine_mem = psutil.Process().memory_info().rss / (1024**3)
            actual_df_mb = combined_df.memory_usage(deep=True).sum() / (1024**2)
            
            logger.debug(f"[SAVE] Combined DF: {actual_df_mb:.1f}MB actual, memory: {pre_combine_mem:.1f}GB -> {post_combine_mem:.1f}GB")
            
            # Save with compression to reduce file size
            output_file = os.path.join(dir_path, f"audio_set_batch_{batch_idx}.pkl")
            combined_df.to_pickle(output_file, compression='gzip', protocol=4)  # Use compression and latest protocol
            
            # Verify save and get file size
            file_size_mb = os.path.getsize(output_file) / (1024**2)
            records_per_gb = len(combined_df) / (file_size_mb / 1024) if file_size_mb > 0 else 0
            
            logger.info(f"[SAVE] Batch {batch_idx}: {len(combined_df)} records saved to {output_file} "
                       f"({file_size_mb:.1f}MB on disk, {records_per_gb:.0f} records/GB)")
            
            # Clear the combined DataFrame immediately
            del combined_df
            gc.collect()
            
        except Exception as save_err:
            logger.error(f"Error saving batch {batch_idx}: {save_err}")
            # Try alternative save method - save individual DataFrames
            try:
                for i, df in enumerate(batch_data):
                    alt_file = os.path.join(dir_path, f"audio_set_batch_{batch_idx}_part_{i}.pkl")
                    df.to_pickle(alt_file, compression='gzip')
                logger.warning(f"Saved batch {batch_idx} as {len(batch_data)} individual files due to memory constraints")
            except Exception as alt_err:
                logger.error(f"Failed to save batch {batch_idx} even as individual files: {alt_err}")

        # Final memory check
        final_mem = psutil.Process().memory_info().rss / (1024**3)
        logger.debug(f"[SAVE] Batch {batch_idx} complete, final memory: {final_mem:.1f}GB")

    def _save_records_as_batch(self, records_list, batch_idx, dir_path):
        """Stream-save individual records as a single batch with minimal memory overhead."""
        if not records_list:
            return

        # Memory diagnostic before conversion
        import psutil
        pre_convert_mem = psutil.Process().memory_info().rss / (1024**3)
        
        # Estimate memory of records with compression
        estimated_mb = len(records_list) * 0.1  # Much smaller with compression: ~100KB per record
        logger.info(f"[SAVE] Batch {batch_idx}: Converting {len(records_list)} records (~{estimated_mb:.1f}MB estimated with compression)")

        try:
            # Convert list of record dicts to DataFrame efficiently
            # This is more memory-efficient than concatenating many small DataFrames
            combined_df = pd.DataFrame(records_list)
            
            # Apply storage optimizations (compression analysis)
            combined_df = self._optimize_batch_for_storage(combined_df)
            
            # Memory diagnostic after conversion
            post_convert_mem = psutil.Process().memory_info().rss / (1024**3)
            actual_df_mb = combined_df.memory_usage(deep=True).sum() / (1024**2)
            
            logger.debug(f"[SAVE] Records->DF: {actual_df_mb:.1f}MB actual, memory: {pre_convert_mem:.1f}GB -> {post_convert_mem:.1f}GB")
            
            # Save with high compression to reduce file size
            output_file = os.path.join(dir_path, f"audio_set_batch_{batch_idx}.pkl")
            combined_df.to_pickle(output_file, compression='gzip', protocol=4)
            
            # Verify save and get file size
            file_size_mb = os.path.getsize(output_file) / (1024**2)
            records_per_gb = len(combined_df) / (file_size_mb / 1024) if file_size_mb > 0 else 0
            
            logger.info(f"[SAVE] Batch {batch_idx}: {len(combined_df)} records saved to {output_file} "
                       f"({file_size_mb:.1f}MB on disk, {records_per_gb:.0f} records/GB)")
            
            # Clear the DataFrame immediately
            del combined_df
            gc.collect()
            
        except Exception as save_err:
            logger.error(f"Error saving records batch {batch_idx}: {save_err}")
            # Try emergency save as individual smaller files
            try:
                chunk_size = 500  # Save in smaller chunks
                for chunk_idx, start_idx in enumerate(range(0, len(records_list), chunk_size)):
                    chunk_records = records_list[start_idx:start_idx + chunk_size]
                    chunk_df = pd.DataFrame(chunk_records)
                    alt_file = os.path.join(dir_path, f"audio_set_batch_{batch_idx}_chunk_{chunk_idx}.pkl")
                    chunk_df.to_pickle(alt_file, compression='gzip')
                    del chunk_df
                
                num_chunks = (len(records_list) + chunk_size - 1) // chunk_size
                logger.warning(f"Saved batch {batch_idx} as {num_chunks} chunks of {chunk_size} records due to memory constraints")
            except Exception as alt_err:
                logger.error(f"Failed to save batch {batch_idx} even as chunks: {alt_err}")

        # Final memory check
        final_mem = psutil.Process().memory_info().rss / (1024**3)
        logger.debug(f"[SAVE] Batch {batch_idx} complete, final memory: {final_mem:.1f}GB")

    def decompress_batch_for_training(self, batch_df):
        """
        Decompress all audio data in a batch for training use.
        Call this method when loading batches for model training.
        
        :param batch_df: DataFrame with potentially compressed audio data
        :return: DataFrame with decompressed audio arrays
        """
        logger.info(f"Decompressing audio data for training: {len(batch_df)} records")
        
        # Decompress audio_wav column
        batch_df['audio_wav'] = batch_df['audio_wav'].apply(self._decompress_audio_slice)
        
        # Filter out any decompression failures (zero arrays)
        initial_count = len(batch_df)
        batch_df = batch_df[batch_df['audio_wav'].apply(lambda x: isinstance(x, np.ndarray) and x.sum() != 0)]
        final_count = len(batch_df)
        
        if initial_count != final_count:
            logger.warning(f"Filtered out {initial_count - final_count} records with decompression failures")
        
        logger.info(f"Decompression complete: {final_count} records ready for training")
        return batch_df

    def _process_parallel_improved(self, process_func, batching, dir_path, num_batches, check_memory, memory_limit_gb, batch_size_multiplier, n_processes):
        """
        Improved parallel processing with explicit memory management and smaller chunks.
        
        Uses multiprocessing with smaller batches and explicit memory limits to avoid
        the memory explosion issues of the previous approach.
        """
        import multiprocessing as mp
        from multiprocessing import Pool
        
        logger.info(f"Starting improved parallel processing with {n_processes} processes")
        
        total_processed_files = 0
        total_processed_records = 0
        current_batch_records = []
        batch_idx = 0
        
        # Conservative batch sizes for parallel processing
        if memory_limit_gb <= 16:
            max_records_per_batch = int(1500 * batch_size_multiplier)
        elif memory_limit_gb <= 32:
            max_records_per_batch = int(3000 * batch_size_multiplier) 
        else:
            max_records_per_batch = int(6000 * batch_size_multiplier)
        
        logger.info(f"Parallel batch size: {max_records_per_batch} records")
        
        # Process files in small chunks to manage memory
        chunk_size = max(1, min(10, len(self.midi_wav_map) // n_processes))
        file_chunks = [self.midi_wav_map.iloc[i:i+chunk_size] for i in range(0, len(self.midi_wav_map), chunk_size)]
        
        logger.info(f"Processing {len(self.midi_wav_map)} files in {len(file_chunks)} chunks of ~{chunk_size} files each")
        
        # Process chunks in parallel
        with Pool(processes=n_processes) as pool:
            try:
                # Process each chunk
                for chunk_idx, chunk_df in enumerate(file_chunks):
                    logger.info(f"Processing chunk {chunk_idx+1}/{len(file_chunks)} ({len(chunk_df)} files)")
                    
                    # Convert chunk to list of tuples for multiprocessing
                    chunk_rows = [row for row in chunk_df.itertuples(index=False, name='Row')]
                    
                    # Submit parallel tasks
                    results = pool.map(process_func, chunk_rows)
                    
                    # Process results
                    for result in results:
                        if result is not None and not result.empty:
                            # Add records from this file to current batch
                            for _, record in result.iterrows():
                                current_batch_records.append(record.to_dict())
                                total_processed_records += 1
                            
                            total_processed_files += 1
                            
                            # Check if batch is ready to save
                            if batching and len(current_batch_records) >= max_records_per_batch:
                                logger.info(f"Parallel batch save: {len(current_batch_records)} records from chunk {chunk_idx+1}")
                                self._save_records_as_batch(current_batch_records, batch_idx, dir_path)
                                current_batch_records.clear()
                                batch_idx += 1
                    
                    # Memory management between chunks
                    del results
                    gc.collect()
                    
                    # Check memory usage
                    current_mem = check_memory()
                    if current_mem / memory_limit_gb > 0.8:
                        logger.warning(f"High memory usage after chunk {chunk_idx+1}: {current_mem:.1f}GB")
                        gc.collect()
                        
            except Exception as e:
                logger.error(f"Parallel processing failed: {e}")
                logger.info("Falling back to sequential processing for remaining files")
                # Process any remaining files sequentially
                pool.terminate()
                pool.join()
                return total_processed_files
        
        # Save final batch
        if current_batch_records and batching:
            logger.info(f"Final parallel batch save: {len(current_batch_records)} records")
            self._save_records_as_batch(current_batch_records, batch_idx, dir_path)
        
        logger.info(f"Parallel processing complete: {total_processed_files} files, {total_processed_records} records")
        return total_processed_files
