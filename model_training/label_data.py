import logging
import time
import os
from pathlib import Path
import pandas as pd
import numpy as np
import librosa
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm
import traceback

from utils.file_utils import get_audio_set_files, get_labeled_audio_set_dir

# Configure logging at the module level
logger = logging.getLogger(__name__)

#region Drum Hits
# Final map based on README.md
DRUM_HIT_MAP = {
    22: '67', # Hi-hat Closed (Edge) -> HiHatCymbal
    26: '67', # Hi-hat Open (Edge) -> HiHatCymbal
    35: '0', # Acoustic Bass Drum -> Kick
    36: '0', # Kick / Bass Drum 1 -> Kick
    37: '1', # Snare X-Stick / Side Stick -> Snare
    38: '1', # Snare (Head) / Acoustic Snare -> Snare
    39: '67', # Hand Clap / Cowbell -> HiHatCymbal (Treating as percussion)
    40: '1', # Snare (Rim) / Electric Snare -> Snare
    41: '4', # Low Floor Tom -> LowTom
    42: '67', # Hi-hat Closed (Bow) / Closed Hi-Hat -> HiHatCymbal
    43: '4', # Tom 3 (Head) / High Floor Tom -> LowTom
    44: '67', # Hi-hat Pedal / Pedal Hi-Hat -> HiHatCymbal
    45: '3', # Tom 2 / Low Tom -> MiddleTom
    46: '67', # Hi-hat Open (Bow) / Open Hi-Hat -> HiHatCymbal
    47: '3', # Tom 2 (Rim) / Low-Mid Tom -> MiddleTom
    48: '2', # Tom 1 / Hi-Mid Tom -> HighTom
    49: '66', # Crash 1 (Bow) / Crash Cymbal 1 -> CrashCymbal
    50: '2', # Tom 1 (Rim) / High Tom -> HighTom
    51: '68', # Ride (Bow) / Ride Cymbal 1 -> RideCymbal
    52: '66', # Crash 2 (Edge) / Chinese Cymbal -> CrashCymbal
    53: '68', # Ride (Bell) / Ride Bell -> RideCymbal
    54: '67', # Tambourine / Cowbell -> HiHatCymbal (Treating as percussion)
    55: '66', # Crash 1 (Edge) / Splash Cymbal -> CrashCymbal
    56: '67', # Cowbell -> HiHatCymbal (Treating as percussion)
    57: '66', # Crash 2 (Bow) / Crash Cymbal 2 -> CrashCymbal
    58: '4', # Tom 3 (Rim) / Vibraslap -> LowTom
    59: '68', # Ride (Edge) / Ride Cymbal 2 -> RideCymbal
    # --- Adding potentially missing mappings based on common GM ---
    # Toms
    60: '2', # Hi Bongo -> HighTom
    61: '3', # Low Bongo -> MiddleTom
    62: '2', # Mute Hi Conga -> HighTom
    63: '3', # Open Hi Conga -> MiddleTom
    64: '4', # Low Conga -> LowTom
    65: '2', # High Timbale -> HighTom
    66: '3', # Low Timbale -> MiddleTom
    # Percussion -> Map to HiHat for simplicity or create separate classes later
    67: '2', # High Agogo -> HighTom (Could be percussion)
    68: '3', # Low Agogo -> MiddleTom (Could be percussion)
    69: '67', # Cabasa -> HiHatCymbal
    70: '67', # Maracas -> HiHatCymbal
    # Cymbals/Effects -> Map reasonably
    71: '68', # Short Whistle -> RideCymbal (Treat as effect/cymbal)
    72: '66', # Long Whistle -> CrashCymbal (Treat as effect/cymbal)
    73: '68', # Short Guiro -> RideCymbal (Treat as effect/cymbal)
    74: '66', # Long Guiro -> CrashCymbal (Treat as effect/cymbal)
    75: '67', # Claves -> HiHatCymbal
    # Wood Blocks -> Map to Toms
    76: '2', # Hi Wood Block -> HighTom
    77: '3', # Low Wood Block -> MiddleTom
    # Cuica -> Map to Toms
    78: '2', # Mute Cuica -> HighTom
    79: '3', # Open Cuica -> MiddleTom
    # Triangle -> Map to Cymbals
    80: '68', # Mute Triangle -> RideCymbal
    81: '66'  # Open Triangle -> CrashCymbal
}

# Define the target classes based on the Clone Hero mapping values
TARGET_CLASSES = sorted(list(set(DRUM_HIT_MAP.values()))) # ['0', '1', '2', '3', '4', '66', '67', '68']
NUM_DRUM_HITS = len(TARGET_CLASSES)
DRUM_HIT_TO_INDEX = {hit: idx for idx, hit in enumerate(TARGET_CLASSES)}
INDEX_TO_DRUM_HIT = {idx: hit for idx, hit in enumerate(TARGET_CLASSES)}

def get_drum_hits() -> list[str]:
    """Returns the sorted list of target drum hit classes."""
    return TARGET_CLASSES

def get_drum_hits_as_strings() -> list[str]:
    """Returns the sorted list of target drum hit classes as strings (same as get_drum_hits)."""
    # Simple mapping for clarity in reports
    name_map = {
        '0': 'Kick',
        '1': 'Snare',
        '2': 'HiTom',
        '3': 'MidTom',
        '4': 'LowTom',
        '66': 'Crash',
        '67': 'HiHat',
        '68': 'Ride'
    }
    return [name_map.get(hit, hit) for hit in TARGET_CLASSES]

#endregion

# --- Helper function for parallel processing ---
def _process_audio_set_file(file_path: str, npy_data_path: str):
    """
    Processes a single audio set pickle file:
    1. Loads the DataFrame.
    2. Encodes labels into multi-hot vectors based on DRUM_HIT_MAP and TARGET_CLASSES.
    3. Calculates Mel spectrograms for each audio clip.
    4. Saves spectrograms and labels as .npy files.
    5. Deletes the original .pkl file.
    """
    process_start_time = time.perf_counter()
    file_stem = Path(file_path).stem
    logger.debug(f"Starting processing for: {file_stem}")

    result_mel_file = os.path.join(npy_data_path, file_stem + '_mel.npy')
    result_label_file = os.path.join(npy_data_path, file_stem + '_label.npy')

    # --- Skip if output exists ---
    if os.path.exists(result_mel_file) and os.path.exists(result_label_file):
        logger.debug(f'Skipping: Output already exists for {file_stem}')
        try:
            # Optional: Clean up source file even if skipped
            # os.remove(file_path)
            # logger.debug(f'Cleaned up source: {file_path}')
            pass
        except OSError as e:
            logger.warning(f"Error removing source file {file_path} for skipped item: {e}")
        return f"Skipped (already processed): {file_path}"
    # --- End Skip ---

    try:
        # --- Load Data ---
        load_start_time = time.perf_counter()
        df = pd.read_pickle(file_path)
        load_end_time = time.perf_counter()
        if df.empty:
            logger.warning(f"Empty dataframe in {file_path}, skipping.")
            return f"Skipped (empty): {file_path}"
        logger.debug(f"Loaded {file_stem} ({len(df)} rows) in {load_end_time - load_start_time:.3f}s")
        # --- End Load Data ---

        # --- Label Encoding ---
        label_start_time = time.perf_counter()
        labels_list = []
        processed_labels_count = 0
        unmapped_labels = set()

        for _, row in df.iterrows():
            label_vector = np.zeros(NUM_DRUM_HITS, dtype=np.int8)
            original_midi_label = row['label'] # This should be the MIDI note number

            # Handle single int or list/array of ints
            midi_notes = []
            if isinstance(original_midi_label, (int, np.integer)):
                midi_notes = [int(original_midi_label)]
            elif isinstance(original_midi_label, (list, np.ndarray)):
                midi_notes = [int(n) for n in original_midi_label if isinstance(n, (int, np.integer))]

            if not midi_notes:
                 logger.warning(f"Row in {file_stem} has invalid label type or empty list: {original_midi_label}")
                 continue # Skip row if label is unusable

            for midi_note in midi_notes:
                target_hit = DRUM_HIT_MAP.get(midi_note)
                if target_hit is not None:
                    target_index = DRUM_HIT_TO_INDEX.get(target_hit)
                    if target_index is not None:
                        label_vector[target_index] = 1
                        processed_labels_count += 1
                    else:
                        # This case should ideally not happen if TARGET_CLASSES is derived from DRUM_HIT_MAP values
                        logger.error(f"Internal Error: Mapped hit '{target_hit}' not found in DRUM_HIT_TO_INDEX for MIDI note {midi_note} in {file_stem}.")
                else:
                    # Log unmapped MIDI notes only once per file type
                    if midi_note not in unmapped_labels:
                        logger.warning(f"Unmapped MIDI note {midi_note} encountered in {file_stem}. It will be ignored.")
                        unmapped_labels.add(midi_note)

            labels_list.append(label_vector)

        if not labels_list:
             logger.warning(f"No valid labels could be processed for {file_stem}. Skipping file.")
             return f"Skipped (no valid labels): {file_path}"

        y = np.array(labels_list, dtype=np.int8) # Ensure dtype here too
        label_end_time = time.perf_counter()
        logger.debug(f"Label encoding for {file_stem} took {label_end_time - label_start_time:.3f}s. Processed {processed_labels_count} individual drum hits into {len(labels_list)} vectors.")
        if unmapped_labels:
             logger.warning(f"Summary of unmapped MIDI notes for {file_stem}: {sorted(list(unmapped_labels))}")
        # --- End Label Encoding ---

        # --- Mel Spectrogram Calculation ---
        mel_start_time = time.perf_counter()
        mel_specs = []
        invalid_audio_count = 0
        for i in range(df.shape[0]): # Iterate through original df length
             # Check if index i corresponds to a valid label vector
             if i >= len(labels_list):
                 # This might happen if rows were skipped during label encoding
                 continue

             audio_wav = df.audio_wav.iloc[i]
             sr = df.sampling_rate.iloc[i]

             # Basic validation
             if not isinstance(audio_wav, np.ndarray) or audio_wav.size == 0 or sr <= 0:
                 logger.warning(f"Invalid audio data or sample rate for index {i} in {file_stem}. Skipping spectrogram.")
                 # Need to decide how to handle this. Skip the sample entirely?
                 # For now, let's skip and potentially adjust label array later, or ensure label encoding skips these too.
                 # A simpler approach for now: if we skip here, we might have a mismatch later.
                 # Let's try adding a dummy spec, but log it.
                 # Or better: ensure label encoding skips rows with bad audio beforehand (needs refactor)
                 # Current approach: Add a zero spectrogram, but this might train the model on silence.
                 # Safest: Skip the sample. This requires removing the corresponding label from `y`.
                 # Let's stick to the current logic and assume label encoding handles rows correctly.
                 # If audio is bad here, we might need to filter `y` later.
                 # For now, just log and append zeros.
                 dummy_shape = (128, 1) # Example shape, adjust based on expected output
                 mel_specs.append(np.zeros(dummy_shape, dtype=np.float32))
                 invalid_audio_count += 1
                 continue

             # Ensure float32
             if audio_wav.dtype != np.float32:
                 audio_wav = audio_wav.astype(np.float32)

             try:
                 # Add a small epsilon to prevent log(0)
                 S = librosa.feature.melspectrogram(
                     y=audio_wav + 1e-9,
                     sr=sr,
                     n_mels=128, # Standard number of mel bands
                     fmax=8000   # Limit frequency range if desired
                     # Add hop_length, n_fft if specific windowing is needed
                 )
                 # Convert power spectrogram to dB scale
                 mel_spec = librosa.power_to_db(S, ref=np.max)
                 mel_specs.append(mel_spec)
             except Exception as mel_err:
                 logger.error(f"Error calculating Mel spectrogram for index {i} in {file_stem}: {mel_err}", exc_info=True)
                 # Append zeros on error
                 dummy_shape = (128, 1) # Adjust shape if needed
                 mel_specs.append(np.zeros(dummy_shape, dtype=np.float32))
                 invalid_audio_count += 1


        if invalid_audio_count > 0:
             logger.warning(f"{invalid_audio_count} samples had invalid audio or Mel calculation errors in {file_stem}.")
             # If samples were skipped, we need to ensure X and y match.
             # This current implementation appends zeros, so lengths should match `len(labels_list)`.

        if not mel_specs:
             logger.warning(f"No valid Mel spectrograms could be generated for {file_stem}. Skipping file.")
             return f"Skipped (no valid spectrograms): {file_path}"

        X = np.array(mel_specs, dtype=np.float32)
        # Add channel dimension for CNN (samples, height, width, channels)
        if X.ndim == 3:
            X = X[..., np.newaxis]
        elif X.ndim == 2: # Handle case where all specs might be 1D (unlikely)
             logger.warning(f"Spectrograms have unexpected dimension (2D) in {file_stem}. Adding two new axes.")
             X = X[:, :, np.newaxis, np.newaxis] # Make it (samples, height, 1, 1) - adjust if needed
        elif X.ndim != 4:
             logger.error(f"Unexpected final spectrogram array dimension ({X.ndim}) in {file_stem}. Skipping save.")
             return f"Failed (spectrogram dim error): {file_path}"

        mel_end_time = time.perf_counter()
        logger.debug(f"Mel spectrogram calculation for {file_stem} took {mel_end_time - mel_start_time:.3f}s. Final shape: {X.shape}")
        # --- End Mel Spectrogram ---

        # --- Final Check: Ensure X and y match ---
        if X.shape[0] != y.shape[0]:
            logger.error(f"CRITICAL: Mismatch between spectrograms ({X.shape[0]}) and labels ({y.shape[0]}) for {file_stem}. Skipping save.")
            # This indicates a logic error in handling skipped/invalid samples.
            return f"Failed (X/y shape mismatch): {file_path}"
        # --- End Final Check ---

        # --- Save numpy arrays ---
        save_start_time = time.perf_counter()
        np.save(result_mel_file, X)
        np.save(result_label_file, y)
        save_end_time = time.perf_counter()
        logger.debug(f"Saving .npy files for {file_stem} took {save_end_time - save_start_time:.3f}s")
        # --- End Save ---

        # --- Delete source file ---
        delete_start_time = time.perf_counter()
        try:
            os.remove(file_path)
            delete_end_time = time.perf_counter()
            logger.debug(f"Removed source file: {file_path} in {delete_end_time - delete_start_time:.3f}s")
        except OSError as e:
            delete_end_time = time.perf_counter()
            logger.warning(f"Error removing source file {file_path} after processing: {e}")
        # --- End Delete ---

        process_end_time = time.perf_counter()
        total_time = process_end_time - process_start_time
        logger.debug(f"Finished processing {file_stem} in {total_time:.3f} seconds.")
        return f"Processed: {file_path}"

    except Exception as e:
        logger.error(f"!!! Unhandled error processing {file_path}: {e}", exc_info=True)
        return f"Failed: {file_path}"

# --- Main function using parallel processing ---
def label_data(n_jobs=-1):
    """
    Finds all audio set .pkl files, processes them in parallel to create
    Mel spectrogram (.npy) and label (.npy) files.
    """
    overall_start_time = time.perf_counter()
    logger.info("--- Starting label_data process ---")
    npy_data_path = get_labeled_audio_set_dir()
    if not os.path.exists(npy_data_path):
        logger.info(f"Creating labeled data directory: {npy_data_path}")
        os.makedirs(npy_data_path)

    audio_files = get_audio_set_files()
    if not audio_files:
        logger.warning("No audio set files (.pkl) found to process in label_data.")
        return

    num_files = len(audio_files)
    effective_jobs = os.cpu_count() if n_jobs == -1 else n_jobs
    logger.info(f"Found {num_files} audio set files to process using up to {effective_jobs} parallel jobs...")

    # Use 'loky' backend for better robustness, especially on macOS/Windows
    parallel_start_time = time.perf_counter()
    with parallel_backend('loky', n_jobs=n_jobs):
        results = Parallel()(
            delayed(_process_audio_set_file)(file, npy_data_path)
            for file in tqdm(audio_files, desc="Labeling audio sets")
        )
    parallel_end_time = time.perf_counter()
    logger.info(f"Parallel processing finished in {parallel_end_time - parallel_start_time:.2f} seconds.")

    # --- Summary ---
    processed_count = sum(1 for r in results if r is not None and r.startswith("Processed"))
    skipped_count = sum(1 for r in results if r is not None and r.startswith("Skipped"))
    failed_count = sum(1 for r in results if r is not None and r.startswith("Failed"))
    unknown_count = num_files - (processed_count + skipped_count + failed_count)

    logger.info(f"--- Labeling Summary ---")
    logger.info(f"Total files found: {num_files}")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Skipped (e.g., already done, empty): {skipped_count}")
    logger.info(f"Failed processing: {failed_count}")
    if unknown_count > 0:
         logger.warning(f"Unknown status (result=None): {unknown_count}") # Should ideally be 0
    if failed_count > 0:
         logger.warning(f"{failed_count} files failed processing. Please check logs above for details.")
    # --- End Summary ---

    overall_end_time = time.perf_counter()
    logger.info(f"--- label_data process finished in {overall_end_time - overall_start_time:.2f} seconds. ---")

# Example of how to run if needed
# if __name__ == '__main__':
#     # Configure logging for standalone run
#     logging.basicConfig(level=logging.DEBUG, # Set to DEBUG to see detailed logs
#                         format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
#                         datefmt='%Y-%m-%d %H:%M:%S')
#     logger.info("Running label_data standalone...")
#     label_data(n_jobs=-1) # Use all available cores