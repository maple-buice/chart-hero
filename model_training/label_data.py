import os
from pathlib import Path

import pandas as pd
import numpy as np

import librosa
from joblib import Parallel, delayed
from tqdm import tqdm

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
        41, # Low Floor Tom
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

DRUM_HITS_LIST = get_drum_hits()
DRUM_HITS_SET = set(DRUM_HITS_LIST) # Use set for faster lookups if needed
DRUM_HIT_TO_INDEX = {hit: i for i, hit in enumerate(DRUM_HITS_LIST)}
NUM_DRUM_HITS = len(DRUM_HITS_LIST)

def get_drum_hits_as_strings() -> list[str]:
    drum_hits = []
    for drum_hit in get_drum_hits():
        drum_hits.append(str(drum_hit))
    return drum_hits

# --- Helper function for parallel processing ---
def _process_audio_set_file(file_path: str, npy_data_path: str):
    """Processes a single audio set pickle file to create Mel spectrogram and label .npy files."""
    result_mel_file = os.path.join(npy_data_path, Path(file_path).stem + '_mel.npy')
    result_label_file = os.path.join(npy_data_path, Path(file_path).stem + '_label.npy')

    if os.path.exists(result_mel_file) and os.path.exists(result_label_file):
        # print(f'Skipping: Output exists for {file_path}')
        try:
            os.remove(file_path) # Clean up source if output exists
            # print(f'Cleaned up source: {file_path}')
        except OSError as e:
            print(f"Error removing source file {file_path}: {e}")
        return f"Skipped (already processed): {file_path}"

    try:
        df = pd.read_pickle(file_path)
        if df.empty:
            print(f"Warning: Empty dataframe in {file_path}, skipping.")
            return f"Skipped (empty): {file_path}"

        # --- Optimized Label Encoding (Multi-Hot) ---
        labels_list = []
        for _, row in df.iterrows():
            label_vector = np.zeros(NUM_DRUM_HITS, dtype=np.int8) # Use int8 for space efficiency
            current_label = row['label']
            if isinstance(current_label, (list, np.ndarray)): # Handle multiple hits
                for hit in current_label:
                    if hit in DRUM_HIT_TO_INDEX:
                        label_vector[DRUM_HIT_TO_INDEX[hit]] = 1
            elif isinstance(current_label, (int, np.integer)): # Handle single hit
                if current_label in DRUM_HIT_TO_INDEX:
                    label_vector[DRUM_HIT_TO_INDEX[current_label]] = 1
            # Else: label might be invalid or empty, vector remains zeros
            labels_list.append(label_vector)
        y = np.array(labels_list)
        # --- End Label Encoding ---

        # --- Optimized Mel Spectrogram Calculation ---
        mel_specs = []
        for i in range(df.shape[0]):
            audio_wav = df.audio_wav.iloc[i]
            sr = df.sampling_rate.iloc[i]
            # Ensure float32 for librosa
            if audio_wav.dtype != np.float32:
                audio_wav = audio_wav.astype(np.float32)
            
            # Add a small epsilon to avoid log(0) if audio is silent
            S = librosa.feature.melspectrogram(
                y=audio_wav,
                sr=sr,
                n_mels=128,
                fmax=8000
            )
            mel_spec = librosa.power_to_db(S, ref=np.max) # Convert to dB scale
            mel_specs.append(mel_spec)

        # Create final array with float32
        X = np.array(mel_specs, dtype=np.float32) 
        # Add channel dimension
        if X.ndim == 3: 
            X = X[..., np.newaxis]
        # --- End Mel Spectrogram ---

        # Save numpy arrays
        np.save(result_mel_file, X)
        np.save(result_label_file, y)

        # Delete source file after successful processing
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error removing source file {file_path} after processing: {e}")

        return f"Processed: {file_path}"

    except Exception as e:
        print(f"!!! Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return f"Failed: {file_path}"

# --- Main function using parallel processing ---
def label_data(n_jobs=-1):
    npy_data_path = get_labeled_audio_set_dir()
    if not os.path.exists(npy_data_path):
        os.makedirs(npy_data_path)
        
    audio_files = get_audio_set_files()
    if not audio_files:
        print("No audio set files found to process.")
        return

    print(f"Found {len(audio_files)} audio set files to process...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_audio_set_file)(file, npy_data_path) 
        for file in tqdm(audio_files, desc="Labeling audio sets")
    )
    
    # Optional: Print summary of results
    processed_count = sum(1 for r in results if r.startswith("Processed"))
    skipped_count = sum(1 for r in results if r.startswith("Skipped"))
    failed_count = sum(1 for r in results if r.startswith("Failed"))
    print(f"Labeling complete. Processed: {processed_count}, Skipped: {skipped_count}, Failed: {failed_count}")

# Example of how to call it if run as a script
# if __name__ == '__main__':
#    label_data() # Use all available cores