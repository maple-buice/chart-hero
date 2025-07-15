"""
Refactored data preparation script for transformer-based drum transcription.

This script replaces the complex, multi-stage pipeline of the original data_preparation.py 
and label_data.py. It uses a PyTorch Dataset to process raw E-GMD data directly 
into spectrograms and labels, leveraging PyTorch's DataLoader for efficient, 
parallel data loading.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import pandas as pd
import librosa
import mido
import os
from pathlib import Path
import logging
from tqdm import tqdm
import argparse
import itertools
import math
import soundfile as sf

# Add project root to Python path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_training.transformer_data import SpectrogramProcessor
from model_training.transformer_config import get_config, BaseConfig
from model_training.transformer_config import DRUM_HIT_MAP, TARGET_CLASSES, DRUM_HIT_TO_INDEX

logger = logging.getLogger(__name__)

class EGMDRawDataset(Dataset):
    """
    PyTorch Dataset for the E-GMD dataset.
    Processes raw audio/MIDI pairs into spectrograms and labels on the fly.
    """
    def __init__(self, data_map: pd.DataFrame, dataset_dir: Path, config: BaseConfig):
        self.data_map = data_map
        self.dataset_dir = dataset_dir
        self.config = config
        self.processor = SpectrogramProcessor(config)
        
        self.all_notes = self._create_note_map()

    def _create_note_map(self):
        """
        Pre-processes all MIDI files to create a flat list of all notes in the dataset.
        Each item in the list will be a tuple: (track_id, note_info, audio_filename)
        """
        all_notes = []
        for index, row in tqdm(self.data_map.iterrows(), total=len(self.data_map), desc="Processing MIDI files"):
            midi_filename = self.dataset_dir / row['midi_filename']
            try:
                notes = self._extract_notes_from_midi(midi_filename)
                for note in notes:
                    track_id = row.get('track_id', index)
                    all_notes.append({
                        "track_id": track_id,
                        "note": note,
                        "audio_filename": self.dataset_dir / row['audio_filename']
                    })
            except Exception as e:
                logger.warning(f"Could not process MIDI file {midi_filename}: {e}")
        logger.info(f"Created a total of {len(all_notes)} notes from {len(self.data_map)} files.")
        return all_notes

    def _extract_notes_from_midi(self, midi_path: Path):
        """Extracts note events from a MIDI file."""
        midi_file = mido.MidiFile(midi_path)
        ticks_per_beat = midi_file.ticks_per_beat
        tempo = 500000  # Default tempo

        for msg in midi_file.tracks[0]:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
        
        notes = []
        time_log = 0
        temp_dict = {}

        for msg in midi_file.tracks[-1]:
            time_log += msg.time
            
            if msg.type == 'note_on' and msg.velocity > 0:
                temp_dict[msg.note] = time_log
            elif (msg.type == 'note_on' and msg.velocity == 0) or msg.type == 'note_off':
                if msg.note in temp_dict:
                    start_tick = temp_dict.pop(msg.note)
                    end_tick = time_log
                    start_time = mido.tick2second(start_tick, ticks_per_beat, tempo)
                    end_time = mido.tick2second(end_tick, ticks_per_beat, tempo)
                    notes.append({'note': msg.note, 'start': start_time, 'end': end_time})
        
        # Merge notes with the same start time
        merged_notes = []
        key_func = lambda x: x['start']
        sorted_notes = sorted(notes, key=key_func)

        for key, group in itertools.groupby(sorted_notes, key_func):
            group_list = list(group)
            if len(group_list) > 1:
                midi_notes = [item['note'] for item in group_list]
                start_time = key
                end_time = max(item['end'] for item in group_list)
                merged_notes.append({'label': midi_notes, 'start': start_time, 'end': end_time})
            else:
                item = group_list[0]
                merged_notes.append({'label': [item['note']], 'start': item['start'], 'end': item['end']})

        return merged_notes

    def __len__(self):
        return len(self.all_notes)

    def __getitem__(self, idx: int):
        note_item = self.all_notes[idx]
        audio_filename = note_item['audio_filename']
        note_info = note_item['note']
        
        start_time = note_info['start']
        end_time = note_info['end']

        # Load audio segment
        try:
            # Define segment length based on config
            segment_length_sec = self.config.max_audio_length
            # Center the segment around the note
            segment_start_sec = start_time - (segment_length_sec / 2) + ((end_time - start_time) / 2)
            segment_start_sec = max(0, segment_start_sec)

            frame_offset = int(segment_start_sec * self.config.sample_rate)
            num_frames = int(segment_length_sec * self.config.sample_rate)
            
            audio, sr = torchaudio.load(audio_filename, frame_offset=frame_offset, num_frames=num_frames)

            if sr != self.config.sample_rate:
                audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.config.sample_rate)

        except Exception as e:
            logger.warning(f"Could not load audio file {audio_filename}: {e}")
            return None # Dataloader worker will ignore None types

        # Process audio to spectrogram
        spectrogram = self.processor.audio_to_spectrogram(audio)
        
        # Encode label
        label_vector = torch.zeros(len(TARGET_CLASSES), dtype=torch.float32)
        for midi_note in note_info['label']:
            target_hit = DRUM_HIT_MAP.get(midi_note)
            if target_hit:
                target_index = DRUM_HIT_TO_INDEX.get(target_hit)
                if target_index is not None:
                    label_vector[target_index] = 1
        
        return spectrogram, label_vector

def collate_fn(batch):
    """Custom collate function to filter out None values."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)

class data_preparation:
    def __init__(self, directory_path, dataset, sample_ratio=1.0, diff_threshold=1.0, **kwargs):
        self.directory_path = Path(directory_path)
        self.dataset_type = dataset
        self.sample_ratio = sample_ratio
        self.diff_threshold = diff_threshold
        
        csv_path = next(self.directory_path.glob('*.csv'))
        self.full_df = pd.read_csv(csv_path).dropna()
        
        # Filter by duration difference
        logger.info("Calculating audio durations...")
        durations = [sf.info(self.directory_path / f).duration for f in tqdm(self.full_df['audio_filename'])]
        self.full_df['wav_length'] = durations
        self.full_df['diff'] = np.abs(self.full_df['duration'] - self.full_df['wav_length'])
        self.midi_wav_map = self.full_df[self.full_df['diff'] <= self.diff_threshold].copy()
        
        logger.info(f"Initialized with {len(self.midi_wav_map)} files after filtering.")

    def create_audio_set(self, dir_path, num_batches=None, **kwargs):
        output_dir = Path(dir_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        config = get_config('local') # Use a default config
        
        sampled_df = self.midi_wav_map.sample(frac=self.sample_ratio).reset_index()
        
        # Split data
        train_df, val_df, test_df = np.split(sampled_df.sample(frac=1, random_state=42), 
                                             [int(.8*len(sampled_df)), int(.9*len(sampled_df))])

        for mode, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            logger.info(f"Processing {mode} set with {len(df)} files...")
            dataset = EGMDRawDataset(df, self.directory_path, config)
            
            if len(dataset) == 0:
                logger.warning(f"No notes found for {mode} set. Skipping.")
                continue
                
            data_loader = DataLoader(
                dataset,
                batch_size=kwargs.get('batch_size', 256),
                shuffle=False,
                num_workers=kwargs.get('num_workers', 4),
                collate_fn=collate_fn,
                pin_memory=True
            )

            # Process and save data
            all_spectrograms = []
            all_labels = []
            for i, (spectrograms, labels) in enumerate(tqdm(data_loader, desc=f"Generating {mode} data")):
                if spectrograms is not None:
                    all_spectrograms.append(spectrograms.numpy())
                    all_labels.append(labels.numpy())
                
                # Save in batches
                if num_batches and (i + 1) % (len(data_loader) // num_batches) == 0:
                    if all_spectrograms:
                        batch_num = (i + 1) // (len(data_loader) // num_batches)
                        self._save_batch(all_spectrograms, all_labels, mode, batch_num, output_dir)
                        all_spectrograms, all_labels = [], []

            # Save any remaining data
            if all_spectrograms:
                batch_num = (num_batches or 0) + 1
                self._save_batch(all_spectrograms, all_labels, mode, batch_num, output_dir)

        logger.info("Data preparation complete.")
        return len(sampled_df)

    def _save_batch(self, spectrograms, labels, mode, batch_num, output_dir):
        final_spectrograms = np.concatenate(spectrograms)
        final_labels = np.concatenate(labels)
        
        if final_spectrograms.ndim == 3:
            final_spectrograms = final_spectrograms[:, np.newaxis, :, :]

        logger.info(f"Saving {mode} batch {batch_num}. Spectrogram shape: {final_spectrograms.shape}, Labels shape: {final_labels.shape}")
        
        # Match the old file naming convention for compatibility
        # e.g., 0_train_mel.npy
        np.save(output_dir / f"{batch_num}_{mode}_mel.npy", final_spectrograms)
        np.save(output_dir / f"{batch_num}_{mode}_label.npy", final_labels)