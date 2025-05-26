"""
Data preparation and loading for transformer-based drum transcription.
Handles patch-based spectrogram processing and efficient data loading.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import pandas as pd
import librosa
import os
from typing import Tuple, Dict, List, Optional, Union
from pathlib import Path
import logging
from tqdm import tqdm
import pickle

from .transformer_config import BaseConfig
from .label_data import get_drum_hits


logger = logging.getLogger(__name__)


class SpectrogramProcessor:
    """Handles audio to spectrogram conversion with patch-based tokenization."""
    
    def __init__(self, config: BaseConfig):
        self.config = config
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            power=2.0
        )
        
    def audio_to_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio waveform to log-mel spectrogram."""
        # Ensure audio is the right length
        target_length = int(self.config.max_audio_length * self.config.sample_rate)
        
        if audio.shape[-1] > target_length:
            # Random crop for training data augmentation
            start_idx = torch.randint(0, audio.shape[-1] - target_length + 1, (1,)).item()
            audio = audio[..., start_idx:start_idx + target_length]
        elif audio.shape[-1] < target_length:
            # Pad with zeros
            padding = target_length - audio.shape[-1]
            audio = F.pad(audio, (0, padding))
        
        # Ensure exactly the target length
        audio = audio[..., :target_length]
        
        # Convert to mel spectrogram
        mel_spec = self.mel_transform(audio)
        
        # Convert to log scale
        log_mel_spec = torch.log(mel_spec + 1e-8)
        
        # Ensure consistent shape: [channels, freq, time] -> [channels, time, freq]
        # MelSpectrogram outputs [channels, n_mels, time_frames]
        # We want [channels, time_frames, n_mels] for consistency
        if log_mel_spec.shape[1] == self.config.n_mels:
            # Current shape is [channels, n_mels, time_frames], transpose to [channels, time_frames, n_mels]
            log_mel_spec = log_mel_spec.transpose(1, 2)
        
        # Normalize to [-1, 1] range
        log_mel_spec = 2 * (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min()) - 1
        
        return log_mel_spec
    
    def prepare_patches(self, spectrogram: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Prepare spectrogram for patch-based processing.
        
        Args:
            spectrogram: Log-mel spectrogram [channels, time, freq]
            
        Returns:
            Padded spectrogram and patch shape (time_patches, freq_patches)
        """
        channels, time_frames, freq_bins = spectrogram.shape
        patch_time, patch_freq = self.config.patch_size
        
        # Calculate required padding
        time_padding = (patch_time - (time_frames % patch_time)) % patch_time
        freq_padding = (patch_freq - (freq_bins % patch_freq)) % patch_freq
        
        # Pad spectrogram - padding order is (left, right, top, bottom) for last two dims
        # For tensor [channels, time, freq], we pad freq (last dim) then time (second-to-last)
        if time_padding > 0 or freq_padding > 0:
            spectrogram = F.pad(spectrogram, (0, freq_padding, 0, time_padding))
        
        # Calculate patch dimensions after padding
        final_time_frames = time_frames + time_padding
        final_freq_bins = freq_bins + freq_padding
        
        time_patches = final_time_frames // patch_time
        freq_patches = final_freq_bins // patch_freq
        
        return spectrogram, (time_patches, freq_patches)


class DrumDataset(Dataset):
    """Dataset for drum transcription with transformer-compatible formatting."""
    
    def __init__(self, 
                 data_df: pd.DataFrame,
                 config: BaseConfig,
                 audio_dir: str,
                 mode: str = 'train',
                 augment: bool = True):
        self.data_df = data_df.copy()
        self.config = config
        self.audio_dir = Path(audio_dir)
        self.mode = mode
        self.augment = augment and (mode == 'train')
        
        self.processor = SpectrogramProcessor(config)
        self.drum_mapping = self._create_drum_mapping()
        
        logger.info(f"Created {mode} dataset with {len(self.data_df)} samples")
        
    def _create_drum_mapping(self) -> Dict[int, int]:
        """Create mapping from MIDI notes to drum class indices."""
        # Based on the drum mapping from model_training/README.md
        midi_to_class = {
            0: 0,   # Kick
            1: 1,   # Snare
            2: 2,   # High Tom
            3: 3,   # Middle Tom
            4: 4,   # Low Tom
            66: 5,  # Crash Cymbal
            67: 6,  # Hi-Hat Cymbal
            68: 7,  # Ride Cymbal
        }
        return midi_to_class
    
    def __len__(self) -> int:
        return len(self.data_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data_df.iloc[idx]
        
        try:
            # Load pre-processed audio from the data_preparation step
            audio_data = row['audio_wav']
            sr = row['sampling_rate']
            
            # Convert numpy array to torch tensor
            if isinstance(audio_data, list):
                audio_data = np.array(audio_data)
            audio = torch.from_numpy(audio_data).float()
            
            # Ensure audio is 2D (channels, samples)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            # Resample if necessary
            if sr != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
                audio = resampler(audio)
            
            # Audio is already mono from data preparation, but ensure consistency
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Apply audio augmentation for training
            if self.augment:
                audio = self._augment_audio(audio)
            
            # The audio is already pre-segmented from data_preparation step
            # with padding included, so we use it directly
            audio_segment = audio
            
            # Extract timing info for reference (though audio is already segmented)
            start_time = row['start']
            end_time = row['end']
            
            # Convert to spectrogram
            spectrogram = self.processor.audio_to_spectrogram(audio_segment)
            
            # Prepare patches
            spectrogram, patch_shape = self.processor.prepare_patches(spectrogram)
            
            # Create drum label vector
            label_vector = self._create_label_vector(row['label'])
            
            return {
                'spectrogram': spectrogram,
                'labels': label_vector,
                'patch_shape': torch.tensor(patch_shape),
                'track_id': torch.tensor(row['track_id'], dtype=torch.long),
                'start_time': torch.tensor(start_time, dtype=torch.float32),
                'end_time': torch.tensor(end_time, dtype=torch.float32)
            }
            
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            # Return dummy data to avoid breaking the batch
            return self._get_dummy_sample()
    
    def _augment_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply audio augmentations for training."""
        try:
            # Time stretching using resample (alternative to stretch which doesn't exist)
            if torch.rand(1) < 0.3:
                stretch_factor = 0.9 + torch.rand(1).item() * 0.2  # Random between 0.9 and 1.1
                # Use resample to simulate time stretching
                original_sr = self.config.sample_rate
                stretched_sr = int(original_sr * stretch_factor)
                if stretched_sr != original_sr:
                    resampler_stretch = torchaudio.transforms.Resample(original_sr, stretched_sr)
                    resampler_back = torchaudio.transforms.Resample(stretched_sr, original_sr)
                    audio = resampler_back(resampler_stretch(audio))
            
            # Pitch shifting
            if torch.rand(1) < 0.3:
                n_steps = torch.randint(-2, 3, (1,)).item()
                try:
                    audio = torchaudio.functional.pitch_shift(audio, self.config.sample_rate, n_steps)
                except Exception:
                    # Skip pitch shifting if it fails
                    pass
            
            # Add noise
            if torch.rand(1) < 0.2:
                noise_factor = 0.001 + torch.rand(1).item() * 0.009  # Random between 0.001 and 0.01
                noise = torch.randn_like(audio) * noise_factor
                audio = audio + noise
        
        except Exception as e:
            # If any augmentation fails, return original audio
            logger.warning(f"Audio augmentation failed: {e}")
        
        return audio
    
    def _create_label_vector(self, label: Union[int, List[int]]) -> torch.Tensor:
        """Create multi-hot label vector from drum labels."""
        label_vector = torch.zeros(self.config.num_drum_classes, dtype=torch.float32)
        
        if isinstance(label, (int, np.integer)):
            labels = [label]
        else:
            labels = label if isinstance(label, list) else [label]
        
        for lbl in labels:
            if lbl in self.drum_mapping:
                class_idx = self.drum_mapping[lbl]
                label_vector[class_idx] = 1.0
            else:
                # Map unknown labels to "other" class
                label_vector[-1] = 1.0
        
        return label_vector
    
    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Return dummy sample for error cases."""
        dummy_time = int(self.config.max_audio_length * self.config.sample_rate / self.config.hop_length) + 1
        dummy_spec = torch.zeros(1, dummy_time, self.config.n_mels)
        dummy_spec, patch_shape = self.processor.prepare_patches(dummy_spec)
        
        return {
            'spectrogram': dummy_spec,
            'labels': torch.zeros(self.config.num_drum_classes, dtype=torch.float32),
            'patch_shape': torch.tensor(patch_shape),
            'track_id': torch.tensor(0, dtype=torch.long),
            'start_time': torch.tensor(0.0, dtype=torch.float32),
            'end_time': torch.tensor(0.0, dtype=torch.float32)
        }


def create_data_loaders(config: BaseConfig, 
                       data_dir: str,
                       audio_dir: str,
                       batch_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for train, validation, and test sets."""
    
    if batch_size is None:
        batch_size = config.train_batch_size
    
    # Load data splits
    train_files = list(Path(data_dir).glob("*_train.pkl"))
    val_files = list(Path(data_dir).glob("*_val.pkl"))
    test_files = list(Path(data_dir).glob("*_test.pkl"))
    
    if not train_files:
        raise FileNotFoundError(f"No training data found in {data_dir}")
    
    # Load and concatenate data
    train_dfs = [pd.read_pickle(f) for f in train_files]
    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    
    val_dfs = [pd.read_pickle(f) for f in val_files]
    val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
    
    test_dfs = [pd.read_pickle(f) for f in test_files]
    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
    
    logger.info(f"Loaded {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")
    
    # Create datasets
    train_dataset = DrumDataset(train_df, config, audio_dir, mode='train', augment=True)
    val_dataset = DrumDataset(val_df, config, audio_dir, mode='val', augment=False)
    test_dataset = DrumDataset(test_df, config, audio_dir, mode='test', augment=False)
    
    # Create data loaders with memory-optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=getattr(config, 'persistent_workers', False) and config.num_workers > 0,
        prefetch_factor=getattr(config, 'prefetch_factor', 2) if config.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=getattr(config, 'persistent_workers', False) and config.num_workers > 0,
        prefetch_factor=getattr(config, 'prefetch_factor', 2) if config.num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=getattr(config, 'persistent_workers', False) and config.num_workers > 0,
        prefetch_factor=getattr(config, 'prefetch_factor', 2) if config.num_workers > 0 else None
    )
    
    return train_loader, val_loader, test_loader


def convert_legacy_data(legacy_data_dir: str, output_dir: str, audio_dir: str) -> None:
    """Convert legacy pickle files to transformer-compatible format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all pickle files
    pickle_files = list(Path(legacy_data_dir).glob("*.pkl"))
    
    logger.info(f"Converting {len(pickle_files)} legacy data files...")
    
    for pickle_file in tqdm(pickle_files):
        try:
            # Load legacy data
            df = pd.read_pickle(pickle_file)
            
            # Filter out rows with missing audio files
            valid_rows = []
            for _, row in df.iterrows():
                audio_path = Path(audio_dir) / row['audio_filename']
                if audio_path.exists():
                    valid_rows.append(row)
                else:
                    logger.warning(f"Audio file not found: {audio_path}")
            
            if valid_rows:
                filtered_df = pd.DataFrame(valid_rows)
                output_file = output_path / pickle_file.name
                filtered_df.to_pickle(output_file)
                logger.info(f"Converted {pickle_file.name}: {len(df)} -> {len(filtered_df)} samples")
            else:
                logger.warning(f"No valid samples in {pickle_file.name}")
                
        except Exception as e:
            logger.error(f"Error converting {pickle_file}: {e}")


if __name__ == "__main__":
    # Test data loading
    from .transformer_config import get_config
    
    config = get_config("local")
    
    # Create dummy data for testing
    dummy_df = pd.DataFrame({
        'audio_filename': ['test_audio.wav'] * 10,
        'track_id': range(10),
        'start': np.random.uniform(0, 10, 10),
        'end': np.random.uniform(2, 12, 10),
        'label': np.random.choice([0, 1, 2, 66, 67, 68], 10)
    })
    
    # Test dataset creation
    dataset = DrumDataset(dummy_df, config, '/tmp', mode='test', augment=False)
    print(f"Dataset created with {len(dataset)} samples")
    
    # Test sample loading (will use dummy data due to missing files)
    sample = dataset[0]
    print(f"Sample spectrogram shape: {sample['spectrogram'].shape}")
    print(f"Sample labels shape: {sample['labels'].shape}")
    print("Data loading test completed!")