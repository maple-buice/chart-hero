#!/usr/bin/env python3
"""
Quick test script to verify the data pipeline with a small subset of E-GMD data.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_training.transformer_config import get_config
from model_training.transformer_data import SpectrogramProcessor, DrumDataset
import torch
import torchaudio

def test_data_pipeline():
    """Test the transformer data pipeline with actual E-GMD files."""
    print("Testing transformer data pipeline with E-GMD dataset...")
    
    # Set up paths
    egmd_dir = Path("datasets/e-gmd-v1.0.0")
    csv_file = egmd_dir / "e-gmd-v1.0.0.csv"
    
    if not csv_file.exists():
        print(f"❌ CSV file not found: {csv_file}")
        return False
    
    # Load CSV and get a few samples
    print(f"Loading CSV from: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Found {len(df)} total samples in dataset")
    
    # Take a small sample for testing
    sample_df = df.head(5).copy()
    print(f"Testing with {len(sample_df)} samples")
    
    # Add required columns for transformer data pipeline
    sample_df['track_id'] = range(len(sample_df))
    sample_df['start'] = 0.0  # Start from beginning
    sample_df['end'] = sample_df['duration'].clip(upper=8.0)  # Max 8 seconds
    sample_df['label'] = [0, 1, 2, 66, 67]  # Different drum types for testing
    
    print("\nSample data:")
    print(sample_df[['midi_filename', 'audio_filename', 'duration', 'start', 'end', 'label']].head())
    
    # Test configuration
    config = get_config("local")
    print(f"\nUsing config: {config.__class__.__name__}")
    print(f"Device: {config.device}")
    print(f"Max audio length: {config.max_audio_length}s")
    print(f"Sample rate: {config.sample_rate}")
    print(f"Patch size: {config.patch_size}")
    
    # Test spectrogram processor
    print("\nTesting spectrogram processor...")
    processor = SpectrogramProcessor(config)
    
    # Load a test audio file
    test_audio_file = egmd_dir / sample_df.iloc[0]['audio_filename']
    if not test_audio_file.exists():
        print(f"❌ Test audio file not found: {test_audio_file}")
        return False
    
    print(f"Loading test audio: {test_audio_file}")
    audio, sr = torchaudio.load(str(test_audio_file))
    print(f"Original audio shape: {audio.shape}, sample rate: {sr}")
    
    # Convert to mono and resample if needed
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
        print(f"Converted to mono: {audio.shape}")
    
    if sr != config.sample_rate:
        resampler = torchaudio.transforms.Resample(sr, config.sample_rate)
        audio = resampler(audio)
        print(f"Resampled to {config.sample_rate} Hz: {audio.shape}")
    
    # Test spectrogram conversion
    spectrogram = processor.audio_to_spectrogram(audio)
    print(f"Spectrogram shape: {spectrogram.shape}")
    
    # Test patch preparation
    padded_spec, patch_shape = processor.prepare_patches(spectrogram)
    print(f"Padded spectrogram shape: {padded_spec.shape}")
    print(f"Patch shape: {patch_shape}")
    
    # Test dataset creation
    print("\nTesting dataset creation...")
    try:
        dataset = DrumDataset(
            sample_df, 
            config, 
            str(egmd_dir), 
            mode='test', 
            augment=False
        )
        print(f"Dataset created with {len(dataset)} samples")
        
        # Test loading a sample
        print("Testing sample loading...")
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Spectrogram shape: {sample['spectrogram'].shape}")
        print(f"Labels shape: {sample['labels'].shape}")
        print(f"Labels: {sample['labels']}")
        print(f"Track ID: {sample['track_id']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔍 Chart-Hero Data Pipeline Test")
    print("=" * 40)
    
    success = test_data_pipeline()
    
    if success:
        print("\n✅ Data pipeline test passed!")
        print("\nNext steps:")
        print("1. Run full data preparation:")
        print("   python prepare_egmd_data.py --sample-ratio 0.1")
        print("2. Start transformer training:")
        print("   python model_training/train_transformer.py --config auto")
    else:
        print("\n❌ Data pipeline test failed!")
        print("Please check the errors above and ensure:")
        print("- E-GMD dataset is properly extracted to datasets/e-gmd-v1.0.0/")
        print("- CSV file exists at datasets/e-gmd-v1.0.0/e-gmd-v1.0.0.csv")
        print("- Audio files are accessible")

if __name__ == "__main__":
    main()