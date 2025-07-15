import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import tempfile
import torch
from chart_hero.model_training.data_preparation import data_preparation

@pytest.fixture
def mock_dataset(tmp_path):
    """Create a mock dataset with a CSV file and dummy audio files."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    
    # Create a mock CSV file
    csv_data = {
        'midi_filename': ['a.mid', 'b.mid', 'c.mid'],
        'audio_filename': ['a.wav', 'b.wav', 'c.wav'],
        'duration': [1.0, 2.0, 3.0]
    }
    df = pd.DataFrame(csv_data)
    csv_path = dataset_dir / "data.csv"
    df.to_csv(csv_path, index=False)
    
    # Create dummy audio files
    for audio_file in csv_data['audio_filename']:
        (dataset_dir / audio_file).touch()
        
    return dataset_dir

def test_data_preparation_initialization(mock_dataset):
    """Test that the data_preparation class initializes correctly."""
    
    # Mock soundfile.info to avoid reading actual audio files
    with patch('soundfile.info') as mock_sf_info:
        mock_sf_info.return_value = MagicMock(duration=1.0)
        
        data_prep = data_preparation(
            directory_path=str(mock_dataset),
            dataset='egmd',
            sample_ratio=1.0,
            diff_threshold=2.0
        )
        
        assert data_prep is not None
        assert data_prep.dataset_type == 'egmd'
        assert len(data_prep.midi_wav_map) == 3

def test_create_audio_set(mock_dataset):
    """Test that the create_audio_set method works correctly."""
    
    with patch('soundfile.info') as mock_sf_info:
        mock_sf_info.return_value = MagicMock(duration=1.0)
        
        data_prep = data_preparation(
            directory_path=str(mock_dataset),
            dataset='egmd',
            sample_ratio=1.0,
            diff_threshold=2.0
        )
        
    mock_dataset_instance = MagicMock()
    mock_dataset_instance.__len__.return_value = 1
    mock_dataset_class = MagicMock(return_value=mock_dataset_instance)
    
    dummy_spectrogram = torch.randn(1, 1, 256, 128)
    dummy_label = torch.randn(1, 8)
    mock_dataloader_class = MagicMock(return_value=[(dummy_spectrogram, dummy_label)])
            
    with tempfile.TemporaryDirectory() as temp_dir:
        num_files = data_prep.create_audio_set(
            dir_path=temp_dir,
            num_batches=1,
            dataset_class=mock_dataset_class,
            dataloader_class=mock_dataloader_class
        )
                
        assert num_files == 3
                
        # Check that the output files were created
        output_files = list(Path(temp_dir).glob("*.npy"))
        assert len(output_files) == 6 # 2 for each of train, val, test
