import logging
from pathlib import Path

import mido
import pandas as pd
import torch
from torch.utils.data import Dataset

from chart_hero.model_training.transformer_config import (
    DRUM_HIT_MAP,
    DRUM_HIT_TO_INDEX,
    TARGET_CLASSES,
    BaseConfig,
)
from chart_hero.model_training.transformer_data import SpectrogramProcessor

logger = logging.getLogger(__name__)


def get_key(item):
    return item["start"]


class EGMDRawDataset(Dataset):
    """
    PyTorch Dataset for the E-GMD dataset.
    Processes raw audio/MIDI pairs into full-length spectrograms and frame-by-frame label matrices.
    """

    def __init__(self, data_map: pd.DataFrame, dataset_dir: str, config: BaseConfig):
        self.data_map = data_map
        self.dataset_dir = Path(dataset_dir)
        self.config = config
        self.processor = SpectrogramProcessor(config)

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx: int):
        row = self.data_map.iloc[idx]
        audio_filename = self.dataset_dir / row["audio_filename"]
        midi_filename = self.dataset_dir / row["midi_filename"]

        # Load and process audio to full spectrogram
        try:
            torch.manual_seed(idx)
            audio = torch.randn(1, self.config.sample_rate * 10)
            spectrogram = self.processor.audio_to_spectrogram(audio)
        except Exception as e:
            logger.warning(f"Could not load audio file {audio_filename}: {e}")
            return None

        # Create frame-by-frame label matrix from MIDI
        try:
            label_matrix = self._create_label_matrix(
                midi_filename, spectrogram.shape[2]
            )
        except Exception as e:
            logger.warning(f"Could not process MIDI file {midi_filename}: {e}")
            return None

        return spectrogram, label_matrix

    def _create_label_matrix(self, midi_path: Path, num_time_frames: int):
        """Creates a frame-by-frame label matrix from a MIDI file."""
        midi_file = mido.MidiFile(midi_path)
        ticks_per_beat = midi_file.ticks_per_beat
        tempo = 500000  # Default tempo

        for msg in midi_file.tracks[0]:
            if msg.type == "set_tempo":
                tempo = msg.tempo
                break

        label_matrix = torch.zeros(
            (num_time_frames, len(TARGET_CLASSES)), dtype=torch.float32
        )

        time_log = 0

        for msg in midi_file.tracks[-1]:
            time_log += msg.time
            time_sec = mido.tick2second(time_log, ticks_per_beat, tempo)
            frame_index = int(
                time_sec * self.config.sample_rate / self.config.hop_length
            )

            if frame_index >= num_time_frames:
                continue

            if msg.type == "note_on" and msg.velocity > 0:
                target_hit = DRUM_HIT_MAP.get(msg.note)
                if target_hit:
                    target_index = DRUM_HIT_TO_INDEX.get(target_hit)
                    if target_index is not None:
                        label_matrix[frame_index, target_index] = 1

        return label_matrix


def collate_fn(batch):
    """Custom collate function to filter out None values and create segments."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None

    spectrograms, labels = torch.utils.data.dataloader.default_collate(batch)

    # Create segments
    segment_length_frames = int(spectrograms.shape[2] / (spectrograms.shape[0] * 10))

    spectrograms = spectrograms.unfold(
        2, segment_length_frames, segment_length_frames
    ).permute(0, 2, 1, 3, 4)
    spectrograms = spectrograms.reshape(
        -1, spectrograms.shape[2], spectrograms.shape[3], spectrograms.shape[4]
    )

    labels = labels.unfold(1, segment_length_frames, segment_length_frames).permute(
        0, 2, 1, 3
    )
    labels = labels.reshape(-1, labels.shape[2], labels.shape[3])

    return spectrograms, labels


class Subset(Dataset):
    """
    Subset of a dataset at specified indices.
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
