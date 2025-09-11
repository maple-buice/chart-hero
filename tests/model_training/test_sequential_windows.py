import numpy as np
import torch

from chart_hero.model_training.transformer_config import get_config
from chart_hero.model_training.transformer_data import SlidingWindowDataset
from chart_hero.model_training.lightning_module import DrumTranscriptionModule


def _create_dummy_pair(tmp_path, config, length):
    spec = np.random.rand(config.n_mels, length).astype(np.float32)
    labels = np.random.randint(0, 2, (length, config.num_drum_classes), dtype=np.int32).astype(
        np.float32
    )
    spec_file = tmp_path / "song_mel.npy"
    label_file = tmp_path / "song_label.npy"
    np.save(spec_file, spec)
    np.save(label_file, labels)
    return [(str(spec_file), str(label_file))]


def test_dataset_returns_sequences(tmp_path):
    config = get_config("local")
    config.enable_sequential_windows = True
    config.sequence_length = 2
    config.set_window_length(40 * config.hop_length / config.sample_rate)
    pairs = _create_dummy_pair(tmp_path, config, length=120)
    dataset = SlidingWindowDataset(pairs, config, mode="val")
    spec, labels = dataset[0]
    assert spec.shape[0] == config.sequence_length
    assert labels.shape[0] == config.sequence_length


def test_module_training_step_with_sequences():
    config = get_config("local")
    config.enable_sequential_windows = True
    config.sequence_length = 2
    max_time_patches = config.max_seq_len // config.patch_size[0]
    model = DrumTranscriptionModule(config, max_time_patches=max_time_patches)
    model.trainer = None
    spectrogram = torch.randn(
        1, config.sequence_length, 1, config.n_mels, config.max_seq_len
    )
    labels = torch.randint(
        0,
        2,
        (1, config.sequence_length, config.max_seq_len, config.num_drum_classes),
    ).float()
    batch = (spectrogram, labels)
    loss = model.training_step(batch, 0)
    assert loss is not None
