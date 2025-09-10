import numpy as np

from chart_hero.model_training.transformer_config import get_config
from chart_hero.model_training.transformer_data import SlidingWindowDataset


def _create_dummy_pair(tmp_path, config, length):
    spec = np.random.rand(config.n_mels, length).astype(np.float32)
    labels = np.random.randint(
        0, 2, (length, config.num_drum_classes), dtype=np.int32
    ).astype(np.float32)
    spec_file = tmp_path / "song_mel.npy"
    label_file = tmp_path / "song_label.npy"
    np.save(spec_file, spec)
    np.save(label_file, labels)
    return [(str(spec_file), str(label_file))]


def test_validation_windows_deterministic(tmp_path):
    config = get_config("local")
    config.max_seq_len = 40
    config.max_audio_length = 40 * config.hop_length / config.sample_rate
    pairs = _create_dummy_pair(tmp_path, config, length=100)
    dataset = SlidingWindowDataset(pairs, config, mode="val")
    assert len(dataset) == 4
    expected = [(0, 0), (0, 20), (0, 40), (0, 60)]
    assert dataset.window_indices == expected
    for spec, labels in dataset:
        assert spec.shape[-1] == config.max_seq_len
        assert labels.shape[0] == config.max_seq_len


def test_training_windows_change_with_epoch(tmp_path):
    config = get_config("local")
    config.max_seq_len = 40
    config.max_audio_length = 40 * config.hop_length / config.sample_rate
    config.window_jitter_ratio = 0.5
    pairs = _create_dummy_pair(tmp_path, config, length=120)
    dataset = SlidingWindowDataset(pairs, config, mode="train")
    dataset.set_epoch(0)
    idx0 = dataset.window_indices.copy()
    dataset.set_epoch(1)
    idx1 = dataset.window_indices.copy()
    assert idx0 != idx1
    assert len(idx0) == len(idx1)
