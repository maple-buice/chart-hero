from chart_hero.model_training.transformer_config import get_config

def test_set_window_length_adjusts_seq_len_and_batch():
    config = get_config("local")
    base_train = config.train_batch_size
    base_val = config.val_batch_size
    frames_per_sec = config.sample_rate / config.hop_length

    config.set_window_length(30.0)

    assert config.max_audio_length == 30.0
    assert config.window_length_seconds == 30.0
    assert config.max_seq_len == int(30.0 * frames_per_sec)
    assert config.train_batch_size == max(1, int(base_train * 20.0 / 30.0))
    assert config.val_batch_size == max(1, int(base_val * 20.0 / 30.0))


def test_cloud_config_window_length_override():
    config = get_config("cloud", max_audio_length=30.0)
    frames_per_sec = config.sample_rate / config.hop_length
    assert config.max_audio_length == 30.0
    assert config.window_length_seconds == 30.0
    assert config.max_seq_len == int(30.0 * frames_per_sec)
