import numpy as np
import torch

from chart_hero.inference.charter import Charter
from chart_hero.model_training.transformer_config import get_config, get_drum_hits


def test_charter_predict_offsets(monkeypatch):
    config = get_config("local")
    config.patch_stride = config.patch_size[0]
    dummy_base = torch.nn.Module()
    monkeypatch.setattr(
        "chart_hero.inference.charter.load_model_from_checkpoint",
        lambda cfg, path, max_time_patches=None: dummy_base,
    )
    charter = Charter(config, "dummy_model.ckpt")

    class Dummy(torch.nn.Module):
        def __init__(self, C):
            super().__init__()
            self.C = C
        def forward(self, x):
            B = x.shape[0]
            logits = torch.full((B, 1, self.C), -10.0)
            logits[:, 0, 0] = 10.0
            return {"logits": logits}

    charter.model = Dummy(len(get_drum_hits()))

    fps = config.sample_rate / config.hop_length
    offset_frames = int(1 * fps)
    n_mels = config.n_mels
    patch = config.patch_size[0]
    segs = [
        {
            "spec": np.zeros((n_mels, offset_frames)),
            "start_frame": 0,
            "end_frame": offset_frames,
            "total_frames": offset_frames + patch,
        },
        {
            "spec": np.ones((n_mels, patch)),
            "start_frame": offset_frames,
            "end_frame": offset_frames + patch,
            "total_frames": offset_frames + patch,
        },
    ]

    df = charter.predict(segs)
    assert charter.last_offset_samples == offset_frames * config.hop_length
    assert not df.empty
    first = int(df.iloc[0]["peak_sample"])
    assert first >= charter.last_offset_samples
