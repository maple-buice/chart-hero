import os
from typing import Any, Optional

import torch

from chart_hero.model_training.lightning_module import DrumTranscriptionModule


def select_device(config: Any) -> torch.device:
    """Select an appropriate torch device based on availability and config."""
    device_str = getattr(config, "device", None)
    if device_str:
        return torch.device(device_str)
    has_mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    if has_mps:
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_from_checkpoint(
    config: Any, model_path: str | os.PathLike[str], max_time_patches: Optional[int] = None
) -> DrumTranscriptionModule:
    """Load a DrumTranscriptionModule from a checkpoint path."""
    model = DrumTranscriptionModule.load_from_checkpoint(
        str(model_path), config=config, max_time_patches=max_time_patches, strict=False
    )
    model.eval()
    return model
