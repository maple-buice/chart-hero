import torch
from torch import Tensor
from typing import Optional


def augment_spectrogram_time_masking(
    spec: Tensor,
    num_masks: int = 1,
    max_mask_percentage: float = 0.1,
    mask_value: Optional[float] = None,
) -> Tensor:
    """Apply time masking to a spectrogram (frequency x time)."""
    if mask_value is None:
        mask_value = float(spec.min())
    num_freq_bins, num_time_steps = spec.shape
    for _ in range(num_masks):
        max_mask_width = int(max_mask_percentage * num_time_steps)
        if max_mask_width < 1:
            max_mask_width = 1
        mask_width = int(torch.randint(1, max_mask_width + 1, (1,)).item())
        if num_time_steps - mask_width <= 0:
            continue
        start_step = int(torch.randint(0, num_time_steps - mask_width + 1, (1,)).item())
        spec[:, start_step : start_step + mask_width] = mask_value
    return spec


def augment_spectrogram_frequency_masking(
    spec: Tensor,
    num_masks: int = 1,
    max_mask_percentage: float = 0.15,
    mask_value: Optional[float] = None,
) -> Tensor:
    """Apply frequency masking to a spectrogram (frequency x time)."""
    if mask_value is None:
        mask_value = float(spec.min())
    num_freq_bins, num_time_steps = spec.shape
    for _ in range(num_masks):
        max_mask_height = int(max_mask_percentage * num_freq_bins)
        if max_mask_height < 1:
            max_mask_height = 1
        mask_height = int(torch.randint(1, max_mask_height + 1, (1,)).item())
        if num_freq_bins - mask_height <= 0:
            continue
        start_bin = int(torch.randint(0, num_freq_bins - mask_height + 1, (1,)).item())
        spec[start_bin : start_bin + mask_height, :] = mask_value
    return spec
