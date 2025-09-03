from typing import Optional, Any

import numpy as np
from numpy.typing import NDArray


def augment_spectrogram_time_masking(
    spec: NDArray[Any],
    num_masks: int = 1,
    max_mask_percentage: float = 0.1,
    mask_value: Optional[float] = None,
    overwrite: bool = False,
) -> NDArray[Any]:
    """
    Apply time masking to a spectrogram (frequency x time).
    """
    if not overwrite:
        spec = spec.copy()

    if mask_value is None:
        mask_value = float(spec.min())

    num_freq_bins, num_time_steps = spec.shape

    for _ in range(num_masks):
        max_mask_width = int(max_mask_percentage * num_time_steps)
        if max_mask_width < 1:
            max_mask_width = 1
        mask_width = int(np.random.randint(1, max_mask_width + 1))
        if num_time_steps - mask_width < 0:
            continue
        start_step = int(np.random.randint(0, num_time_steps - mask_width))
        spec[:, start_step : start_step + mask_width] = mask_value

    return spec


def augment_spectrogram_frequency_masking(
    spec: NDArray[Any],
    num_masks: int = 1,
    max_mask_percentage: float = 0.15,
    mask_value: Optional[float] = None,
    overwrite: bool = False,
) -> NDArray[Any]:
    """
    Apply frequency masking to a spectrogram (frequency x time).
    """
    if not overwrite:
        spec = spec.copy()

    if mask_value is None:
        mask_value = float(spec.min())

    num_freq_bins, num_time_steps = spec.shape

    for _ in range(num_masks):
        max_mask_height = int(max_mask_percentage * num_freq_bins)
        if max_mask_height < 1:
            max_mask_height = 1
        mask_height = int(np.random.randint(1, max_mask_height + 1))
        if num_freq_bins - mask_height < 0:
            continue
        start_bin = int(np.random.randint(0, num_freq_bins - mask_height))
        spec[start_bin : start_bin + mask_height, :] = mask_value

    return spec
