# mypy: ignore-errors
## This script contains functions to facilitate audio data augmentation for The AnNOTEators Project ##

import logging
import time
from math import sqrt
from typing import Any, Callable, Dict, List, Optional, Tuple

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pedalboard
from librosa.effects import pitch_shift
from numpy.typing import NDArray
from pedalboard import LowpassFilter, Pedalboard, Reverb
from tqdm.auto import tqdm

tqdm.pandas()

# Consider using audiomentations for a more robust and efficient pipeline in the future
# from audiomentations import Compose, AddGaussianNoise, PitchShift, LowPassFilter, TimeStretch

# Configure logging at the module level
logger = logging.getLogger(__name__)

# --- Type Aliases ---
# An augmentation function takes a numpy array (audio clip) and returns a modified numpy array.
# It can accept additional keyword arguments.
AugmentationFunction = Callable[..., NDArray[Any]]


# --- Helper for Randomization ---
def _get_random_value(param: Tuple[float, float] | float | int) -> float:
    """Gets a random value if param is a tuple/list (min, max), otherwise returns the param."""
    if isinstance(param, (list, tuple)) and len(param) == 2:
        return np.random.uniform(param[0], param[1])
    # If param is not a list/tuple, it must be a number.
    # We ensure it's a float for consistency.
    if isinstance(param, (int, float)):
        return float(param)
    # Raise an error for unexpected types.
    raise TypeError(
        f"Parameter 'param' must be a number or a tuple of two numbers, not {type(param)}"
    )


# --- Modified Augmentation Functions with Randomization ---


def add_lowpass_filter(
    audio_clip: NDArray[Any],
    sample_rate: int = 44100,
    cutoff_freq_hz_range: Tuple[float, float] = (800, 5000),
) -> NDArray[Any]:
    """
    Add pedalboard.LowpassFilter to an audio file with randomized cutoff frequency.

    Parameters:
        audio_clip: The audio clip (numpy array).
        sample_rate: This number specifies the audio sampling rate.
        cutoff_freq_hz_range: Tuple (min, max) for the lowpass filter frequency cutoff in Hz.
                              A random value within this range will be chosen.

    Returns:
        An augmented numpy.ndarray, with LowpassFilter applied.
    """
    cutoff_freq = _get_random_value(cutoff_freq_hz_range)
    pb = Pedalboard([LowpassFilter(cutoff_frequency_hz=cutoff_freq)])
    processed_audio: NDArray[Any] = pb(audio_clip, sample_rate)
    return processed_audio.flatten()


def add_pedalboard_effects(
    audio_clip: NDArray[Any],
    sample_rate: int = 44100,
    pb: Optional[Pedalboard] = None,
    room_size_range: Tuple[float, float] = (0.1, 0.9),
    cutoff_freq_hz_range: Tuple[float, float] = (800, 5000),
) -> NDArray[Any]:
    """
    Add pedalboard effects (Reverb, Lowpass) with randomized parameters.

    Parameters:
        audio_clip: The audio clip (numpy array).
        sample_rate: This number specifies the audio sampling rate.
        pb: pedalboard.Pedalboard initialized with effects. If None, Reverb and LowpassFilter
            with randomized parameters are used.
        room_size_range: Tuple (min, max) for Reverb room_size.
        cutoff_freq_hz_range: Tuple (min, max) for LowpassFilter cutoff frequency in Hz.

    Returns:
        An augmented numpy.ndarray, with Pedalboard effects applied.
    """
    if pb is None:
        room_size = _get_random_value(room_size_range)
        cutoff_freq = _get_random_value(cutoff_freq_hz_range)
        pb = Pedalboard(
            [
                Reverb(room_size=room_size),
                LowpassFilter(cutoff_frequency_hz=cutoff_freq),
            ]
        )

    processed_audio: NDArray[Any] = pb(audio_clip, sample_rate)
    return processed_audio.flatten()


def add_white_noise(
    audio_clip: NDArray[Any],
    snr_db_range: Tuple[float, float] = (5, 30),
    random_state: Optional[int] = None,
) -> NDArray[Any]:
    """
    Add white noise to an audio signal, scaling by a randomized signal-to-noise ratio (SNR).

    Parameters:
        audio_clip: The audio clip (numpy array).
        snr_db_range: Tuple (min_snr_db, max_snr_db) for the target SNR in dB.
                      A random SNR will be chosen from this range.
        random_state: Integer seed for numpy's random number generator (optional).

    Returns:
        An augmented numpy.ndarray, with white noise added.
    """
    if isinstance(random_state, int):
        np.random.seed(
            seed=random_state
        )  # Note: This sets global seed, consider using np.random.Generator

    snr = _get_random_value(snr_db_range)

    audio_clip_rms = sqrt(np.mean(audio_clip**2))
    if audio_clip_rms == 0:  # Avoid division by zero for silent clips
        return audio_clip

    noise_rms = sqrt(audio_clip_rms**2 / (10 ** (snr / 10)))
    white_noise = np.random.normal(loc=0, scale=noise_rms, size=audio_clip.shape[0])

    return audio_clip + white_noise


# --- Safer apply_augmentations using dictionary lookup ---
def apply_augmentations(
    df: pd.DataFrame,
    audio_col: str = "audio_wav",
    aug_col_names: Optional[List[str]] = None,
    **aug_param_dict: Any,
) -> pd.DataFrame:
    """
    Helper function for applying a specified set of augmentations to a dataframe containing audio.
    Uses a dictionary lookup instead of eval() for safety.

    Parameters:
        df: Audio dataframe containing the audio_col specified and in which augmentations will be stored.
        audio_col: String corresponding to the name of the column to use for audio data augmentation,
                   in numpy array format.
        aug_col_names: Optional list of names to use for augmented columns. If None, uses the
                       augmentation function names. Must match the order and number of functions
                       in aug_param_dict.
        aug_param_dict: Dictionary where keys are strings matching keys in AVAILABLE_AUGMENTATIONS
                        and values are dictionaries of parameters for that function.

    Returns:
        A dataframe with new columns containing augmented numpy arrays.
    """
    aug_start_time = time.perf_counter()
    logger.debug(f"Applying augmentations with parameters: {aug_param_dict}")

    aug_df = df.copy(deep=True)

    applied_funcs = []
    for func_name, params in aug_param_dict.items():
        if func_name in AVAILABLE_AUGMENTATIONS:
            print(f"Applying {func_name}")
            augmentation_func: AugmentationFunction = AVAILABLE_AUGMENTATIONS[func_name]
            # Using progress_apply can be slow; consider alternatives for large datasets
            aug_df[func_name] = aug_df[audio_col].progress_apply(
                lambda x: augmentation_func(x, **params)
            )
            applied_funcs.append(func_name)
        else:
            print(
                f"Warning: Unknown or disallowed augmentation function '{func_name}' skipped."
            )

    if aug_col_names is not None:
        if len(aug_col_names) == len(applied_funcs):
            col_dict = dict(zip(applied_funcs, aug_col_names))
            aug_df.rename(columns=col_dict, inplace=True)
        else:
            print(
                "Warning: Length of aug_col_names does not match the number of successfully applied augmentations. Skipping rename."
            )

    aug_end_time = time.perf_counter()
    logger.debug(
        f"Augmentation applied in {aug_end_time - aug_start_time:.4f} seconds."
    )

    return aug_df


def augment_pitch(
    audio_clip: NDArray[Any],
    sample_rate: int = 44100,
    n_steps_range: Tuple[int, int] = (-3, 3),
    bins_per_octave: int = 12,
    res_type: str = "kaiser_best",
) -> NDArray[Any]:
    """
    Augment the pitch of an audio file by a randomized number of steps.

    Parameters:
        audio_clip: The audio clip (numpy array).
        sample_rate: This number specifies the audio sampling rate.
        n_steps_range: Tuple (min_steps, max_steps) for pitch shifting. A random integer
                       number of steps within this range (inclusive) will be chosen.
        bins_per_octave: Number of steps per octave (12 for semitones).
        res_type: Resampling strategy for librosa.pitch_shift.

    Returns:
        An augmented numpy.ndarray, pitch shifted.
    """
    if not (isinstance(n_steps_range, (list, tuple)) and len(n_steps_range) == 2):
        print(
            "Warning: n_steps_range should be a tuple/list of (min, max). Using n_steps=0."
        )
        n_steps = 0
    else:
        # Ensure integer steps are chosen, and that the step is not 0
        n_steps = np.random.randint(n_steps_range[0], n_steps_range[1] + 1)
        while n_steps == 0:
            n_steps = np.random.randint(n_steps_range[0], n_steps_range[1] + 1)

    if n_steps == 0:
        return audio_clip  # No shift needed

    return pitch_shift(
        y=audio_clip,
        sr=sample_rate,
        n_steps=float(n_steps),  # librosa expects float
        bins_per_octave=bins_per_octave,
        res_type=res_type,
    )


def augment_pitch_jitter(
    audio_clip: NDArray[Any],
    sample_rate: int = 44100,
    pitch_range_cents: Tuple[float, float] = (-50, 50),
) -> NDArray[Any]:
    """
    Apply subtle, random pitch jitter to an audio clip.
    This helps prevent the model from overfitting to the specific tuning of kit pieces.
    """
    cents_shift = _get_random_value(pitch_range_cents)
    n_steps = cents_shift / 100.0  # Convert cents to semitones

    if abs(n_steps) < 0.01:  # No significant shift
        return audio_clip

    return pitch_shift(y=audio_clip, sr=sample_rate, n_steps=n_steps)


def augment_time_stretch(
    audio_clip: NDArray[Any], rate_range: Tuple[float, float] = (0.95, 1.05)
) -> NDArray[Any]:
    """
    Apply subtle, random time stretching to an audio clip.
    This alters the attack/decay characteristics.
    """
    rate = _get_random_value(rate_range)

    if abs(rate - 1.0) < 0.01:  # No significant stretch
        return audio_clip

    return librosa.effects.time_stretch(y=audio_clip, rate=rate)


def augment_dynamic_eq(
    audio_clip: NDArray[Any], sample_rate: int = 44100
) -> NDArray[Any]:
    """
    Apply a randomized EQ to alter the timbre of the audio.
    """
    # Apply pre-emphasis outside of the Pedalboard chain
    preemphasized_audio: NDArray[Any] = librosa.effects.preemphasis(audio_clip)

    board = Pedalboard(
        [
            pedalboard.PeakFilter(
                cutoff_frequency_hz=_get_random_value((200, 2000)),
                gain_db=_get_random_value((-6, 6)),
                q=_get_random_value((0.5, 2.0)),
            ),
            pedalboard.HighShelfFilter(
                cutoff_frequency_hz=_get_random_value((1500, 8000)),
                gain_db=_get_random_value((-10, 10)),
                q=_get_random_value((0.5, 1.5)),
            ),
            pedalboard.LowShelfFilter(
                cutoff_frequency_hz=_get_random_value((80, 400)),
                gain_db=_get_random_value((-10, 10)),
                q=_get_random_value((0.5, 1.5)),
            ),
        ]
    )
    processed_audio: NDArray[Any] = board(preemphasized_audio, sample_rate)
    return processed_audio.flatten()


# --- Spectrogram Augmentation ---


def augment_spectrogram_time_masking(
    spec: NDArray[Any],
    num_masks: int = 1,
    max_mask_percentage: float = 0.1,
    mask_value: Optional[float] = None,
    overwrite: bool = False,
) -> NDArray[Any]:
    """
    Apply time masking to a spectrogram (e.g., Mel spectrogram).
    Masks vertical bands corresponding to time steps.

    Parameters:
        spec: Numpy array (frequency, time).
        num_masks: Integer, number of masks to apply.
        max_mask_percentage: Float (0 to 1), maximum percentage of total time steps a single mask can cover.
        mask_value: Value to set the masked region to. If None, uses the global minimum of the spectrogram.
        overwrite: Boolean signifying whether to overwrite the input array.

    Returns:
        Augmented spectrogram (numpy array).
    """
    if not overwrite:
        spec = spec.copy()

    if mask_value is None:
        mask_value = spec.min()

    num_freq_bins, num_time_steps = spec.shape

    for _ in range(num_masks):
        # Determine mask width (number of time steps)
        max_mask_width = int(max_mask_percentage * num_time_steps)
        if max_mask_width < 1:
            max_mask_width = 1  # Ensure at least 1 step can be masked
        mask_width = np.random.randint(1, max_mask_width + 1)

        # Determine start position
        if num_time_steps - mask_width < 0:
            continue  # Skip if mask is wider than spectrogram
        start_step = np.random.randint(0, num_time_steps - mask_width)

        # Apply mask
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
    Apply frequency masking to a spectrogram (e.g., Mel spectrogram).
    Masks horizontal bands corresponding to frequency bins.

    Parameters:
        spec: Numpy array (frequency, time).
        num_masks: Integer, number of masks to apply.
        max_mask_percentage: Float (0 to 1), maximum percentage of total frequency bins a single mask can cover.
        mask_value: Value to set the masked region to. If None, uses the global minimum of the spectrogram.
        overwrite: Boolean signifying whether to overwrite the input array.

    Returns:
        Augmented spectrogram (numpy array).
    """
    if not overwrite:
        spec = spec.copy()

    if mask_value is None:
        mask_value = spec.min()

    num_freq_bins, num_time_steps = spec.shape

    for _ in range(num_masks):
        # Determine mask height (number of frequency bins)
        max_mask_height = int(max_mask_percentage * num_freq_bins)
        if max_mask_height < 1:
            max_mask_height = 1  # Ensure at least 1 bin can be masked
        mask_height = np.random.randint(1, max_mask_height + 1)

        # Determine start position
        if num_freq_bins - mask_height < 0:
            continue  # Skip if mask is taller than spectrogram
        start_bin = np.random.randint(0, num_freq_bins - mask_height)

        # Apply mask
        spec[start_bin : start_bin + mask_height, :] = mask_value

    return spec


# --- Deprecated/Modified Span Augmentation ---
def get_span_indices(
    dim: int,
    min_span: int = 5,
    max_span: Optional[int] = None,
    span_variation: int = 0,
) -> List[int]:
    """
    Helper function to find array indices for span augmentation.
    """
    # Add variation to min/max span lengths
    rand_min_span = min_span + np.random.randint(
        low=-span_variation, high=span_variation + 1
    )
    if rand_min_span <= 0:
        rand_min_span = 1

    rand_max_span = max_span
    if max_span is not None:
        rand_max_span = max_span + np.random.randint(
            low=-span_variation, high=span_variation + 1
        )
        if rand_max_span < rand_min_span:
            rand_max_span = rand_min_span
        if rand_max_span <= 0:
            rand_max_span = 1

    # Determine span length
    if rand_max_span is not None:
        span_len = np.random.randint(rand_min_span, rand_max_span + 1)
    else:
        span_len = rand_min_span

    span_len = min(span_len, dim)
    if span_len <= 0:
        span_len = 1

    if dim - span_len < 0:
        start_ind = 0
        span_len = dim
    else:
        start_ind = np.random.randint(0, dim - span_len)

    end_ind = start_ind + span_len - 1
    return [start_ind, end_ind]


# --- Visualization (Unchanged) ---
def compare_waveforms(
    df: pd.DataFrame,
    i: int,
    signal_cols: List[str],
    signal_labs: Optional[List[str]] = None,
    sample_rate: int = 44100,
    max_pts: Optional[int] = None,
    alpha: float | List[float] = 0.5,
    fontsizes: List[int] = [24, 18, 20],
    figsize: Tuple[int, int] = (16, 12),
    leg_loc: str = "best",
    title: str = "",
) -> None:
    if signal_labs is None:
        signal_labs = signal_cols
    elif (len(signal_labs) < len(signal_cols)) or not isinstance(signal_labs, list):
        print("Not enough labels were provided. Using column names as labels.")
        signal_labs = signal_cols

    alpha_list: List[float]
    if isinstance(alpha, float):
        alpha_list = [alpha for _ in signal_cols]
    elif len(alpha) < len(signal_cols):
        print("Alpha list insufficient, using first value for all signals.")
        alpha_list = [alpha[0] for _ in signal_cols]
    else:
        alpha_list = alpha

    plt.figure(figsize=figsize)

    for col, lab, alp in zip(signal_cols, signal_labs, alpha_list):
        audio_data = df.loc[i, col].astype(np.float32)

        waveshow_kwargs: Dict[str, Any] = {
            "y": audio_data,
            "sr": sample_rate,
            "label": lab,
            "alpha": alp,
        }
        if max_pts is not None:
            waveshow_kwargs["max_points"] = max_pts

        librosa.display.waveshow(**waveshow_kwargs)

    if not title:
        plt.title(
            f"Comparison of audio clips for element: {i}, label: {df.loc[i, 'label']}",
            fontsize=fontsizes[0],
        )
    else:
        plt.title(title, fontsize=fontsizes[0])

    plt.gca().xaxis.label.set_fontsize(fontsizes[1])
    plt.legend(signal_labs, loc=leg_loc, fontsize=fontsizes[2])


# --- Original augment_spectrogram_spans (Marked for review/deprecation) ---
def augment_spectrogram_spans(
    spec: NDArray[Any],
    spans: int = 3,
    span_ranges: List[List[int]] = [[1, 4], [1, 6]],
    span_variation: int = 1,
    ind_lists: Optional[List[List[List[int]]]] = None,
    sig_val: Optional[float] = None,
    overwrite: bool = False,
) -> NDArray[Any]:
    """
    DEPRECATION WARNING: This function is kept for legacy purposes.
    Consider using augment_spectrogram_time_masking and augment_spectrogram_frequency_masking.
    """
    if not overwrite:
        spec = spec.copy()

    if sig_val is None:
        sig_val = spec.min()

    if not isinstance(spans, int):
        print("Value for spans was not an integer. Setting spans to 1.")
        spans = 1

    if not ind_lists:
        ind_lists = []
        dims = spec.shape
        for _ in range(spans):
            time_inds = get_span_indices(
                dims[1],
                min_span=span_ranges[0][0],
                max_span=span_ranges[0][1],
                span_variation=span_variation,
            )
            freq_inds = get_span_indices(
                dims[0],
                min_span=span_ranges[1][0],
                max_span=span_ranges[1][1],
                span_variation=span_variation,
            )
            ind_lists.append([time_inds, freq_inds])

    for ind_list in ind_lists:
        time_inds, freq_inds = ind_list[0], ind_list[1]
        spec[freq_inds[0] : freq_inds[1] + 1, time_inds[0] : time_inds[1] + 1] = sig_val

    return spec


# --- Available augmentations dictionary ---
AVAILABLE_AUGMENTATIONS: Dict[str, AugmentationFunction] = {
    "add_lowpass_filter": add_lowpass_filter,
    "add_pedalboard_effects": add_pedalboard_effects,
    "add_white_noise": add_white_noise,
    "augment_pitch": augment_pitch,
}
