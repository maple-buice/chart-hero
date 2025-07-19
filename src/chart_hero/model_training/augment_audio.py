## This script contains functions to facilitate audio data augmentation for The AnNOTEators Project ##

import logging  # Import logging
import time  # Import time
from math import sqrt

import librosa.display
import matplotlib.pyplot as plt
import numpy as np  # Use np consistently
import pedalboard
from librosa.effects import pitch_shift
from numpy import mean, random
from pedalboard import LowpassFilter, Pedalboard, Reverb
from tqdm.auto import tqdm

tqdm.pandas()

# Consider using audiomentations for a more robust and efficient pipeline in the future
# from audiomentations import Compose, AddGaussianNoise, PitchShift, LowPassFilter, TimeStretch

# Configure logging at the module level
logger = logging.getLogger(__name__)


# --- Helper for Randomization ---
def _get_random_value(param):
    """Gets a random value if param is a tuple/list (min, max), otherwise returns the param."""
    if isinstance(param, (list, tuple)) and len(param) == 2:
        return random.uniform(param[0], param[1])
    return param


# --- Modified Augmentation Functions with Randomization ---


def add_lowpass_filter(audio_clip, sample_rate=44100, cutoff_freq_hz_range=(800, 5000)):
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
    try:
        return pb(audio_clip, sample_rate)
    except Exception as e:
        print(f"Error applying LowpassFilter: {e}")
        return audio_clip  # Return original clip on error


def add_pedalboard_effects(
    audio_clip,
    sample_rate=44100,
    pb=None,
    room_size_range=(0.1, 0.9),
    cutoff_freq_hz_range=(800, 5000),
):
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

    try:
        return pb(audio_clip, sample_rate)
    except Exception as e:
        print(f"Error applying Pedalboard effects: {e}")
        return audio_clip  # Return original clip on error


def add_white_noise(audio_clip, snr_db_range=(5, 30), random_state=None):
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
        random.seed(
            seed=random_state
        )  # Note: This sets global seed, consider using np.random.Generator

    snr = _get_random_value(snr_db_range)

    audio_clip_rms = sqrt(mean(audio_clip**2))
    if audio_clip_rms == 0:  # Avoid division by zero for silent clips
        return audio_clip

    noise_rms = sqrt(audio_clip_rms**2 / (10 ** (snr / 10)))
    white_noise = random.normal(loc=0, scale=noise_rms, size=audio_clip.shape[0])

    return audio_clip + white_noise


# --- Safer apply_augmentations using dictionary lookup ---
# Note: AVAILABLE_AUGMENTATIONS dictionary is defined at the end of the file
# after all functions are defined


def apply_augmentations(
    df, audio_col="audio_wav", aug_col_names=None, **aug_param_dict
):
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

    Example usage:
        aug_params = {
            'add_white_noise': {'snr_db_range': (15, 35)},
            'augment_pitch': {'n_steps_range': (-2, 2)},
            'add_pedalboard_effects': {'room_size_range': (0.2, 0.8)}
        }
        # Assuming 'aug_wn', 'aug_pitch', 'aug_effects' are desired column names
        augmented_df = apply_augmentations(audio_df, aug_col_names=['aug_wn', 'aug_pitch', 'aug_effects'], **aug_params)
    """
    aug_start_time = time.perf_counter()
    logger.debug(f"Applying augmentations with parameters: {aug_param_dict}")

    aug_df = df.copy(deep=True)

    applied_funcs = []
    for func_name, params in aug_param_dict.items():
        if func_name in AVAILABLE_AUGMENTATIONS:
            print(f"Applying {func_name}")
            augmentation_func = AVAILABLE_AUGMENTATIONS[func_name]
            try:
                # Using progress_apply can be slow; consider alternatives for large datasets
                aug_df[func_name] = aug_df[audio_col].progress_apply(  # type: ignore[attr-defined]
                    lambda x: augmentation_func(x, **params)
                )
                applied_funcs.append(func_name)
            except Exception as e:
                print(f"Error applying {func_name} via progress_apply: {e}")
                # Optionally remove the column if creation failed partway
                if func_name in aug_df.columns:
                    del aug_df[func_name]
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
    audio_clip,
    sample_rate=44100,
    n_steps_range=(-3, 3),
    bins_per_octave=12,
    res_type="kaiser_best",
):
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
        n_steps = random.randint(n_steps_range[0], n_steps_range[1] + 1)
        while n_steps == 0:
            n_steps = random.randint(n_steps_range[0], n_steps_range[1] + 1)

    if n_steps == 0:
        return audio_clip  # No shift needed

    try:
        return pitch_shift(
            y=audio_clip,
            sr=sample_rate,
            n_steps=float(n_steps),  # librosa expects float
            bins_per_octave=bins_per_octave,
            res_type=res_type,
        )
    except Exception as e:
        print(f"Error applying pitch shift: {e}")
        return audio_clip  # Return original clip on error


def augment_pitch_jitter(audio_clip, sample_rate=44100, pitch_range_cents=(-50, 50)):
    """
    Apply subtle, random pitch jitter to an audio clip.
    This helps prevent the model from overfitting to the specific tuning of kit pieces.
    """
    cents_shift = _get_random_value(pitch_range_cents)
    n_steps = cents_shift / 100.0  # Convert cents to semitones

    if abs(n_steps) < 0.01:  # No significant shift
        return audio_clip

    try:
        return pitch_shift(y=audio_clip, sr=sample_rate, n_steps=n_steps)
    except Exception as e:
        print(f"Error applying pitch jitter: {e}")
        return audio_clip


def augment_time_stretch(audio_clip, rate_range=(0.95, 1.05)):
    """
    Apply subtle, random time stretching to an audio clip.
    This alters the attack/decay characteristics.
    """
    rate = _get_random_value(rate_range)

    if abs(rate - 1.0) < 0.01:  # No significant stretch
        return audio_clip

    try:
        return librosa.effects.time_stretch(y=audio_clip, rate=rate)
    except Exception as e:
        print(f"Error applying time stretch: {e}")
        return audio_clip


def augment_dynamic_eq(audio_clip, sample_rate=44100):
    """
    Apply a randomized EQ to alter the timbre of the audio.
    """
    # Apply pre-emphasis outside of the Pedalboard chain
    audio_clip = librosa.effects.preemphasis(audio_clip)

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
    try:
        return board(audio_clip, sample_rate)
    except Exception as e:
        print(f"Error applying dynamic EQ: {e}")
        return audio_clip


def augment_distortion(audio_clip, sample_rate=44100, drive_db_range=(0, 12)):
    """
    Apply subtle distortion to the audio.
    """
    drive_db = _get_random_value(drive_db_range)
    if drive_db < 0.1:
        return audio_clip

    board = Pedalboard([pedalboard.Distortion(drive_db=drive_db)])
    try:
        return board(audio_clip, sample_rate)
    except Exception as e:
        print(f"Error applying distortion: {e}")
        return audio_clip


# --- Spectrogram Augmentation ---


def augment_spectrogram_time_masking(
    spec, num_masks=1, max_mask_percentage=0.1, mask_value=None, overwrite=False
):
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
        mask_width = random.randint(1, max_mask_width + 1)

        # Determine start position
        if num_time_steps - mask_width < 0:
            continue  # Skip if mask is wider than spectrogram
        start_step = random.randint(0, num_time_steps - mask_width)

        # Apply mask
        spec[:, start_step : start_step + mask_width] = mask_value

    return spec


def augment_spectrogram_frequency_masking(
    spec, num_masks=1, max_mask_percentage=0.15, mask_value=None, overwrite=False
):
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
        mask_height = random.randint(1, max_mask_height + 1)

        # Determine start position
        if num_freq_bins - mask_height < 0:
            continue  # Skip if mask is taller than spectrogram
        start_bin = random.randint(0, num_freq_bins - mask_height)

        # Apply mask
        spec[start_bin : start_bin + mask_height, :] = mask_value

    return spec


# --- Deprecated/Modified Span Augmentation ---
# Note: The original augment_spectrogram_spans mixed time and frequency in a less standard way.
# Replaced with separate time and frequency masking functions above, following SpecAugment principles.
# Keeping the old helper function in case it's needed for the original logic, but recommend using the new ones.


def get_span_indices(dim, min_span=5, max_span=None, span_variation=0):
    """
    Helper function to find array indices for span augmentation (used by original augment_spectrogram_spans).
    Consider using the new time/frequency masking functions instead.

    dim: Integer representing the dimension of an array.
    min_span: Integer for minimum span length; may be less than min_span if lower bound is near 0.
    max_span: Integer for maximum span length, defaults to None. Uses lower bound to set max span.
    span_variation: Integer corresponding to variation to introduce into span lengths.

    Returns:
        Span indices [start_index, end_index]
    """

    # Add variation to min/max span lengths
    rand_min_span = min_span + random.randint(
        low=-span_variation, high=span_variation + 1
    )
    if rand_min_span <= 0:
        rand_min_span = 1

    rand_max_span = max_span
    if max_span is not None:
        rand_max_span = max_span + random.randint(
            low=-span_variation, high=span_variation + 1
        )
        if rand_max_span < rand_min_span:
            rand_max_span = rand_min_span  # Ensure max >= min
        if rand_max_span <= 0:
            rand_max_span = 1

    # Determine span length
    if rand_max_span is not None:
        span_len = random.randint(rand_min_span, rand_max_span + 1)
    else:
        span_len = rand_min_span

    # Ensure span length is not greater than dimension
    span_len = min(span_len, dim)
    if span_len <= 0:
        span_len = 1

    # Determine start position
    if (
        dim - span_len < 0
    ):  # Should not happen with the min(span_len, dim) above, but safe check
        start_ind = 0
        span_len = dim
    else:
        start_ind = random.randint(0, dim - span_len)

    end_ind = start_ind + span_len - 1  # Inclusive end index

    return [start_ind, end_ind]


# --- Visualization (Unchanged) ---
def compare_waveforms(
    df,
    i,
    signal_cols,
    signal_labs=None,
    sample_rate=44100,
    max_pts=None,
    alpha=0.5,
    fontsizes=[24, 18, 20],
    figsize=(16, 12),
    leg_loc="best",
    title="",
):
    # ... existing code ...
    # (No changes needed in this function based on the request)
    if signal_labs is None:
        signal_labs = signal_cols
    elif (len(signal_labs) < len(signal_cols)) or not isinstance(signal_labs, list):
        print(
            "Not enough labels were provided in list form. Using column names as labels."
        )
        signal_labs = signal_cols

    if isinstance(alpha, float):
        alpha = [alpha for s in signal_cols]
    elif len(alpha) < len(signal_cols):
        print(
            "Alpha list contained insufficient values, using first value for all signals."
        )
        alpha = [alpha[0] for s in signal_cols]

    plt.figure(figsize=figsize)

    for col, lab, alp in list(zip(signal_cols, signal_labs, alpha)):
        # Ensure data is float for waveplot
        audio_data = df.loc[i, col].astype(np.float32)
        librosa.display.waveshow(
            audio_data, sr=sample_rate, max_points=max_pts, label=lab, alpha=alp
        )

    if title == "":
        plt.title(
            "Comparison of audio clips for element: {}, label: {}".format(
                i, df.loc[i, "label"]
            ),
            fontsize=fontsizes[0],
        )
    else:
        plt.title(title, fontsize=fontsizes[0])

    plt.gca().xaxis.label.set_fontsize(fontsizes[1])
    plt.legend(signal_labs, loc=leg_loc, fontsize=fontsizes[2])


# --- Original augment_spectrogram_spans (Marked for review/deprecation) ---
def augment_spectrogram_spans(
    spec,
    spans=3,
    span_ranges=[[1, 4], [1, 6]],
    span_variation=1,
    ind_lists=None,
    sig_val=None,
    overwrite=False,
):
    """
    DEPRECATION WARNING: This function applies rectangular masks based on separate time/frequency span calculations.
    Consider using augment_spectrogram_time_masking and augment_spectrogram_frequency_masking instead
    for standard SpecAugment-style masking.

    Set spectrogram (or other 2-d numpy array) span(s) to background or another signal.

    Parameters:
        spec: Numpy array; presumed to be a spectrogram or other signal approprtiate for dropout augmentation.
        spans: Integer corresponding to the number of dropout spans to construct.
        span_ranges: List of lists corresponding to integers for x (time) and y (freq) dimensions used in spans.
        span_variation: Integer corresponding to variation to introduce into span lengths.
        ind_lists: List of lists corresponding to array indices to augment; By default, a list will be
            created.
        sig_val: value to set sub arrays to; minimum (background) value is used by default.
        overwrite: Boolean signifying whether to overwrite the input array.

    Returns:
        An augmented spectrogram (or other numpy array) with dropout spans applied.
    """
    if not overwrite:
        spec = spec.copy()

    if sig_val is None:  # Corrected check for None
        sig_val = spec.min()  # use for setting to background

    if not isinstance(spans, int):
        print(
            "Value supplied for spans supplied was not an integer. Setting spans to 1."
        )
        spans = 1

    if not ind_lists:
        ind_lists = []
        dims = spec.shape
        for i in range(0, spans):
            # Note: Original logic used get_span_indices for both time (dims[1]) and freq (dims[0])
            # This creates rectangular blocks, not just time or frequency stripes.
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
            ind_lists.append(
                [time_inds, freq_inds]
            )  # Store as [time_indices, freq_indices]

    for ind_list in ind_lists:
        time_inds = ind_list[0]
        freq_inds = ind_list[1]
        # Apply mask using frequency indices for rows, time indices for columns
        spec[freq_inds[0] : freq_inds[1] + 1, time_inds[0] : time_inds[1] + 1] = sig_val

    return spec


# --- Available augmentations dictionary (defined at end after all functions) ---
AVAILABLE_AUGMENTATIONS = {
    "add_lowpass_filter": add_lowpass_filter,
    "add_pedalboard_effects": add_pedalboard_effects,
    "add_white_noise": add_white_noise,
    "augment_pitch": augment_pitch,
    # Add other functions here if they are intended to be used via this helper
}
