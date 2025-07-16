import numpy as np
import pytest

from chart_hero.model_training.augment_audio import (
    add_lowpass_filter,
    add_pedalboard_effects,
    add_white_noise,
    augment_pitch,
    augment_spectrogram_frequency_masking,
    augment_spectrogram_time_masking,
)


@pytest.fixture
def audio_clip():
    """Returns a dummy audio clip for testing."""
    return np.random.randn(44100)


@pytest.fixture
def spectrogram():
    """Returns a dummy spectrogram for testing."""
    return np.random.randn(128, 216)


def test_add_lowpass_filter(audio_clip):
    """Test the add_lowpass_filter function."""
    augmented_clip = add_lowpass_filter(audio_clip)
    assert augmented_clip.shape == audio_clip.shape
    assert not np.array_equal(augmented_clip, audio_clip)


def test_add_pedalboard_effects(audio_clip):
    """Test the add_pedalboard_effects function."""
    augmented_clip = add_pedalboard_effects(audio_clip)
    assert augmented_clip.shape == audio_clip.shape
    assert not np.array_equal(augmented_clip, audio_clip)


def test_add_white_noise(audio_clip):
    """Test the add_white_noise function."""
    augmented_clip = add_white_noise(audio_clip)
    assert augmented_clip.shape == audio_clip.shape
    assert not np.array_equal(augmented_clip, audio_clip)


def test_augment_pitch(audio_clip):
    """Test the augment_pitch function."""
    augmented_clip = augment_pitch(audio_clip)
    assert augmented_clip.shape == audio_clip.shape
    assert not np.array_equal(augmented_clip, audio_clip)


def test_augment_spectrogram_time_masking(spectrogram):
    """Test the augment_spectrogram_time_masking function."""
    augmented_spec = augment_spectrogram_time_masking(spectrogram)
    assert augmented_spec.shape == spectrogram.shape
    assert not np.array_equal(augmented_spec, spectrogram)
    # Check that some values have been masked
    assert np.min(augmented_spec) == np.min(spectrogram)


def test_augment_spectrogram_frequency_masking(spectrogram):
    """Test the augment_spectrogram_frequency_masking function."""
    augmented_spec = augment_spectrogram_frequency_masking(spectrogram)
    assert augmented_spec.shape == spectrogram.shape
    assert not np.array_equal(augmented_spec, spectrogram)
    # Check that some values have been masked
    assert np.min(augmented_spec) == np.min(spectrogram)
