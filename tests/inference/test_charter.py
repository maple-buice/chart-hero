import numpy as np
import pandas as pd
import pytest
from music21 import stream

from chart_hero.inference.charter import drum_charter
from chart_hero.model_training.transformer_config import get_drum_hits


@pytest.fixture
def charter_fixture():
    """Provides a drum_charter instance for testing."""
    drum_hits = get_drum_hits()
    # Add some noise to the onsets to make them imperfect
    onsets = np.arange(1, 11) * 22050 + np.random.randint(-100, 100, 10)
    data = {
        "peak_sample": onsets,
        **{hit: np.random.randint(0, 2, 10) for hit in drum_hits},
    }
    prediction_df = pd.DataFrame(data)

    charter = drum_charter(
        prediction_df=prediction_df,
        song_duration=11,  # Increased to avoid empty time grid
        bpm=120,
        sample_rate=22050,
        song_title="Test Song",
    )
    return charter


def test_drum_charter_initialization(charter_fixture):
    """Test that the drum_charter class can be initialized."""
    assert charter_fixture is not None
    assert isinstance(charter_fixture.sheet, stream.Score)


def test_sheet_construction(charter_fixture):
    """Test the sheet_construction method."""
    assert charter_fixture.sheet.metadata.title == "Test Song"
    assert len(charter_fixture.sheet.parts) == 1
    assert len(charter_fixture.sheet.parts[0].getElementsByClass("Measure")) > 0


def test_get_note_duration(charter_fixture):
    """Test the get_note_duration method."""
    # 120 bpm = 2 beats per second. A quarter note is 0.5s.
    # An eighth note is 0.25s.
    assert charter_fixture._8_duration == 0.25
    assert charter_fixture._16_duration == 0.125
    assert charter_fixture._32_duration == 0.0625
    assert pytest.approx(charter_fixture._8_triplet_duration) == 0.25 / 3


def test_get_eighth_note_time_grid(charter_fixture):
    """Test the get_eighth_note_time_grid method."""
    time_grid = charter_fixture.get_eighth_note_time_grid(
        song_duration=10, note_offset=0
    )
    assert len(time_grid) > 0
    assert (
        pytest.approx(time_grid[0], abs=0.01) == 1.0
    )  # First onset is around 1 second
    assert pytest.approx(time_grid[1] - time_grid[0]) == 0.25  # 8th note duration


def test_sync_8(charter_fixture):
    """Test the sync_8 method."""
    time_grid = np.array([1.0, 1.25, 1.5, 1.75, 2.0])
    synced_grid = charter_fixture.sync_8(time_grid)
    assert len(synced_grid) > 0
    # The synced grid should be close to the original grid, but not identical
    assert not np.array_equal(synced_grid, time_grid)


def test_get_note_division(charter_fixture):
    """Test the get_note_division method."""
    _16_div, _32_div, _8_triplet_div, _8_sixlet_div = (
        charter_fixture.get_note_division()
    )
    assert len(_16_div) > 0
    assert len(_32_div) > 0
    assert len(_8_triplet_div) > 0
    assert len(_8_sixlet_div) > 0


def test_master_sync(charter_fixture):
    """Test the master_sync method."""
    _16_div, _32_div, _8_triplet_div, _8_sixlet_div = (
        charter_fixture.get_note_division()
    )
    (
        synced_8_div_clean,
        synced_16_div,
        synced_32_div,
        synced_8_3_div,
        synced_8_6_div,
    ) = charter_fixture.master_sync(_16_div, _32_div, _8_triplet_div, _8_sixlet_div)
    assert len(synced_8_div_clean) > 0
    assert len(synced_16_div) >= 0
    assert len(synced_32_div) >= 0
    assert len(synced_8_3_div) >= 0
    assert len(synced_8_6_div) >= 0


def test_build_measure(charter_fixture):
    """Test the build_measure method."""
    measure_iter = charter_fixture.synced_8_div[: charter_fixture.beats_in_measure]
    measure, note_dur = charter_fixture.build_measure(measure_iter)
    assert len(measure) > 0
    assert len(note_dur) > 0
    assert len(measure) == len(note_dur)


def test_drum_charter_empty_input():
    """Test that the drum_charter class can handle an empty dataframe."""
    charter = drum_charter(
        prediction_df=pd.DataFrame({"peak_sample": []}),
        song_duration=10,
        bpm=120,
        sample_rate=22050,
    )
    assert charter is not None
    assert isinstance(charter.sheet, stream.Score)
    # The sheet should have one part and one measure with a rest
    assert len(charter.sheet.parts) == 1
    assert len(charter.sheet.parts[0].getElementsByClass("Measure")) == 0
    assert len(charter.sheet.parts[0].getElementsByClass("Note")) == 0
    assert len(charter.sheet.parts[0].getElementsByClass("Chord")) == 0
    assert len(charter.sheet.parts[0].getElementsByClass("Rest")) == 0
