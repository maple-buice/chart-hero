import numpy as np
import pandas as pd
from music21 import stream

from chart_hero.inference.charter import drum_charter
from chart_hero.model_training.transformer_config import get_drum_hits


def test_drum_charter_initialization():
    """Test that the drum_charter class can be initialized."""
    # Create a dummy prediction dataframe
    drum_hits = get_drum_hits()
    data = {
        "peak_sample": [1000, 2000, 3000],
        **{hit: np.random.randint(0, 2, 3) for hit in drum_hits},
    }
    prediction_df = pd.DataFrame(data)

    # Create a drum_charter object
    charter = drum_charter(
        prediction_df=prediction_df,
        song_duration=10,
        bpm=120,
        sample_rate=22050,
    )

    assert charter is not None
    assert isinstance(charter.sheet, stream.Score)


def test_sheet_construction():
    """Test the sheet_construction method."""
    # Create a dummy prediction dataframe
    drum_hits = get_drum_hits()
    data = {"peak_sample": [1000, 2000, 3000], **{hit: [1, 0, 1] for hit in drum_hits}}
    prediction_df = pd.DataFrame(data)

    # Create a drum_charter object
    charter = drum_charter(
        prediction_df=prediction_df,
        song_duration=10,
        bpm=120,
        sample_rate=22050,
        song_title="Test Song",
    )

    # Check the sheet music
    assert charter.sheet.metadata.title == "Test Song"
    assert len(charter.sheet.parts) == 1
    assert len(charter.sheet.parts[0].getElementsByClass("Measure")) > 0
