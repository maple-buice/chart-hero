import os
from unittest.mock import patch

import pandas as pd

from chart_hero.__main__ import main


@patch("librosa.get_duration")
@patch("chart_hero.main.get_data_from_acousticbrainz")
@patch("chart_hero.main.identify_song")
@patch.dict(os.environ, {"AUDD_API_TOKEN": "dummy_token"})
@patch("sys.argv", ["chart-hero", "-p", "dummy_path"])
def test_main(mock_identify_song, mock_get_data_from_acousticbrainz, mock_get_duration):
    """Test that the main function calls the main function from main.py."""
    # Mock the identify_song function to return a dummy result
    mock_identify_song.return_value = {
        "title": "test_song",
        "artist": "test_artist",
        "musicbrainz": [{"id": "123"}],
    }
    mock_get_data_from_acousticbrainz.return_value = {"bpm": 120}
    mock_get_duration.return_value = 10

    # Patch the drum_extraction function to avoid a long-running process
    with patch("chart_hero.main.drum_extraction") as mock_drum_extraction:
        mock_drum_extraction.return_value = (None, None)
        # Patch the drum_to_frame function to avoid errors
        with patch("chart_hero.main.drum_to_frame") as mock_drum_to_frame:
            mock_drum_to_frame.return_value = (
                pd.DataFrame({"audio_clip": [], "sampling_rate": []}),
                None,
                None,
            )
            # Patch the DrumTranscriptionModule to avoid loading a real model
            with patch("chart_hero.main.DrumTranscriptionModule") as mock_dtm:
                # Patch the drum_charter to avoid creating a real chart
                with patch("chart_hero.main.drum_charter") as mock_dc:
                    main()
                    mock_identify_song.assert_called_once()
                    mock_get_data_from_acousticbrainz.assert_called_once()
                    mock_drum_extraction.assert_called_once()
                    mock_drum_to_frame.assert_called_once()
                    mock_dtm.load_from_checkpoint.assert_called_once()
                    mock_dc.assert_called_once()
