import os
from unittest.mock import patch

import torch

from chart_hero.inference.classes.audd import audd_song_result
from chart_hero.main import main


@patch("librosa.get_duration")
@patch("chart_hero.main.ChartGenerator")
@patch("chart_hero.main.Charter")
@patch("chart_hero.main.audio_to_tensors")
@patch("chart_hero.main.get_data_from_acousticbrainz")
@patch("chart_hero.main.identify_song")
@patch.dict(os.environ, {"AUDD_API_TOKEN": "dummy_token"})
@patch("sys.argv", ["chart-hero", "-p", "dummy_path"])
def test_main(
    mock_identify_song,
    mock_get_data_from_acousticbrainz,
    mock_audio_to_tensors,
    mock_charter,
    mock_chart_generator,
    mock_get_duration,
):
    """Test that the main function calls the new pipeline correctly."""
    mock_identify_song.return_value = audd_song_result.from_dict(
        {
            "artist": "test_artist",
            "title": "test_song",
            "album": "test_album",
            "release_date": "2025-01-01",
            "label": "test_label",
            "timecode": "00:00",
            "song_link": "https://example.com",
            "apple_music": None,
            "spotify": None,
            "musicbrainz": [
                {
                    "id": "123",
                    "artist_credit": [],
                    "disambiguation": "",
                    "isrcs": [],
                    "length": 0,
                    "releases": [],
                    "score": 0,
                    "tags": [],
                    "title": "",
                    "video": None,
                }
            ],
        }
    )
    mock_get_data_from_acousticbrainz.return_value = {"bpm": 120}
    mock_audio_to_tensors.return_value = [torch.randn(1, 1, 100, 128)]
    mock_charter.return_value.predict.return_value = "dummy_chart"
    mock_get_duration.return_value = 180.0

    main()

    mock_identify_song.assert_called_once()
    mock_get_data_from_acousticbrainz.assert_called_once()
    mock_audio_to_tensors.assert_called_once()
    mock_charter.assert_called_once()
    mock_chart_generator.assert_called_once()
    mock_get_duration.assert_called_once()
