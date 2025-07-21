import sys
from unittest.mock import patch

import torch
import torchaudio
from chart_hero.inference.classes.audd import audd_song_result
from chart_hero.main import main


def test_main_script(tmp_path, monkeypatch):
    """
    Test that the main.py script runs without errors.
    """
    # Create a dummy audio file
    dummy_audio_path = tmp_path / "test.wav"
    sample_rate = 44100
    dummy_audio = torch.randn(1, sample_rate * 5)  # 5 seconds of audio
    torchaudio.save(str(dummy_audio_path), dummy_audio, sample_rate)

    # Path to the dummy model
    dummy_model_path = "tests/assets/dummy_model.ckpt"

    # Create dummy demucs model directory and file
    demucs_dir = tmp_path / "demucs"
    demucs_dir.mkdir()
    (demucs_dir / "83fc094f.th").touch()

    # Mock the command line arguments
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "src/chart_hero/main.py",
            "-p",
            str(dummy_audio_path),
            "-km",
            "speed",
            "--model-path",
            dummy_model_path,
            "-o",
            str(tmp_path),
        ],
    )

    with patch("chart_hero.main.identify_song") as mock_identify_song:
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
        with patch(
            "chart_hero.main.get_data_from_acousticbrainz"
        ) as mock_get_data_from_acousticbrainz:
            mock_get_data_from_acousticbrainz.return_value = {}
            with patch("chart_hero.main.audio_to_tensors") as mock_audio_to_tensors:
                mock_audio_to_tensors.return_value = [torch.randn(1, 1, 100, 128)]
                with patch("chart_hero.main.Charter") as mock_charter:
                    mock_charter.return_value.predict.return_value = "dummy_chart"
                    with patch(
                        "chart_hero.main.ChartGenerator"
                    ) as mock_chart_generator:
                        # Run the main function
                        main()

                        # Check that the charter was called
                        mock_charter.assert_called_once()
                        mock_chart_generator.assert_called_once()
