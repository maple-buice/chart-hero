import os
from unittest.mock import patch

import pytest
import requests

from chart_hero.inference.song_identifier import (
    get_data_from_acousticbrainz,
    identify_song,
)


@patch("requests.post")
def test_identify_song_success(mock_post):
    """
    Test that identify_song correctly processes a successful API response.
    """
    # Mock the requests.post call to return a successful response
    mock_response = mock_post.return_value
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "success",
        "result": {
            "artist": "Test Artist",
            "title": "Test Song",
            "album": "Test Album",
            "musicbrainz": [{"id": "test_mbid"}],
        },
    }

    # Create a dummy file for the test
    dummy_file_path = "test.mp3"
    with open(dummy_file_path, "w") as f:
        f.write("test")

    # Set the dummy API token
    os.environ["AUDD_API_TOKEN"] = "test_token"

    # Call the function
    result = identify_song(dummy_file_path)

    # Assert that the function returns the correct result
    assert result["artist"] == "Test Artist"
    assert result["title"] == "Test Song"

    # Clean up the dummy file
    os.remove(dummy_file_path)


@patch("requests.post")
def test_identify_song_no_result(mock_post):
    """
    Test that identify_song raises a ValueError when the API response has no result.
    """
    # Mock the requests.post call to return a response with no result
    mock_response = mock_post.return_value
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "success", "result": None}

    # Create a dummy file for the test
    dummy_file_path = "test.mp3"
    with open(dummy_file_path, "w") as f:
        f.write("test")

    # Set the dummy API token
    os.environ["AUDD_API_TOKEN"] = "test_token"

    # Assert that a ValueError is raised
    with pytest.raises(ValueError, match="No result found in AudD API response"):
        identify_song(dummy_file_path)

    # Clean up the dummy file
    os.remove(dummy_file_path)


@patch("requests.post")
def test_identify_song_api_error(mock_post):
    """
    Test that identify_song raises an HTTPError for a bad API response.
    """
    # Mock the requests.post call to return an error status code
    mock_response = mock_post.return_value
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError

    # Create a dummy file for the test
    dummy_file_path = "test.mp3"
    with open(dummy_file_path, "w") as f:
        f.write("test")

    # Set the dummy API token
    os.environ["AUDD_API_TOKEN"] = "test_token"

    # Assert that an HTTPError is raised
    with pytest.raises(requests.exceptions.HTTPError):
        identify_song(dummy_file_path)

    # Clean up the dummy file
    os.remove(dummy_file_path)


def test_identify_song_no_api_token():
    """
    Test that identify_song raises a ValueError when the AUDD_API_TOKEN is not set.
    """
    # Unset the API token
    if "AUDD_API_TOKEN" in os.environ:
        del os.environ["AUDD_API_TOKEN"]

    # Assert that a ValueError is raised
    with pytest.raises(ValueError, match="AUDD_API_TOKEN environment variable not set"):
        identify_song("test.mp3")


@patch("requests.get")
def test_get_data_from_acousticbrainz_success(mock_get):
    """
    Test that get_data_from_acousticbrainz correctly processes a successful API response.
    """
    # Mock the requests.get call to return a successful response
    mock_response = mock_get.return_value
    mock_response.status_code = 200
    mock_response.json.return_value = {"bpm": 120}

    # Create a dummy song result
    dummy_song = {"musicbrainz": [{"id": "test_mbid"}]}

    # Call the function
    result = get_data_from_acousticbrainz(dummy_song)

    # Assert that the function returns the correct result
    assert result["bpm"] == 120
