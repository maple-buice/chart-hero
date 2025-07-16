import os

import requests

from chart_hero.inference.classes.audd import audd_song_result


def identify_song(path: str) -> audd_song_result:
    api_token = os.environ.get("AUDD_API_TOKEN")
    if not api_token:
        raise ValueError("AUDD_API_TOKEN environment variable not set")

    data = {"api_token": api_token, "return": "musicbrainz"}
    files = {"file": open(path, "rb")}
    response = requests.post("https://api.audd.io/", data=data, files=files)
    response.raise_for_status()  # Raise an exception for bad status codes
    response_json = response.json()

    result = response_json.get("result")
    if not result:
        raise ValueError("No result found in AudD API response")

    return result


def get_data_from_acousticbrainz(song: audd_song_result):
    print(song["musicbrainz"][0]["id"])
    result = requests.get(
        "https://acousticbrainz.org/api/v1/low-level?recording_ids="
        + song["musicbrainz"][0]["id"]
    )

    return result.json()
