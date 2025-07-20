import os

import requests

from chart_hero.inference.classes.audd import audd_song_response, audd_song_result


def identify_song(path: str) -> audd_song_result:
    api_token = os.environ.get("AUDD_API_TOKEN")
    if not api_token:
        raise ValueError("AUDD_API_TOKEN environment variable not set")

    data = {"api_token": api_token, "return": "musicbrainz,spotify,apple_music"}
    files = {"file": open(path, "rb")}
    response = requests.post("https://api.audd.io/", data=data, files=files)
    response.raise_for_status()  # Raise an exception for bad status codes
    response_json = response.json()

    if not response_json or not response_json.get("result"):
        raise ValueError("No result found in AudD API response")

    audd_response = audd_song_response.from_dict(response_json)
    if not audd_response or not audd_response.result:
        raise ValueError("No result found in AudD API response")

    return audd_response.result


def get_data_from_acousticbrainz(song: audd_song_result):
    print(song.musicbrainz[0].id)
    result = requests.get(
        "https://acousticbrainz.org/api/v1/low-level?recording_ids="
        + song.musicbrainz[0].id
    )

    return result.json()
