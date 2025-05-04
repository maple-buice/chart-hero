import json
import os
import requests

from inference.classes.audd import audd_song_response, audd_song_result
from tokens import get_audd_token


def identify_song(path: str) -> audd_song_result:
    # data = {
    #     'api_token': get_audd_token(),
    #     'url': 'https://audd.tech/example.mp3',
    #     'return': 'musicbrainz'
    # }
    # files = {
    #     'file': open(path, 'rb')
    # }
    # response = requests.post('https://api.audd.io/', data=data, files=files)
    # response_json = response.json()
    
    response_json: str
    with open('audd_result.json', 'r') as file:
        response_json = file.read()

    response_object: audd_song_response = json.loads(response_json)
    result = response_object['result']
    # print(result['artist'])
    # print(result['musicbrainz'])
    return result

def get_data_from_acousticbrainz(song: audd_song_result):
    print(song['musicbrainz'][0]['id'])
    result = requests.get('https://acousticbrainz.org/api/v1/low-level?recording_ids=' + song['musicbrainz'][0]['id'])

    return result.json()
    