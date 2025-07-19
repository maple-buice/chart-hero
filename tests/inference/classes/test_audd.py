from chart_hero.inference.classes.audd import audd_song_response


def test_audd_song_response_from_dict():
    """Test that the audd_song_response class can be created from a dictionary."""
    dummy_response = {
        "status": "success",
        "result": {
            "artist": "Test Artist",
            "title": "Test Song",
            "album": "Test Album",
            "release_date": "2025-01-01",
            "label": "Test Label",
            "timecode": "00:00",
            "song_link": "https://example.com",
            "apple_music": {
                "previews": [{"url": "https://example.com/preview"}],
                "artwork": {
                    "width": 100,
                    "height": 100,
                    "url": "https://example.com/artwork",
                    "bgColor": "ffffff",
                    "textColor1": "000000",
                    "textColor2": "000000",
                    "textColor3": "000000",
                    "textColor4": "000000",
                },
                "artistName": "Test Artist",
                "url": "https://example.com/apple_music",
                "discNumber": 1,
                "genreNames": ["Test Genre"],
                "durationInMillis": 1000,
                "releaseDate": "2025-01-01",
                "name": "Test Song",
                "isrc": "US1234567890",
                "albumName": "Test Album",
                "playParams": {"id": "123", "kind": "song"},
                "trackNumber": 1,
                "composerName": "Test Composer",
            },
            "spotify": {
                "album": {
                    "album_type": "album",
                    "artists": [
                        {
                            "external_urls": {
                                "spotify": "https://example.com/spotify_artist"
                            },
                            "href": "https://example.com/spotify_artist",
                            "id": "123",
                            "name": "Test Artist",
                            "type": "artist",
                            "uri": "spotify:artist:123",
                        }
                    ],
                    "available_markets": ["US"],
                    "external_urls": {"spotify": "https://example.com/spotify_album"},
                    "href": "https://example.com/spotify_album",
                    "id": "123",
                    "images": [
                        {
                            "height": 100,
                            "url": "https://example.com/spotify_image",
                            "width": 100,
                        }
                    ],
                    "name": "Test Album",
                    "release_date": "2025-01-01",
                    "release_date_precision": "day",
                    "total_tracks": 1,
                    "type": "album",
                    "uri": "spotify:album:123",
                },
                "artists": [
                    {
                        "external_urls": {
                            "spotify": "https://example.com/spotify_artist"
                        },
                        "href": "https://example.com/spotify_artist",
                        "id": "123",
                        "name": "Test Artist",
                        "type": "artist",
                        "uri": "spotify:artist:123",
                    }
                ],
                "available_markets": ["US"],
                "disc_number": 1,
                "duration_ms": 1000,
                "explicit": False,
                "external_ids": {"isrc": "US1234567890"},
                "external_urls": {"spotify": "https://example.com/spotify_song"},
                "href": "https://example.com/spotify_song",
                "id": "123",
                "is_local": False,
                "name": "Test Song",
                "popularity": 100,
                "track_number": 1,
                "type": "track",
                "uri": "spotify:track:123",
            },
            "musicbrainz": [
                {
                    "artist_credit": [
                        {
                            "artist": {
                                "id": "123",
                                "name": "Test Artist",
                                "sort_name": "Artist, Test",
                                "disambiguation": "Test Disambiguation",
                            },
                            "name": "Test Artist",
                        }
                    ],
                    "disambiguation": "Test Disambiguation",
                    "id": "123",
                    "isrcs": ["US1234567890"],
                    "length": 1000,
                    "releases": [
                        {
                            "artist_credit": [
                                {
                                    "artist": {
                                        "id": "123",
                                        "name": "Test Artist",
                                        "sort_name": "Artist, Test",
                                        "disambiguation": "Test Disambiguation",
                                    },
                                    "name": "Test Artist",
                                }
                            ],
                            "count": 1,
                            "country": "US",
                            "date": "2025-01-01",
                            "disambiguation": "Test Disambiguation",
                            "id": "123",
                            "media": [
                                {
                                    "format": "CD",
                                    "position": 1,
                                    "track": [
                                        {
                                            "id": "123",
                                            "length": 1000,
                                            "number": "1",
                                            "title": "Test Song",
                                        }
                                    ],
                                    "track_count": 1,
                                    "track_offset": 0,
                                }
                            ],
                            "release_events": [
                                {
                                    "area": {
                                        "id": "123",
                                        "iso_3166_1_codes": ["US"],
                                        "name": "United States",
                                        "sort_name": "United States",
                                    },
                                    "date": "2025-01-01",
                                }
                            ],
                            "release_group": {
                                "id": "123",
                                "primary_type": "Album",
                                "secondary_types": [],
                                "title": "Test Album",
                                "type_id": "123",
                            },
                            "status": "Official",
                            "title": "Test Album",
                            "track_count": 1,
                        }
                    ],
                    "score": 100,
                    "tags": [],
                    "title": "Test Song",
                    "video": None,
                }
            ],
        },
    }

    response = audd_song_response.from_dict(dummy_response)
    assert response is not None
    assert response.status == "success"
    assert response.result.artist == "Test Artist"
