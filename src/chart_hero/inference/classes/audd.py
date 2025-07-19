from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Optional, Type, TypeVar

T = TypeVar("T")


def _from_dict(data_class: Type[T], data: Any) -> Optional[T]:
    if data is None:
        return None
    if is_dataclass(data_class):
        field_types = {f.name: f.type for f in data_class.__dataclass_fields__.values()}
        return data_class(
            **{
                key: _from_dict(field_types.get(key), value)  # type: ignore
                for key, value in data.items()
                if field_types.get(key) is not None
            }
        )
    if hasattr(data_class, "__origin__") and data_class.__origin__ is list:  # type: ignore
        return [_from_dict(data_class.__args__[0], item) for item in data]  # type: ignore
    return data


@dataclass
class apple_music_preview:
    url: str


@dataclass
class apple_music_artwork:
    width: int
    height: int
    url: str
    bgColor: str
    textColor1: str
    textColor2: str
    textColor3: str
    textColor4: str


@dataclass
class apple_music_play_params:
    id: str
    kind: str


@dataclass
class apple_music_result:
    previews: list[apple_music_preview]
    artwork: apple_music_artwork
    artistName: str
    url: str
    discNumber: int
    genreNames: list[str]
    durationInMillis: int
    releaseDate: str
    name: str
    isrc: str
    albumName: str
    playParams: apple_music_play_params
    trackNumber: int
    composerName: str


@dataclass
class spotify_external_urls:
    spotify: str


@dataclass
class spotify_external_ids:
    isrc: str


@dataclass
class spotify_artist:
    external_urls: spotify_external_urls
    href: str
    id: str
    name: str
    type: str
    uri: str


@dataclass
class spotify_image:
    height: int
    url: str
    width: int


@dataclass
class spotify_album:
    album_type: str
    artists: list[spotify_artist]
    available_markets: list[Any]
    external_urls: spotify_external_urls
    href: str
    id: str
    images: list[spotify_image]
    name: str
    release_date: str
    release_date_precision: str
    total_tracks: int
    type: str
    uri: str


@dataclass
class spotify_result:
    album: spotify_album
    artists: list[spotify_artist]
    available_markets: list[Any]
    disc_number: int
    duration_ms: int
    explicit: bool
    external_ids: spotify_external_ids
    external_urls: spotify_external_urls
    href: str
    id: str
    is_local: bool
    name: str
    popularity: int
    track_number: int
    type: str
    uri: str


@dataclass
class artist:
    id: str
    name: str
    sort_name: str
    disambiguation: Optional[str]


@dataclass
class artist_credit:
    artist: artist
    name: str


@dataclass
class track:
    id: str
    length: int
    number: int
    title: str


@dataclass
class media:
    format: str
    position: int
    track: list[track]
    track_count: int
    track_offset: int


@dataclass
class area:
    id: str
    iso_3166_1_codes: list[str]
    name: str
    sort_name: str


@dataclass
class release_event:
    area: area
    date: str


@dataclass
class release_group:
    id: str
    primary_type: str
    secondary_types: list[str]
    title: str
    type_id: str


@dataclass
class release:
    artist_credit: list[artist_credit]
    count: int
    country: str
    date: str
    disambiguation: str
    id: str
    media: list[media]
    release_events: Optional[list[release_event]]
    release_group: release_group
    status: str
    title: str
    track_count: int


@dataclass
class musicbrainz_result:
    artist_credit: list[artist_credit]
    disambiguation: str
    id: str
    isrcs: list[str]
    length: int
    releases: list[release]
    score: int
    tags: Optional[list[str]]
    title: str
    video: Optional[str]


@dataclass
class audd_song_result:
    artist: str
    title: str
    album: str
    release_date: str
    label: str
    timecode: str
    song_link: str
    apple_music: apple_music_result
    spotify: spotify_result
    musicbrainz: list[musicbrainz_result]


@dataclass
class audd_song_response:
    status: str
    result: audd_song_result

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[T], data: Any) -> Optional[T]:
        return _from_dict(cls, data)