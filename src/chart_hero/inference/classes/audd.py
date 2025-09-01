from dataclasses import asdict, dataclass
from typing import Any, Optional, TypeVar

T = TypeVar("T")


@dataclass
class apple_music_preview:
    url: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "apple_music_preview":
        return cls(url=data["url"])


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "apple_music_artwork":
        return cls(
            width=data["width"],
            height=data["height"],
            url=data["url"],
            bgColor=data["bgColor"],
            textColor1=data["textColor1"],
            textColor2=data["textColor2"],
            textColor3=data["textColor3"],
            textColor4=data["textColor4"],
        )


@dataclass
class apple_music_play_params:
    id: str
    kind: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "apple_music_play_params":
        return cls(id=data["id"], kind=data["kind"])


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "apple_music_result":
        return cls(
            previews=[apple_music_preview.from_dict(item) for item in data["previews"]],
            artwork=apple_music_artwork.from_dict(data["artwork"]),
            artistName=data["artistName"],
            url=data["url"],
            discNumber=data["discNumber"],
            genreNames=data["genreNames"],
            durationInMillis=data["durationInMillis"],
            releaseDate=data["releaseDate"],
            name=data["name"],
            isrc=data["isrc"],
            albumName=data["albumName"],
            playParams=apple_music_play_params.from_dict(data["playParams"]),
            trackNumber=data["trackNumber"],
            composerName=data["composerName"],
        )


@dataclass
class spotify_external_urls:
    spotify: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "spotify_external_urls":
        return cls(spotify=data["spotify"])


@dataclass
class spotify_external_ids:
    isrc: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "spotify_external_ids":
        return cls(isrc=data["isrc"])


@dataclass
class spotify_artist:
    external_urls: spotify_external_urls
    href: str
    id: str
    name: str
    type: str
    uri: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "spotify_artist":
        return cls(
            external_urls=spotify_external_urls.from_dict(data["external_urls"]),
            href=data["href"],
            id=data["id"],
            name=data["name"],
            type=data["type"],
            uri=data["uri"],
        )


@dataclass
class spotify_image:
    height: int
    url: str
    width: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "spotify_image":
        return cls(height=data["height"], url=data["url"], width=data["width"])


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "spotify_album":
        return cls(
            album_type=data["album_type"],
            artists=[spotify_artist.from_dict(item) for item in data["artists"]],
            available_markets=data["available_markets"],
            external_urls=spotify_external_urls.from_dict(data["external_urls"]),
            href=data["href"],
            id=data["id"],
            images=[spotify_image.from_dict(item) for item in data["images"]],
            name=data["name"],
            release_date=data["release_date"],
            release_date_precision=data["release_date_precision"],
            total_tracks=data["total_tracks"],
            type=data["type"],
            uri=data["uri"],
        )


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "spotify_result":
        return cls(
            album=spotify_album.from_dict(data["album"]),
            artists=[spotify_artist.from_dict(item) for item in data["artists"]],
            available_markets=data["available_markets"],
            disc_number=data["disc_number"],
            duration_ms=data["duration_ms"],
            explicit=data["explicit"],
            external_ids=spotify_external_ids.from_dict(data["external_ids"]),
            external_urls=spotify_external_urls.from_dict(data["external_urls"]),
            href=data["href"],
            id=data["id"],
            is_local=data["is_local"],
            name=data["name"],
            popularity=data["popularity"],
            track_number=data["track_number"],
            type=data["type"],
            uri=data["uri"],
        )


@dataclass
class artist:
    id: str
    name: str
    sort_name: str
    disambiguation: Optional[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "artist":
        return cls(
            id=data["id"],
            name=data["name"],
            sort_name=data["sort_name"],
            disambiguation=data.get("disambiguation"),
        )


@dataclass
class artist_credit:
    artist: artist
    name: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "artist_credit":
        return cls(artist=artist.from_dict(data["artist"]), name=data["name"])


@dataclass
class track:
    id: str
    length: int
    number: int
    title: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "track":
        return cls(
            id=data["id"],
            length=data["length"],
            number=data["number"],
            title=data["title"],
        )


@dataclass
class media:
    format: str
    position: int
    track: list[track]
    track_count: int
    track_offset: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "media":
        return cls(
            format=data["format"],
            position=data["position"],
            track=[track.from_dict(item) for item in data["track"]],
            track_count=data["track_count"],
            track_offset=data["track_offset"],
        )


@dataclass
class area:
    id: str
    iso_3166_1_codes: list[str]
    name: str
    sort_name: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "area":
        return cls(
            id=data["id"],
            iso_3166_1_codes=data["iso_3166_1_codes"],
            name=data["name"],
            sort_name=data["sort_name"],
        )


@dataclass
class release_event:
    area: area
    date: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "release_event":
        return cls(area=area.from_dict(data["area"]), date=data["date"])


@dataclass
class release_group:
    id: str
    primary_type: str
    secondary_types: list[str]
    title: str
    type_id: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "release_group":
        return cls(
            id=data["id"],
            primary_type=data["primary_type"],
            secondary_types=data["secondary_types"],
            title=data["title"],
            type_id=data["type_id"],
        )


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "release":
        return cls(
            artist_credit=[
                artist_credit.from_dict(item) for item in data["artist_credit"]
            ],
            count=data["count"],
            country=data["country"],
            date=data["date"],
            disambiguation=data["disambiguation"],
            id=data["id"],
            media=[media.from_dict(item) for item in data["media"]],
            release_events=[
                release_event.from_dict(item) for item in data["release_events"]
            ]
            if data.get("release_events")
            else None,
            release_group=release_group.from_dict(data["release_group"]),
            status=data["status"],
            title=data["title"],
            track_count=data["track_count"],
        )


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "musicbrainz_result":
        return cls(
            artist_credit=[
                artist_credit.from_dict(item) for item in data["artist_credit"]
            ],
            disambiguation=data["disambiguation"],
            id=data["id"],
            isrcs=data["isrcs"],
            length=data["length"],
            releases=[release.from_dict(item) for item in data["releases"]],
            score=data["score"],
            tags=data.get("tags"),
            title=data["title"],
            video=data.get("video"),
        )


@dataclass
class audd_song_result:
    artist: str
    title: str
    album: str
    release_date: str
    label: str
    timecode: str
    song_link: str
    apple_music: Optional[apple_music_result]
    spotify: Optional[spotify_result]
    musicbrainz: list[musicbrainz_result]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "audd_song_result":
        return cls(
            artist=data["artist"],
            title=data["title"],
            album=data["album"],
            release_date=data["release_date"],
            label=data["label"],
            timecode=data["timecode"],
            song_link=data["song_link"],
            apple_music=apple_music_result.from_dict(data["apple_music"])
            if data.get("apple_music")
            else None,
            spotify=spotify_result.from_dict(data["spotify"])
            if data.get("spotify")
            else None,
            musicbrainz=[
                musicbrainz_result.from_dict(item) for item in data["musicbrainz"]
            ],
        )


@dataclass
class audd_song_response:
    status: str
    result: Optional[audd_song_result]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "audd_song_response":
        return cls(
            status=data["status"],
            result=audd_song_result.from_dict(data["result"])
            if data.get("result")
            else None,
        )
