import pytest
from chart_hero.inference import lyrics
from chart_hero.inference.lyrics import Syllable, Word, Line, to_rb_tokens


def make_line():
    syllables = [
        Syllable(text="hel", t0=0.0, t1=0.25),
        Syllable(text="lo,", t0=0.25, t1=0.5),
    ]
    word1 = Word(text="hello,", t0=0.0, t1=0.5, syllables=syllables)
    word2 = Word(text="world", t0=0.5, t1=1.0, syllables=[
        Syllable(text="world", t0=0.5, t1=1.0)
    ])
    line = Line(text="hello, world", t0=0.0, t1=1.0, words=[word1, word2])
    return [line]


def test_to_rb_tokens_trailing_dash_and_punctuation():
    lines = make_line()
    tokens = [tok for _, tok in to_rb_tokens(lines)]
    assert tokens == ["hel-", "lo,", "world"]


def test_parse_vtt_time_formats_and_final_flush():
    vtt = (
        "WEBVTT\n\n"
        "01:00:00.000 --> 01:00:02.000\nfirst line\n\n"
        "00:01.000 --> 00:03.000\nsecond line"
    )
    lines = lyrics.parse_vtt(vtt)
    assert [ln.text for ln in lines] == ["first line", "second line"]
    assert lines[0].t0 == pytest.approx(3600.0)
    assert lines[1].t0 == pytest.approx(1.0)


def test_fetch_lrclib_search_duration_ms(monkeypatch):
    def fake_http(url: str, timeout: float = 10.0):
        return [{"syncedLyrics": "[00:00.00]x", "durationMs": "120000"}]

    monkeypatch.setattr(lyrics, "_http_json", fake_http)
    lrc = lyrics.fetch_lrclib_search(track="t", artist=None, album=None, duration=120.0)
    assert lrc == "[00:00.00]x"


def test_fetch_audd_unsynced_returns_first_hit(monkeypatch):
    def fake_http(url: str, timeout: float = 10.0):
        return {
            "result": [
                {"title": "A", "artist": "B", "text": "first"},
                {"title": "C", "artist": "D", "text": "second"},
            ]
        }

    monkeypatch.setattr(lyrics, "_http_json", fake_http)
    txt = lyrics.fetch_audd_unsynced_lyrics(title="t", artist="a", api_token="x")
    assert txt == "first"
