import pytest
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
