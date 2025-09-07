import numpy as np
from chart_hero.utils import audio_io


class DummyProc:
    def __init__(self, data: bytes):
        self.stdout = data


def test_load_audio_ffmpeg(monkeypatch):
    data = np.array([0.1, 0.2], dtype=np.float32).tobytes()

    def fake_run(cmd, check, stdout):
        return DummyProc(data)

    monkeypatch.setattr(audio_io, "has_ffmpeg", lambda: True)
    monkeypatch.setattr(audio_io.subprocess, "run", fake_run)

    y, sr = audio_io.load_audio("dummy.wav", sr=22050, mono=True)
    assert sr == 22050
    assert np.allclose(y, np.array([0.1, 0.2], dtype=np.float32))


def test_load_audio_fallback_to_librosa(monkeypatch):
    expected = np.array([0.3, 0.4], dtype=np.float32)

    def fake_load(path, sr=None, mono=True):
        return expected, 44100

    monkeypatch.setattr(audio_io, "has_ffmpeg", lambda: False)
    monkeypatch.setattr(audio_io.librosa, "load", fake_load)

    y, sr = audio_io.load_audio("dummy.wav")
    assert sr == 44100
    assert np.allclose(y, expected)


def test_get_duration_ffprobe(monkeypatch):
    monkeypatch.setattr(audio_io, "has_ffprobe", lambda: True)

    def fake_check_output(cmd, text):
        return "1.23"

    monkeypatch.setattr(audio_io.subprocess, "check_output", fake_check_output)
    dur = audio_io.get_duration("dummy.wav")
    assert dur == 1.23


def test_get_duration_fallback(monkeypatch):
    monkeypatch.setattr(audio_io, "has_ffprobe", lambda: False)
    monkeypatch.setattr(audio_io.librosa, "get_duration", lambda path: 2.34)
    dur = audio_io.get_duration("dummy.wav")
    assert dur == 2.34
