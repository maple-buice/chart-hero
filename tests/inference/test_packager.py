import numpy as np
import pytest
import soundfile as sf
from pathlib import Path

from chart_hero.inference import packager


def _write_wav(path: Path, sr: int = 44100, seconds: float = 0.1) -> Path:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(str(path), y, sr)
    return path


def test_convert_to_ogg_ffmpeg(monkeypatch, tmp_path):
    src = _write_wav(tmp_path / "src.wav")
    dst = tmp_path / "out.ogg"

    monkeypatch.setattr(packager, "which", lambda cmd: "/usr/bin/ffmpeg")

    def fake_run(cmd, check):
        in_path = Path(cmd[cmd.index("-i") + 1])
        out_path = Path(cmd[-1])
        sr = int(cmd[cmd.index("-ar") + 1])
        y, _ = sf.read(str(in_path))
        sf.write(str(out_path), y, sr, format="OGG", subtype="VORBIS")

    monkeypatch.setattr(packager.subprocess, "run", fake_run)

    def fake_get_duration(path):
        info = sf.info(path)
        return info.frames / info.samplerate

    monkeypatch.setattr(packager, "get_duration", fake_get_duration)

    out_path, dur = packager.convert_to_ogg(src, dst)
    info = sf.info(str(out_path))
    assert out_path == dst
    assert info.format == "OGG"
    assert info.samplerate == 44100
    assert pytest.approx(dur, abs=1e-6) == info.frames / info.samplerate


def test_convert_to_ogg_fallback(monkeypatch, tmp_path):
    src = tmp_path / "src.wav"
    dst = tmp_path / "out.ogg"
    y = np.random.randn(4410).astype(np.float32)

    monkeypatch.setattr(packager, "which", lambda cmd: None)
    monkeypatch.setattr(packager, "load_audio", lambda path, sr, mono: (y, 44100))

    out_path, dur = packager.convert_to_ogg(src, dst)
    info = sf.info(str(out_path))
    assert out_path == dst
    assert info.format == "OGG"
    assert info.samplerate == 44100
    assert pytest.approx(dur, abs=1e-6) == len(y) / 44100
