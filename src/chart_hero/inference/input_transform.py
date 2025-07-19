import os

import librosa
import librosa.feature.rhythm
import numpy as np
import torch
from yt_dlp import YoutubeDL


def create_transient_enhanced_spectrogram(y, sr, n_fft, hop_length, n_mels):
    """
    Creates a mel spectrogram where transients are enhanced.
    This function MUST be identical to the one in data_preparation.py
    """
    # 1. Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # 2. Onset Strength Envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # 3. Align and Gate
    min_len = min(log_mel_spec.shape[1], len(onset_env))
    log_mel_spec = log_mel_spec[:, :min_len]
    onset_env = onset_env[:min_len]

    # Normalize onset envelope to [0, 1]
    if np.max(onset_env) > 0:
        onset_env = onset_env / np.max(onset_env)

    # Gate the spectrogram
    transient_enhanced_spec = log_mel_spec * onset_env

    return transient_enhanced_spec


def audio_to_tensors(audio_path: str, config) -> list[torch.Tensor]:
    """
    Transforms an audio file into a list of tensor segments for the model.
    """
    try:
        y, sr = librosa.load(audio_path, sr=config.sample_rate)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return []

    # Create the full transient-enhanced spectrogram
    full_spec = create_transient_enhanced_spectrogram(
        y=y,
        sr=sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
    )

    # Segment the spectrogram into chunks the model can handle
    segment_length_frames = int(
        config.max_audio_length * config.sample_rate / config.hop_length
    )

    tensors = []
    num_frames = full_spec.shape[1]
    for i in range(0, num_frames, segment_length_frames):
        end_frame = i + segment_length_frames
        if end_frame > num_frames:
            spec_segment = np.pad(
                full_spec[:, i:],
                ((0, 0), (0, end_frame - num_frames)),
                mode="constant",
                constant_values=np.min(full_spec),
            )
        else:
            spec_segment = full_spec[:, i:end_frame]

        tensors.append(torch.from_numpy(spec_segment).float().unsqueeze(0))

    return tensors


class yt_audio:
    path: str
    title: str
    description: str
    thumbnail_url: str

    def __init__(self, path: str, title: str, description: str, thumbnail_url: str):
        self.path = path
        self.title = title
        self.description = description
        self.thumbnail_url = thumbnail_url


def get_yt_audio(link) -> tuple[str | None, str | None]:
    download_path = "music/YouTube/"
    ydl_opts = {
        "format": "m4a/bestaudio/best",
        "outtmpl": download_path + "song.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
            }
        ],
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(link, download=True)
        return os.path.join(download_path, "song.m4a"), info.get(
            "title", "Unknown Title"
        )
