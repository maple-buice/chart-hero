import argparse
import json
import os
from pathlib import Path

import librosa
import pandas as pd
import torch

from chart_hero.inference.charter import drum_charter
from chart_hero.inference.input_transform import (
    drum_extraction,
    drum_to_frame,
    get_yt_audio,
)
from chart_hero.inference.song_identifier import (
    get_data_from_acousticbrainz,
    identify_song,
)
from chart_hero.model_training.train_transformer import DrumTranscriptionModule
from chart_hero.model_training.transformer_config import get_config, get_drum_hits


def main():
    """
    Main function to run the drum transcription and charting process.
    """
    parser = argparse.ArgumentParser(
        description="Transcribe and chart drum patterns from an audio file."
    )
    parser.add_argument(
        "-p", "--path", type=str, required=True, help="Path to the audio file."
    )
    parser.add_argument(
        "-l", "--link", type=str, help="Link to a youtube video."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save the output files.",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        type=str,
        default="models/local_transformer_models/best_model.ckpt",
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "-km",
        "--kernel-mode",
        type=str,
        default="performance",
        choices=["speed", "performance"],
        help="Demucs kernel mode.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=16,
        help="Note resolution for the drum chart.",
    )
    parser.add_argument(
        "-b", "--backtrack", action="store_true", help="Enable backtrack for onset detection."
    )
    parser.add_argument(
        "-f",
        "--fixed-clip-length",
        action="store_true",
        help="Use fixed clip length for drum frames.",
    )

    args = parser.parse_args()

    if args.link is not None:
        print(f"Downloading audio track from {args.link}")
        f_path = get_yt_audio(args.link)
    else:
        f_path = args.path

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Identify song
    audd_result = identify_song(f_path)
    with open(out_path / "audd_result.json", "w") as f:
        json.dump(audd_result, f, indent=4)

    # Get acoustic data
    if audd_result and audd_result.get("musicbrainz"):
        mbid = audd_result["musicbrainz"][0]["id"]
        acousticbrainz_result = get_data_from_acousticbrainz(mbid)
        with open(out_path / "acousticbrainz_result.json", "w") as f:
            json.dump(acousticbrainz_result, f, indent=4)
        bpm = acousticbrainz_result.get("bpm")
    else:
        bpm = None

    # Drum extraction
    drum_track, sr = drum_extraction(
        f_path,
        mode=args.kernel_mode,
    )

    # Drum to frame
    df, sr, bpm = drum_to_frame(
        drum_track,
        sr,
        estimated_bpm=bpm,
        resolution=args.resolution,
        backtrack=args.backtrack,
        fixed_clip_length=args.fixed_clip_length,
    )

    # Load model
    config = get_config("local")
    model = DrumTranscriptionModule.load_from_checkpoint(args.model_path, config=config)
    model.eval()

    # Predict
    predictions = []
    for _, row in df.iterrows():
        spectrogram = torch.from_numpy(row["audio_clip"]).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = model(spectrogram)
        predictions.append(output["logits"].squeeze().numpy())

    df_pred = pd.DataFrame(predictions, columns=get_drum_hits())
    df_pred = pd.concat([df, df_pred], axis=1)

    # Create drum chart
    song_duration = librosa.get_duration(y=drum_track, sr=sr)
    sheet_music = drum_charter(
        df_pred,
        song_duration=song_duration,
        bpm=bpm,
        sample_rate=sr,
        song_title=audd_result.get("title"),
    )

    # Save chart
    sheet_music.chart.write(
        "musicxml.pdf", fp=out_path / f"{audd_result.get('title', 'chart')}.pdf"
    )
    print(f"Sheet music saved at {out_path}")
    if args.link is not None:
        os.remove(f_path)


if __name__ == "__main__":
    main()
