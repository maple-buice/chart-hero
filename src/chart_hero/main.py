import argparse
import json
import os
from pathlib import Path

import librosa

from chart_hero.inference.charter import Charter, ChartGenerator
from chart_hero.inference.input_transform import (
    audio_to_tensors,
    get_yt_audio,
)
from chart_hero.inference.song_identifier import (
    get_data_from_acousticbrainz,
    identify_song,
)
from chart_hero.model_training.transformer_config import get_config


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
    parser.add_argument("-l", "--link", type=str, help="Link to a youtube video.")
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
        "-b",
        "--backtrack",
        action="store_true",
        help="Enable backtrack for onset detection.",
    )
    parser.add_argument(
        "-f",
        "--fixed-clip-length",
        action="store_true",
        help="Use fixed clip length for drum frames.",
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Run without making API calls to AudD.",
    )

    args = parser.parse_args()

    if args.link is not None:
        print(f"Downloading audio track from {args.link}")
        f_path, title = get_yt_audio(args.link)
        if f_path is None:
            print("Could not download audio from link.")
            return
    else:
        f_path = args.path
        title = Path(f_path).stem

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if args.no_api:
        audd_result = None
        bpm = 120
    else:
        # Identify song
        audd_result = identify_song(f_path)
        with open(out_path / "audd_result.json", "w") as f:
            json.dump(audd_result, f, indent=4)

        # Get acoustic data
        if audd_result and audd_result.musicbrainz:
            acousticbrainz_result = get_data_from_acousticbrainz(audd_result)
            with open(out_path / "acousticbrainz_result.json", "w") as f:
                json.dump(acousticbrainz_result, f, indent=4)
            bpm = acousticbrainz_result.get("bpm")
        else:
            bpm = None

    # Process the audio file to get the spectrogram tensors
    config = get_config("local")
    spectrogram_tensors = audio_to_tensors(f_path, config)
    if not spectrogram_tensors:
        print("Could not process audio file.")
        return

    # Load the charter with the model
    charter = Charter(config, args.model_path)

    # Predict the drum chart
    prediction_df = charter.predict(spectrogram_tensors)

    # Create the chart
    chart_generator = ChartGenerator(
        prediction_df,
        song_duration=librosa.get_duration(path=f_path),
        bpm=bpm,
        sample_rate=config.sample_rate,
        song_title=title,
    )

    # Save the chart
    chart_generator.sheet.write(
        "musicxml", fp=out_path / f"{title}.musicxml"
    )
    print(f"Sheet music saved at {out_path}")
    if args.link is not None:
        os.remove(f_path)


if __name__ == "__main__":
    main()
