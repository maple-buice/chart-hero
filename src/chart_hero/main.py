import argparse
import json
import os
from pathlib import Path
from shutil import which
from typing import Optional

import librosa
import numpy as np

from chart_hero.inference.artwork import generate_art
from chart_hero.inference.charter import Charter, ChartGenerator
from chart_hero.inference.input_transform import audio_to_tensors, get_yt_audio
from chart_hero.inference.packager import package_clonehero_song
from chart_hero.inference.song_identifier import (
    get_data_from_acousticbrainz,
    identify_song,
)
from chart_hero.inference.types import PredictionRow
from chart_hero.model_training.transformer_config import get_config


def _load_env_local(path: Path) -> None:
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.strip().startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


def estimate_bpm(path: str, sr: int) -> Optional[float]:
    try:
        y, s = librosa.load(path, sr=sr)
        # Use librosa.beat.tempo which returns array of tempi; pick the first
        tempo = librosa.beat.tempo(y=y, sr=s, hop_length=512, aggregate=None)
        if tempo is None or len(tempo) == 0:
            return None
        # Robust statistic: median of local tempos
        return float(np.median(tempo))
    except Exception:
        return None


def _ensure_ffmpeg_available() -> bool:
    missing: list[str] = []
    if which("ffmpeg") is None:
        missing.append("ffmpeg")
    if which("ffprobe") is None:
        missing.append("ffprobe")
    if missing:
        print(
            "Missing required tools: "
            + ", ".join(missing)
            + ".\nPlease install ffmpeg (includes ffprobe) and ensure both are on your PATH.\n"
            + "macOS: brew install ffmpeg | Debian/Ubuntu: sudo apt-get install -y ffmpeg"
        )
        return False
    return True


def main() -> None:
    """
    Main function to run the drum transcription and charting process.
    """
    parser = argparse.ArgumentParser(
        description="Transcribe and chart drum patterns from an audio file."
    )
    # Exactly one of --path or --link must be provided
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("-p", "--path", type=str, help="Path to the audio file.")
    src_group.add_argument("-l", "--link", type=str, help="Link to a youtube video.")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Do not use or update the YouTube audio cache; force a fresh temporary download."
        ),
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
        default=192,
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
    parser.add_argument(
        "--export-clonehero",
        action="store_true",
        help="Export a Clone Hero-ready folder with notes.chart and song.ini.",
    )
    parser.add_argument(
        "--to-clonehero",
        action="store_true",
        help="Write directly into CloneHero/Songs/Chart Hero",
    )
    parser.add_argument(
        "--no-art",
        action="store_true",
        help="Skip artwork generation",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary downloaded audio",
    )
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Do not convert audio to OGG (not recommended)",
    )
    parser.add_argument(
        "--bpm",
        type=float,
        default=None,
        help="Override BPM (skips estimation/API)",
    )

    args = parser.parse_args()

    # Load .env.local to get tokens if present
    _load_env_local(Path(".env.local"))

    yt_info = None
    if args.link is not None:
        # Ensure ffmpeg/ffprobe are available for YouTube audio extraction
        if not _ensure_ffmpeg_available():
            return
        yt_info = get_yt_audio(args.link, no_cache=bool(args.no_cache))
        if yt_info is None:
            print("Could not download audio from link.")
            return
        if getattr(yt_info, "from_cache", False):
            print(f"Using cached audio: {yt_info.path}")
        else:
            print(f"Downloaded audio: {yt_info.path}")
        f_path = yt_info.path
        title = yt_info.title or Path(f_path).stem
        artist = None
        thumb = yt_info.thumbnail_url
    else:
        f_path = args.path
        p = Path(f_path)
        title = p.stem
        artist = None
        thumb = None

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Prefer local BPM estimation unless overridden
    bpm: Optional[float] = args.bpm
    if bpm is None:
        bpm = estimate_bpm(f_path, sr=22050)

    audd_result = None
    # Use AudD primarily to enrich metadata on local files when available
    if not args.no_api and os.environ.get("AUDD_API_TOKEN"):
        try:
            audd_result = identify_song(f_path)
            with open(out_path / "audd_result.json", "w") as f:
                json.dump(audd_result.to_dict(), f, indent=4)
            if audd_result and not artist:
                artist = audd_result.artist
                # Prefer MusicBrainz title if available
                if audd_result.title:
                    title = audd_result.title
            # Optionally fetch acousticbrainz but do not override BPM unless none
            if audd_result and audd_result.musicbrainz and bpm is None:
                acousticbrainz_result = get_data_from_acousticbrainz(audd_result)
                with open(out_path / "acousticbrainz_result.json", "w") as f:
                    json.dump(acousticbrainz_result, f, indent=4)
                bpm = acousticbrainz_result.get("bpm")
        except Exception as e:
            print(f"AudD lookup failed: {e}")
    # Fallback BPM
    if bpm is None:
        bpm = 120.0

    # Process the audio file to get the spectrogram tensors
    config = get_config("local")
    spectrogram_segments = audio_to_tensors(f_path, config)
    if not spectrogram_segments:
        print("Could not process audio file.")
        return

    # Load the charter with the model
    charter = Charter(config, args.model_path)

    # Predict the drum chart
    prediction_df = charter.predict(spectrogram_segments)

    # Create the chart
    chart_generator = ChartGenerator(
        prediction_df,
        song_duration=librosa.get_duration(path=f_path),
        bpm=bpm,
        sample_rate=config.sample_rate,
        song_title=title,
    )

    # Save the chart
    chart_generator.sheet.write("musicxml", fp=out_path / f"{title}.musicxml")
    print(f"Sheet music saved at {out_path}")

    if args.export_clonehero or args.to_clonehero:
        # Generate album/background art unless skipped
        album_path = None
        bg_path = None
        if not args.no_art:
            try:
                album_path, bg_path = generate_art(
                    out_path, title=title, artist=artist, thumbnail_url=thumb
                )
            except Exception:
                pass

        # Determine CH root relative to project
        ch_root = Path("CloneHero")
    # Coerce DataFrame rows into strictly-typed PredictionRow entries
    pred_rows: list[PredictionRow] = []
    for raw in chart_generator.df.to_dict(orient="records"):
        row: PredictionRow = {}
        for k, v in raw.items():
            key = str(k)
            val_int: int
            if isinstance(v, bool):
                val_int = int(v)
            elif isinstance(v, (int,)):
                val_int = int(v)
            else:
                try:
                    val_int = int(float(v))
                except Exception:
                    continue
            row[key] = val_int
        pred_rows.append(row)
        ch_dir = package_clonehero_song(
            clonehero_root=ch_root,
            title=title,
            artist=artist,
            bpm=float(bpm),
            resolution=args.resolution,
            sr_model=config.sample_rate,
            prediction_rows=pred_rows,
            source_audio=Path(f_path),
            album_path=album_path,
            background_path=bg_path,
            convert_audio=not args.no_convert,
        )
        print(f"Clone Hero chart exported to {ch_dir}")

    # Cleanup for no-cache temp download unless --keep-temp is set
    if (
        args.link is not None
        and getattr(args, "no_cache", False)
        and not args.keep_temp
        and yt_info is not None
    ):
        try:
            temp_dir = getattr(yt_info, "temp_dir", None)
            if temp_dir:
                td_path = Path(temp_dir)
                if td_path.exists():
                    for p in sorted(td_path.rglob("*"), reverse=True):
                        try:
                            p.unlink()
                        except Exception:
                            pass
                    try:
                        td_path.rmdir()
                    except Exception:
                        pass
        except Exception:
            pass


if __name__ == "__main__":
    main()
    main()
    main()
