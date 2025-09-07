import argparse
import json
import os
from pathlib import Path
from shutil import which
from typing import Optional

import librosa
import numpy as np
from librosa.feature.rhythm import tempo as lr_tempo

from chart_hero.inference.artwork import generate_art
from chart_hero.inference.charter import Charter, ChartGenerator
from chart_hero.inference.input_transform import audio_to_tensors, get_yt_audio
from chart_hero.inference.lyrics import (
    get_synced_lyrics,
    to_rb_tokens,
    Lyrics,
    parse_lrc,
)
from chart_hero.inference.mid_export import write_notes_mid
from chart_hero.inference.mid_vocals import Phrase as VoxPhrase
from chart_hero.inference.mid_vocals import SyllableEvent as VoxSyllable
from chart_hero.inference.packager import package_clonehero_song
from chart_hero.inference.song_identifier import (
    get_data_from_acousticbrainz,
    identify_song,
    search_musicbrainz_recording,
    get_acousticbrainz_lowlevel_by_mbid,
    extract_bpm_from_acousticbrainz,
    identify_song_cached,
    search_musicbrainz_recording_cached,
    get_acousticbrainz_lowlevel_by_mbid_cached,
)
from chart_hero.inference.types import PredictionRow
from chart_hero.model_training.transformer_config import get_config
from chart_hero.utils.audio_io import get_duration, load_audio


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
        y, s = load_audio(path, sr=sr)
        # Use the updated API for tempo
        tempo = lr_tempo(y=y, sr=s, hop_length=512, aggregate=None)
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
        "--id-source",
        type=str,
        default="auto",
        choices=["auto", "audd", "musicbrainz"],
        help=(
            "Metadata/BPM source: 'auto' (use AudD if available else MusicBrainz), 'audd', or 'musicbrainz'."
        ),
    )
    parser.add_argument(
        "--id-cache",
        type=str,
        default=None,
        help="Path to cache identification lookups (AudD/MusicBrainz/AcousticBrainz) as JSON.",
    )
    parser.add_argument(
        "--lyrics-cache",
        type=str,
        default=None,
        help="Path to a JSON file to cache synced lyrics (LRC) across runs.",
    )
    parser.add_argument(
        "--export-clonehero",
        action="store_true",
        help="Export a Clone Hero-ready folder with notes.mid and song.ini.",
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
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        choices=[
            "local",
            "local_performance",
            "local_max_performance",
            "overnight_default",
            "local_highres",
            "local_micro",
            "cloud",
        ],
        help="Model config to use for inference; if omitted, attempts to adapt to checkpoint",
    )
    parser.add_argument(
        "--patch-stride",
        type=int,
        default=None,
        help="Conv stride (frames) for inference; smaller than patch_size yields higher timing resolution",
    )
    parser.add_argument(
        "--nms-k",
        type=int,
        default=None,
        help="Per-class NMS window in patches (odd number like 3/5)",
    )
    parser.add_argument(
        "--activity-gate",
        type=float,
        default=None,
        help="Drop ticks whose max prob < gate (0..1)",
    )
    parser.add_argument(
        "--cymbal-margin",
        type=float,
        default=None,
        help="Prefer cymbal if pc + margin >= pt; tom wins only if pt >= pc + tom_over_cymbal_margin",
    )
    parser.add_argument(
        "--tom-over-cymbal-margin",
        type=float,
        default=None,
        help="Require tom prob exceed cymbal by this margin to choose tom",
    )
    parser.add_argument(
        "--class-thresholds",
        type=str,
        default=None,
        help="Per-class thresholds '0=0.55,...,68=0.9'",
    )
    parser.add_argument(
        "--class-gains",
        type=str,
        default=None,
        help="Per-class probability multipliers '2=0.5,3=0.5,4=0.5,67=1.1'",
    )
    parser.add_argument(
        "--offsets-json",
        type=str,
        default=None,
        help="JSON with per-class time offsets (ms) to apply at decode",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["conservative", "aggressive"],
        help="Decode preset: conservative (fewer FPs) or aggressive (higher recall)",
    )
    parser.add_argument(
        "--onset-gate",
        type=float,
        default=None,
        help="Aux onset gate threshold (0..1) if model has onset head",
    )
    parser.add_argument(
        "--clonehero-root",
        type=str,
        default=None,
        help="Clone Hero install root directory (contains 'Songs')",
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

    # Prefer local BPM estimation unless overridden; aim to avoid API calls
    bpm: Optional[float] = args.bpm

    # Improved local tempo estimate with histogram confidence
    def _estimate_bpm_local(path: str, sr: int) -> tuple[Optional[float], float]:
        try:
            import numpy as _np

            y, s = load_audio(path, sr=sr)
            # Beat track to get beat frames and intervals
            beats = librosa.beat.beat_track(y=y, sr=s, units="frames")[1]
            if beats is not None and len(beats) >= 3:
                times = librosa.frames_to_time(beats, sr=s)
                iois = _np.diff(times)
                iois = iois[(iois > 0.1) & (iois < 2.0)]  # clamp 30–600 BPM
                if iois.size > 0:
                    bpms = 60.0 / iois
                    # Histogram in 0.5 BPM bins
                    hist, edges = _np.histogram(bpms, bins=_np.arange(30, 241, 0.5))
                    idx = int(hist.argmax())
                    mode_bpm = float((edges[idx] + edges[idx + 1]) / 2.0)
                    conf = float(hist.max() / max(1, hist.sum()))
                    return mode_bpm, conf
            # Fallback to librosa.tempo median when beat track insufficient
            tempo = lr_tempo(y=y, sr=s, hop_length=512, aggregate=None)
            if tempo is not None and len(tempo) > 0:
                return float(np.median(tempo)), 0.3
        except Exception:
            return None, 0.0
        return None, 0.0

    if bpm is None:
        bpm_local, bpm_conf = _estimate_bpm_local(f_path, sr=22050)
        bpm = bpm_local if bpm_local else None

    audd_result = None
    # Metadata/BPM identification pipeline (AudD or MusicBrainz/AcousticBrainz)
    try:
        use_audd = (
            args.id_source in ("auto", "audd")
            and (not args.no_api)
            and os.environ.get("AUDD_API_TOKEN")
        )
        used_mb = False
        if use_audd:
            try:
                id_cache_path = args.id_cache or str(
                    Path("artifacts") / "id_cache.json"
                )
                audd_result = identify_song_cached(f_path, id_cache_path)
                with open(out_path / "audd_result.json", "w") as f:
                    json.dump(audd_result.to_dict(), f, indent=4)
                if audd_result and not artist:
                    artist = audd_result.artist
                    # Prefer MusicBrainz title if available
                    if audd_result.title:
                        title = audd_result.title
                # Optionally fetch AB BPM via MBID
                if audd_result and audd_result.musicbrainz:
                    try:
                        ab = get_data_from_acousticbrainz(audd_result)
                        with open(out_path / "acousticbrainz_result.json", "w") as f:
                            json.dump(ab, f, indent=4)
                        bpm_ab = extract_bpm_from_acousticbrainz(ab)
                        if bpm_ab:
                            bpm = bpm or bpm_ab
                    except Exception:
                        pass
            except Exception as e:
                print(f"AudD lookup failed: {e}")
        # If MB requested or AudD unavailable/failed, try MusicBrainz + AcousticBrainz
        if (args.id_source in ("auto", "musicbrainz")) and (bpm is None):
            yt_title = title or ""
            if not artist and (" - " in yt_title or " – " in yt_title):
                for sep in (" - ", " – "):
                    if sep in yt_title:
                        parts = yt_title.split(sep, 1)
                        if len(parts) == 2:
                            artist = artist or parts[0].strip()
                            title = parts[1].strip() or title
                            break
            id_cache_path = args.id_cache or str(Path("artifacts") / "id_cache.json")
            mb = search_musicbrainz_recording_cached(
                title=title or yt_title,
                artist=artist,
                duration_sec=get_duration(f_path),
                cache_path=id_cache_path,
            )
            if isinstance(mb, dict) and mb.get("id"):
                used_mb = True
                try:
                    with open(out_path / "musicbrainz_result.json", "w") as f:
                        json.dump(mb, f, indent=2)
                except Exception:
                    pass
                ab = get_acousticbrainz_lowlevel_by_mbid_cached(
                    str(mb.get("id")), id_cache_path
                )
                if isinstance(ab, dict):
                    try:
                        with open(out_path / "acousticbrainz_result.json", "w") as f:
                            json.dump(ab, f, indent=2)
                    except Exception:
                        pass
                    bpm_ab = extract_bpm_from_acousticbrainz(ab)
                    if bpm_ab:
                        bpm = bpm_ab
            # Fill metadata from MB if missing
            try:
                if used_mb and isinstance(mb, dict):
                    if not artist:
                        ac = mb.get("artist-credit") or mb.get("artist_credit")
                        if isinstance(ac, list) and ac and isinstance(ac[0], dict):
                            nm = ac[0].get("name") or ac[0].get("artist", {}).get(
                                "name"
                            )
                            if nm:
                                artist = str(nm)
                    if not title and isinstance(mb.get("title"), str):
                        title = str(mb.get("title"))
            except Exception:
                pass
    except Exception:
        pass
    # Fallback BPM
    if bpm is None:
        bpm = 120.0

    # Build config (default local), then adapt to checkpoint if present
    config = get_config(args.config or "local")
    # If a checkpoint is provided, try to import its structural hyperparameters
    try:
        import torch as _torch

        if args.model_path and Path(args.model_path).exists():
            ckpt = _torch.load(args.model_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            # Apply hparams dictionary when available
            hparams = ckpt.get("hyper_parameters") or ckpt.get("hparams")
            if isinstance(hparams, dict):
                for k in (
                    "sample_rate",
                    "n_mels",
                    "n_fft",
                    "hop_length",
                    "hidden_size",
                    "num_layers",
                    "num_heads",
                    "intermediate_size",
                    "patch_size",
                    "patch_stride",
                    "enable_onset_head",
                ):
                    if k in hparams:
                        try:
                            setattr(config, k, hparams[k])
                        except Exception:
                            pass
            # Fallback: infer from weights
            pe = state.get("model.patch_embed.projection.weight")
            if pe is not None and hasattr(pe, "shape") and len(pe.shape) == 3:
                embed_dim, n_mels, patch_time = (
                    int(pe.shape[0]),
                    int(pe.shape[1]),
                    int(pe.shape[2]),
                )
                try:
                    config.hidden_size = embed_dim
                    config.n_mels = n_mels
                    pt = list(getattr(config, "patch_size", (patch_time, 16)))
                    pt[0] = patch_time
                    config.patch_size = tuple(pt)  # type: ignore[assignment]
                except Exception:
                    pass
    except Exception:
        pass
    # Apply balanced inference defaults unless overridden by CLI
    try:
        if getattr(config, "activity_gate", None) is None:
            config.activity_gate = 0.50
        # Strengthen per-class NMS for fewer duplicates
        config.event_nms_kernel_patches = int(
            max(1, getattr(config, "event_nms_kernel_patches", 3))
        )
        if config.event_nms_kernel_patches < 9:
            config.event_nms_kernel_patches = 9
        if getattr(config, "cymbal_margin", None) is None:
            config.cymbal_margin = 0.30
        if float(getattr(config, "tom_over_cymbal_margin", 0.35)) < 0.45:
            config.tom_over_cymbal_margin = 0.45
        # Set per-class thresholds/gains if not already set
        if getattr(config, "class_thresholds", None) is None:
            from chart_hero.model_training.transformer_config import get_drum_hits

            thr_map = {
                "0": 0.55,
                "1": 0.62,
                "2": 0.60,
                "3": 0.60,
                "4": 0.65,
                "66": 0.86,
                "67": 0.88,
                "68": 0.92,
            }
            classes = get_drum_hits()
            config.class_thresholds = [
                float(thr_map.get(c, config.prediction_threshold)) for c in classes
            ]
        if getattr(config, "class_gains", None) is None:
            from chart_hero.model_training.transformer_config import get_drum_hits

            gn_map = {
                "0": 1.00,
                "1": 1.02,
                "2": 1.10,
                "3": 1.10,
                "4": 1.05,
                "66": 1.04,
                "67": 1.06,
                "68": 0.92,
            }
            classes = get_drum_hits()
            config.class_gains = [float(gn_map.get(c, 1.0)) for c in classes]
        # Apply preset overrides if requested
        if args.preset:
            if args.preset == "conservative":
                config.activity_gate = max(
                    0.65, float(getattr(config, "activity_gate", 0.5) or 0.5)
                )
                config.event_nms_kernel_patches = max(
                    11, int(getattr(config, "event_nms_kernel_patches", 9))
                )
                # Strengthen arbitration margins and onset gating (if onset head exists)
                try:
                    config.cymbal_margin = max(
                        0.40, float(getattr(config, "cymbal_margin", 0.1) or 0.1)
                    )
                    config.tom_over_cymbal_margin = max(
                        0.55,
                        float(getattr(config, "tom_over_cymbal_margin", 0.35) or 0.35),
                    )
                    # Safe to set even if model lacks onset head; code checks presence
                    if getattr(config, "onset_gate_threshold", None) is None:
                        config.onset_gate_threshold = 0.65
                except Exception:
                    pass
                # Enforce conservative per-class min spacing (ms)
                try:
                    base_map = getattr(config, "min_spacing_ms_map", None) or {}
                    conservative_map = {
                        "0": 32.0,  # Kick
                        "1": 30.0,  # Snare
                        "2": 28.0,  # HiTom
                        "3": 30.0,  # MidTom
                        "4": 32.0,  # LowTom
                        "66": 26.0,  # Crash
                        "67": 28.0,  # HiHat
                        "68": 30.0,  # Ride
                    }
                    base_map.update(
                        {
                            k: max(base_map.get(k, 0.0), v)
                            for k, v in conservative_map.items()
                        }
                    )
                    config.min_spacing_ms_map = base_map
                    if getattr(config, "min_spacing_ms_default", None) is None:
                        config.min_spacing_ms_default = 24.0
                except Exception:
                    pass
                # Harden cymbal thresholds slightly if present
                if getattr(config, "class_thresholds", None):
                    lab = get_drum_hits()
                    thr = list(config.class_thresholds)
                    idx_map = {k: i for i, k in enumerate(lab)}
                    for k, v in {"66": 0.90, "67": 0.92, "68": 0.94}.items():
                        i = idx_map.get(k)
                        if i is not None:
                            thr[i] = max(thr[i], v)
                    config.class_thresholds = thr
            elif args.preset == "aggressive":
                config.activity_gate = min(
                    0.45, float(getattr(config, "activity_gate", 0.5) or 0.5)
                )
                config.event_nms_kernel_patches = min(
                    7, int(getattr(config, "event_nms_kernel_patches", 9))
                )
                if getattr(config, "class_thresholds", None):
                    lab = get_drum_hits()
                    thr = list(config.class_thresholds)
                    idx_map = {k: i for i, k in enumerate(lab)}
                    for k, v in {"66": 0.82, "67": 0.84, "68": 0.90}.items():
                        i = idx_map.get(k)
                        if i is not None:
                            thr[i] = min(thr[i], v)
                    config.class_thresholds = thr
    except Exception:
        pass
    if args.patch_stride is not None and args.patch_stride > 0:
        try:
            config.patch_stride = int(args.patch_stride)
        except Exception:
            pass
    if args.nms_k is not None:
        config.event_nms_kernel_patches = int(args.nms_k)
    if args.activity_gate is not None:
        config.activity_gate = float(args.activity_gate)
    if args.cymbal_margin is not None:
        config.cymbal_margin = float(args.cymbal_margin)
    if args.tom_over_cymbal_margin is not None:
        config.tom_over_cymbal_margin = float(args.tom_over_cymbal_margin)
    if args.onset_gate is not None:
        config.onset_gate_threshold = float(args.onset_gate)
    # Parse per-class thresholds and gains
    if args.class_thresholds:
        from chart_hero.model_training.transformer_config import get_drum_hits

        mapping = {}
        for kv in args.class_thresholds.split(","):
            kv = kv.strip()
            if not kv or "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            try:
                mapping[str(k.strip())] = float(v)
            except Exception:
                pass
        if mapping:
            classes = get_drum_hits()
            config.class_thresholds = [
                float(mapping.get(c, config.prediction_threshold)) for c in classes
            ]
    if args.class_gains:
        from chart_hero.model_training.transformer_config import get_drum_hits

        mapping = {}
        for kv in args.class_gains.split(","):
            kv = kv.strip()
            if not kv or "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            try:
                mapping[str(k.strip())] = float(v)
            except Exception:
                pass
        if mapping:
            classes = get_drum_hits()
            config.class_gains = [float(mapping.get(c, 1.0)) for c in classes]
    # Load per-class time offsets (ms) if provided
    if args.offsets_json:
        try:
            import json as _json
            from chart_hero.model_training.transformer_config import get_drum_hits

            with open(args.offsets_json, "r") as f:
                payload = _json.load(f)
            classes = get_drum_hits()
            m = payload.get("offsets_ms", payload)
            if isinstance(m, dict):
                lst = [float(m.get(c, 0.0)) for c in classes]
            elif isinstance(m, list):
                lst = [float(x) for x in m][: len(classes)]
            else:
                lst = None
            if lst and len(lst) == len(classes):
                config.class_time_offsets_ms = lst
        except Exception:
            pass
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
        song_duration=get_duration(f_path),
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

        # Determine CH root relative to project or override via CLI
        ch_root = (
            Path(args.clonehero_root) if args.clonehero_root else Path("CloneHero")
        )
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
            write_chart=False,
        )
        print(f"Clone Hero chart exported to {ch_dir}")

        # Lyrics: fetch synced lyrics
        try:
            duration_sec = float(get_duration(f_path))
        except Exception:
            duration_sec = None  # type: ignore[assignment]

        spotify_id = None
        if (
            "audd_result" in locals()
            and audd_result is not None
            and getattr(audd_result, "spotify", None)
        ):
            try:
                spotify_id = audd_result.spotify.id  # type: ignore[assignment]
            except Exception:
                spotify_id = None

        lyrics = None
        # Lyrics cache setup
        cache_key: Optional[str] = None
        cache_path: Optional[Path] = None
        try:
            cache_path = (
                Path(args.lyrics_cache)
                if args.lyrics_cache
                else (Path("artifacts") / "lyrics_cache.json")
            )
            if args.link:
                cache_key = f"yt:{args.link}"
            if spotify_id:
                cache_key = f"sp:{spotify_id}"
            if not cache_key and title:
                cache_key = f"meta:{artist or ''}|{title}"
        except Exception:
            cache_key = None
            cache_path = None
        try:
            # Logging inputs for lyrics fetch
            print(
                "Fetching lyrics with:",
                json.dumps(
                    {
                        "link": args.link,
                        "title": title,
                        "artist": artist,
                        "album": getattr(audd_result, "album", None)
                        if "audd_result" in locals() and audd_result
                        else None,
                        "duration": duration_sec,
                        "spotify_id": spotify_id,
                    },
                    ensure_ascii=False,
                ),
            )
            # Try cache first if available (LRC)
            cache_used = False
            if cache_key and cache_path and cache_path.exists():
                try:
                    with cache_path.open("r", encoding="utf-8") as f:
                        payload = json.load(f)
                    ent = payload.get(cache_key)
                    if isinstance(ent, dict) and isinstance(ent.get("raw_lrc"), str):
                        lrc = ent["raw_lrc"]
                        lines = parse_lrc(lrc)
                        if lines:
                            lyrics = Lyrics(
                                source=str(ent.get("source") or "cache"),
                                confidence=float(ent.get("confidence", 0.8)),
                                lines=lines,
                                raw_lrc=lrc,
                            )
                            cache_used = True
                            print("Loaded lyrics from cache.")
                except Exception:
                    pass
            if not cache_used:
                lyrics = get_synced_lyrics(
                    link=args.link,
                    title=title,
                    artist=artist,
                    album=getattr(audd_result, "album", None)
                    if "audd_result" in locals() and audd_result
                    else None,
                    duration=duration_sec,
                    spotify_id=spotify_id,
                )
                # Save to cache if we have raw LRC
                try:
                    if (
                        lyrics
                        and getattr(lyrics, "raw_lrc", None)
                        and cache_key
                        and cache_path is not None
                    ):
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        payload: dict = {}
                        if cache_path.exists():
                            try:
                                with cache_path.open("r", encoding="utf-8") as f:
                                    payload = json.load(f)
                            except Exception:
                                payload = {}
                        payload[cache_key] = {
                            "source": lyrics.source,
                            "confidence": lyrics.confidence,
                            "raw_lrc": lyrics.raw_lrc,
                            "title": title,
                            "artist": artist,
                        }
                        with cache_path.open("w", encoding="utf-8") as f:
                            json.dump(payload, f, indent=2)
                except Exception:
                    pass
            if lyrics is None:
                print("No synced lyrics found.")
            else:
                try:
                    print(
                        f"Lyrics found from {lyrics.source}; lines={len(lyrics.lines)}"
                    )
                except Exception:
                    print("Lyrics found.")
        except Exception as e:
            print(f"Lyrics fetch failed: {e}")
            lyrics = None

        # Build combined notes.mid: drums always, vocals if available
        try:
            vocals_syllables = None
            vocals_phrases = None
            if lyrics and lyrics.lines:
                tokens = to_rb_tokens(lyrics.lines)
                vocals_syllables = [
                    VoxSyllable(text=tok, t0=syl.t0, t1=syl.t1) for (syl, tok) in tokens
                ]
                vocals_phrases = [
                    VoxPhrase(t0=ln.t0, t1=ln.t1)
                    for ln in lyrics.lines
                    if (ln.t1 > ln.t0)
                ]
            notes_mid = write_notes_mid(
                out_dir=Path(ch_dir),
                bpm=float(bpm),
                ppq=480,
                sr=config.sample_rate,
                prediction_rows=pred_rows,
                vocals_syllables=vocals_syllables,
                vocals_phrases=vocals_phrases,
            )
            print(f"notes.mid written: {notes_mid}")
        except Exception as e:
            print(f"Failed to write notes.mid: {e}")

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
