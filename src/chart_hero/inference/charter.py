import os
from typing import Any

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from music21 import chord, meter, note, stream
from numpy.typing import NDArray

from chart_hero.model_training.lightning_module import DrumTranscriptionModule
from chart_hero.model_training.transformer_config import get_drum_hits

from .types import PredictionRow, Segment, TransformerConfig


class Charter:
    def __init__(self, config: TransformerConfig, model_path: str | os.PathLike[str]):
        self.config = config
        # Attempt to read structural hyperparameters from the checkpoint to avoid
        # shape mismatches (e.g., patch_size, hidden_size, time_embed length).
        max_time_patches = None
        try:
            import torch as _torch

            ckpt = _torch.load(str(model_path), map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            # Derive time positional embedding length
            te = state.get("model.pos_encoding.time_embed")
            if te is not None and hasattr(te, "shape") and len(te.shape) == 3:
                max_time_patches = int(te.shape[1])
            # Derive patch/kernel size and hidden dims from patch_embed weights
            pe = state.get("model.patch_embed.projection.weight")
            if pe is not None and hasattr(pe, "shape") and len(pe.shape) == 3:
                # shape: [embed_dim, n_mels, patch_time]
                embed_dim, n_mels, patch_time = (
                    int(pe.shape[0]),
                    int(pe.shape[1]),
                    int(pe.shape[2]),
                )
                # Apply to config if differs
                try:
                    self.config.hidden_size = embed_dim  # type: ignore[attr-defined]
                    self.config.n_mels = n_mels  # type: ignore[attr-defined]
                    pt = list(getattr(self.config, "patch_size", (patch_time, 16)))
                    pt[0] = patch_time
                    self.config.patch_size = tuple(pt)  # type: ignore[attr-defined]
                except Exception:
                    pass
            # If hyperparameters saved by Lightning exist, prefer them for consistency
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
                            setattr(self.config, k, hparams[k])
                        except Exception:
                            pass
        except Exception:
            pass

        # Load model with strict=False to allow criterion buffers etc. to differ
        self.model = DrumTranscriptionModule.load_from_checkpoint(
            str(model_path),
            config=self.config,
            max_time_patches=max_time_patches,
            strict=False,
        )
        self.model.eval()
        # Try to load calibrated per-class thresholds saved during training
        # Only if not already provided on the config
        if not getattr(self.config, "class_thresholds", None):
            try:
                import json
                from pathlib import Path

                thr_path = Path(model_path).parent / "class_thresholds.json"
                if thr_path.exists():
                    with open(thr_path, "r") as f:
                        data = json.load(f)
                    thrs = data.get("class_thresholds")
                    if isinstance(thrs, list) and thrs:
                        self.config.class_thresholds = [float(x) for x in thrs]
            except Exception:
                pass

    from typing import Sequence

    def predict(self, segments: Sequence[Segment | torch.Tensor]) -> pd.DataFrame:
        """
        Run model inference over spectrogram segments and return a DataFrame of
        event-level predictions suitable for chart writing.

        segments: sequence of dicts from audio_to_tensors with keys:
          - 'spec': np.ndarray (n_mels, frames)
          - 'start_frame': int
          - 'end_frame': int
          - 'total_frames': int
        """
        if not segments:
            return pd.DataFrame(columns=["peak_sample"] + get_drum_hits())

        device = torch.device(
            self.config.device
            if torch.cuda.is_available()
            or getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(device)

        patch = int(self.config.patch_size[0])
        # Actual step between logits along time may be smaller if the model uses
        # an overlap stride; infer it from the conv stride when available.
        try:
            stride_frames = int(
                getattr(self.model.model.patch_embed.projection, "stride", (patch,))[0]
            )
        except Exception:
            stride_frames = patch
        hop = int(self.config.hop_length)
        classes = get_drum_hits()
        rows: list[PredictionRow] = []

        with torch.no_grad():
            # Batch reasonably to avoid OOM
            batch_size = 4
            # Normalize input: accept legacy tensor lists too
            norm_segments: list[Segment] = []
            seg_len_frames = int(
                self.config.max_audio_length
                * self.config.sample_rate
                / self.config.hop_length
            )
            for idx, seg in enumerate(segments):
                if isinstance(seg, torch.Tensor):
                    ten = seg.detach().cpu().float().squeeze()
                    if ten.dim() != 2:
                        continue
                    if ten.shape[0] == self.config.n_mels:
                        spec_np = ten.numpy()
                    else:
                        spec_np = ten.t().numpy()
                    norm_segments.append(
                        {
                            "spec": spec_np,
                            "start_frame": idx * seg_len_frames,
                            "end_frame": idx * seg_len_frames + spec_np.shape[1],
                            "total_frames": spec_np.shape[1],
                        }
                    )
                else:
                    spec = seg["spec"]
                    arr = torch.from_numpy(spec).float().detach().cpu()
                    if arr.dim() == 2:
                        n0, _ = arr.shape
                        spec_np = (
                            arr.numpy() if n0 == self.config.n_mels else arr.t().numpy()
                        )
                    else:
                        a = arr.squeeze()
                        if a.dim() != 2:
                            continue
                        n0, _ = a.shape
                        spec_np = (
                            a.numpy() if n0 == self.config.n_mels else a.t().numpy()
                        )
                    norm_segments.append(
                        {
                            "spec": spec_np,
                            "start_frame": int(
                                seg.get("start_frame", idx * seg_len_frames)
                            ),
                            "end_frame": int(
                                seg.get(
                                    "end_frame", idx * seg_len_frames + spec_np.shape[1]
                                )
                            ),
                            "total_frames": int(
                                seg.get("total_frames", spec_np.shape[1])
                            ),
                        }
                    )

            for i in range(0, len(norm_segments), batch_size):
                batch = norm_segments[i : i + batch_size]
                # Build tensor: [B, 1, n_mels, frames]
                specs = [torch.from_numpy(b["spec"]).float() for b in batch]
                # pad to max frames in batch to allow stacking
                max_f = max(s.shape[1] for s in specs)
                padded = []
                for s in specs:
                    if s.shape[1] < max_f:
                        pad_w = max_f - s.shape[1]
                        fill = float(s.min().item()) if s.numel() > 0 else 0.0
                        ps = torch.full(
                            (s.shape[0], max_f), fill_value=fill, dtype=s.dtype
                        )
                        ps[:, : s.shape[1]] = s
                        s = ps
                    padded.append(s.unsqueeze(0).unsqueeze(0))  # -> [1,1,n_mels,frames]
                x = torch.cat(padded, dim=0).to(device)
                out = self.model(x)
                logits = out["logits"].cpu()  # [B, T_patches, C]
                probs = torch.sigmoid(logits)
                # Optional onset gating from auxiliary head
                onset_thr = getattr(self.config, "onset_gate_threshold", None)
                onset_probs: torch.Tensor | None = None
                if onset_thr is not None and "onset_logits" in out:
                    onset_probs = torch.sigmoid(out["onset_logits"]).cpu()  # [B, T]

                # Apply optional per-class gains to re-calibrate probabilities
                if getattr(self.config, "class_gains", None):
                    gains = self.config.class_gains
                    if isinstance(gains, (list, tuple)) and len(gains) == len(classes):
                        g = torch.tensor(gains, dtype=probs.dtype).view(1, 1, -1)
                        probs = torch.clamp(probs * g, 0.0, 1.0)

                # Thresholds
                thr = self.config.prediction_threshold
                thr_vec = None
                if getattr(self.config, "class_thresholds", None):
                    ct = self.config.class_thresholds
                    if isinstance(ct, (list, tuple)) and len(ct) == len(classes):
                        # shape [1, C] to broadcast against [T, C]
                        thr_vec = torch.tensor(ct).view(1, -1).to(probs.device)

                # Optional decode min spacing control (ms)
                min_default = getattr(self.config, "min_spacing_ms_default", None)
                min_map = getattr(self.config, "min_spacing_ms_map", None) or {}
                last_time_ms: dict[str, float] = {}

                # Optional per-class time offset corrections (ms)
                class_time_offsets_ms = getattr(
                    self.config, "class_time_offsets_ms", None
                )
                classes_list = classes

                for b_idx, seg in enumerate(batch):
                    seg_probs = probs[b_idx]  # [T_patches, C]
                    T_p = seg_probs.shape[0]
                    # Build thresholds vector [C]
                    if thr_vec is not None:
                        thr_row = thr_vec.squeeze(0)
                    else:
                        thr_row = torch.full(
                            (seg_probs.shape[1],), thr, device=seg_probs.device
                        )

                    # Label names -> indices
                    idx = {lab: i for i, lab in enumerate(classes)}

                    # Optional per-class NMS along time to reduce duplicate hits
                    k = max(1, int(getattr(self.config, "event_nms_kernel_patches", 3)))
                    if k > 1 and T_p > 1:
                        # seg_probs: [T,C] -> [1,C,T] for pooling
                        p_ct = seg_probs.transpose(0, 1).unsqueeze(0)
                        pooled = F.max_pool1d(
                            p_ct, kernel_size=k, stride=1, padding=k // 2
                        )
                        # keep only local maxima
                        keep_mask = (p_ct >= pooled).squeeze(0).transpose(0, 1)  # [T,C]
                    else:
                        keep_mask = torch.ones_like(seg_probs, dtype=torch.bool)

                    for t in range(T_p):
                        p_t = seg_probs[t]  # [C]
                        km_t = keep_mask[t]
                        # Binary activations by threshold
                        act = (p_t >= thr_row) & km_t

                        # Optional onset gate: require onset probability >= threshold
                        if onset_probs is not None:
                            try:
                                if float(onset_probs[b_idx, t].item()) < float(
                                    onset_thr
                                ):
                                    continue
                            except Exception:
                                pass

                        # Optional activity gate: if nothing is strong enough, skip whole tick
                        gate = getattr(self.config, "activity_gate", None)
                        if gate is not None:
                            if float(p_t.max().item()) < float(gate):
                                continue

                        # Pairwise arbitration per color (prefer higher prob when both fire)
                        # Yellow: tom '2' vs hat '67'
                        y_tom = idx.get("2")
                        y_cym = idx.get("67")
                        b_tom = idx.get("3")
                        b_cym = idx.get("68")
                        g_tom = idx.get("4")
                        g_cym = idx.get("66")

                        margin = float(getattr(self.config, "cymbal_margin", 0.1))
                        tom_margin = float(
                            getattr(self.config, "tom_over_cymbal_margin", 0.35)
                        )

                        def choose_pair(
                            tom_i: int | None, cym_i: int | None
                        ) -> tuple[int, int]:
                            if tom_i is None or cym_i is None:
                                # Fallback: nothing to arbitrate
                                return (
                                    int(act[tom_i].item()) if tom_i is not None else 0,
                                    int(act[cym_i].item()) if cym_i is not None else 0,
                                )
                            a_t = bool(act[tom_i].item())
                            a_c = bool(act[cym_i].item())
                            if a_t and a_c:
                                pt = float(p_t[tom_i].item())
                                pc = float(p_t[cym_i].item())
                                # Selection policy:
                                # - If cymbal within margin of tom, choose cymbal
                                # - Else require tom to exceed cymbal by a larger margin to choose tom
                                if (pc + margin) >= pt:
                                    return (0, 1)
                                if pt >= (pc + tom_margin):
                                    return (1, 0)
                                # Otherwise still prefer cymbal
                                return (0, 1)
                            return (int(a_t), int(a_c))

                        y_t, y_c = choose_pair(y_tom, y_cym)
                        b_t, b_c = choose_pair(b_tom, b_cym)
                        g_t, g_c = choose_pair(g_tom, g_cym)

                        # If nothing active on any class, skip
                        i0 = idx.get("0")
                        i1 = idx.get("1")
                        has_kick = bool(act[i0].item()) if i0 is not None else False
                        has_snare = bool(act[i1].item()) if i1 is not None else False
                        if not (
                            has_kick
                            or has_snare
                            or (y_t or y_c or b_t or b_c or g_t or g_c)
                        ):
                            continue

                        # Map to frame center using kernel size and stride
                        frame_idx = (
                            seg["start_frame"] + t * stride_frames + (patch // 2)
                        )
                        # Apply per-class time offsets by shifting sample position
                        # Use the largest offset among active classes to avoid splitting events
                        add_ms = 0.0
                        if isinstance(class_time_offsets_ms, (list, tuple)) and len(
                            class_time_offsets_ms
                        ) == len(classes_list):
                            # Determine active classes indices
                            active_indices: list[int] = []
                            for name, idx_i in [
                                ("0", idx.get("0")),
                                ("1", idx.get("1")),
                                ("2", idx.get("2")),
                                ("3", idx.get("3")),
                                ("4", idx.get("4")),
                                ("66", idx.get("66")),
                                ("67", idx.get("67")),
                                ("68", idx.get("68")),
                            ]:
                                if idx_i is not None and bool(act[idx_i].item()):
                                    active_indices.append(idx_i)
                            if active_indices:
                                add_ms = -max(
                                    float(class_time_offsets_ms[i])
                                    for i in active_indices
                                )
                        peak_sample = int(
                            (frame_idx * hop)
                            + (add_ms * self.config.sample_rate / 1000.0)
                        )

                        row: PredictionRow = {
                            "peak_sample": peak_sample,
                            "0": int(act[i0].item()) if i0 is not None else 0,
                            "1": int(act[i1].item()) if i1 is not None else 0,
                            "2": y_t,
                            "3": b_t,
                            "4": g_t,
                            "66": g_c,
                            "67": y_c,
                            "68": b_c,
                        }

                        # Optional min spacing per class (skip if within limit)
                        def _ms_from_sample(samp: int) -> float:
                            return float(samp) * 1000.0 / float(self.config.sample_rate)

                        now_ms = _ms_from_sample(peak_sample)
                        keep = True
                        for lab in ["0", "1", "2", "3", "4", "66", "67", "68"]:
                            if row.get(lab, 0) != 1:
                                continue
                            last_ms = last_time_ms.get(lab)
                            limit = (
                                min_map.get(lab, min_default)
                                if min_default is not None
                                else min_map.get(lab)
                            )
                            if (
                                last_ms is not None
                                and limit is not None
                                and (now_ms - last_ms) < float(limit)
                            ):
                                keep = False
                                break
                        if not keep:
                            continue
                        for lab in ["0", "1", "2", "3", "4", "66", "67", "68"]:
                            if row.get(lab, 0) == 1:
                                last_time_ms[lab] = now_ms
                        rows.append(row)

        if not rows:
            return pd.DataFrame(columns=["peak_sample"] + classes)

        df = pd.DataFrame(rows)
        # Combine duplicate ticks by OR-ing the class indicators
        df = (
            df.groupby("peak_sample", as_index=False)
            .agg({**{c: "max" for c in classes}})
            .sort_values("peak_sample")
        )
        return df


class ChartGenerator:
    """
    Create an object that store human readable sheet music file transcribed from the model output.
    """

    def __init__(
        self,
        prediction_df: pd.DataFrame,
        song_duration: float,
        bpm: float | None,
        sample_rate: int,
        beats_in_measure: int = 4,
        note_value: int = 4,
        note_offset: int | None = None,
        song_title: str | None = None,
    ) -> None:
        self.sheet: Any
        self.offset = False
        self.beats_in_measure = beats_in_measure * 2
        self.note_value = note_value
        # Fallback to a reasonable default BPM if none provided
        self.bpm = bpm or 120
        self.df = prediction_df
        self.sample_rate = sample_rate
        self.onsets = prediction_df.peak_sample
        if self.onsets.empty:
            from typing import Any, cast

            self.sheet = stream.Score()
            drum_part = cast(Any, stream).Part()
            drum_part.id = "drums"
            drum_part.append(
                cast(Any, meter).TimeSignature(
                    f"{int(self.beats_in_measure / 2)}/{self.note_value}"
                )
            )
            cast(Any, self.sheet).insert(0, drum_part)
            return
        self.note_line = self.onsets.apply(
            lambda x: librosa.samples_to_time(x, sr=sample_rate)
        ).to_numpy()
        self.get_note_duration()

        if note_offset is None:
            total_8_note = []
            for n in range(min(20, len(self.df))):
                temp_8_div = self.get_eighth_note_time_grid(
                    song_duration, note_offset=n
                )
                temp_synced_8_div = self.sync_8(temp_8_div)
                total_8_note.append(
                    len(
                        np.intersect1d(
                            np.around(self.note_line, 8),
                            np.around(temp_synced_8_div, 8),
                        )
                    )
                )
            note_offset = int(np.argmax(total_8_note))
        else:
            pass

        if (note_offset or 0) > 0:
            self.offset = True

        _8_div = self.get_eighth_note_time_grid(
            song_duration, note_offset=int(note_offset or 0)
        )
        self.synced_8_div = self.sync_8(_8_div)

        _16_div, _32_div, _8_triplet_div, _8_sixlet_div = self.get_note_division()

        (
            self.synced_8_div_clean,
            self.synced_16_div,
            self.synced_32_div,
            self.synced_8_3_div,
            self.synced_8_6_div,
        ) = self.master_sync(_16_div, _32_div, _8_triplet_div, _8_sixlet_div)

        self.pitch_dict: dict[float, list[int | str]] = self.get_pitch_dict()
        stream_time_map, stream_pitch, stream_note = self.build_stream()
        self.music21_data = self.get_music21_data(
            stream_time_map, stream_pitch, stream_note
        )
        from typing import cast

        self.sheet = self.sheet_construction(
            cast(dict[int, dict[str, list[object]]], self.music21_data),
            song_title=song_title,
        )

    def get_music21_data(
        self,
        stream_time_map: list[list[float]],
        stream_pitch: list[list[list[int | str]]],
        stream_note: list[list[float]],
    ) -> dict[int, dict[str, object]]:
        """
        A function to clean up and merge all the necessary information in a format that can pass to the sheet_construction step to build sheet music
        """
        music21_data = {}
        for i in range(len(stream_time_map)):
            music21_data[i] = {"pitch": stream_pitch[i], "note_type": stream_note[i]}
        return music21_data

    def sheet_construction(
        self,
        music21_data: dict[int, dict[str, list[object]]],
        song_title: str | None = None,
    ) -> stream.Score:
        """
        A function to build sheet music using Music21 library
        """
        # from music21 import stream, note, chord, meter, layout

        # Create a new stream (sheet music)
        sheet = stream.Score()

        # Add title and other metadata
        if song_title:
            from music21 import metadata

            sheet.metadata = metadata.Metadata(title=song_title)

        # Create a drum part
        from typing import Any, cast

        drum_part = cast(Any, stream).Part()
        drum_part.id = "drums"

        # Add time signature
        drum_part.append(
            cast(Any, meter).TimeSignature(
                f"{int(self.beats_in_measure / 2)}/{self.note_value}"
            )
        )

        # Add notes to the part
        for measure_num in sorted(music21_data.keys()):
            measure = cast(Any, stream).Measure(number=measure_num)
            pitch_lists_obj = music21_data[measure_num]["pitch"]
            note_types_obj = music21_data[measure_num]["note_type"]
            from typing import cast as _cast

            pitch_lists_t = _cast(list[list[int | str]], pitch_lists_obj)
            note_types_t = _cast(list[float], note_types_obj)
            for i, pitch_list in enumerate(pitch_lists_t):
                note_type = note_types_t[i]

                n: note.GeneralNote
                if "rest" in pitch_list:
                    n = cast(Any, note).Rest()
                else:
                    # Create a chord for multiple drum hits at the same time
                    n = cast(Any, chord).Chord(pitch_list)

                n.duration.quarterLength = note_type
                measure.append(n)
            drum_part.append(measure)

        # Add drum part to the sheet music
        cast(Any, sheet).insert(0, drum_part)

        return sheet

    def build_stream(
        self,
    ) -> tuple[list[list[float]], list[list[list[int | str]]], list[list[float]]]:
        """
        A function to clean up and merge all the necessary information in a format that can pass to the build_stream step to build sheet music
        """
        measure_log = 0
        stream_time_map: list[list[float]] = []
        stream_pitch: list[list[list[int | str]]] = []
        stream_note: list[list[float]] = []
        synced_8_div = np.around(self.synced_8_div, 8)
        for i in range(len(synced_8_div) // self.beats_in_measure):
            measure_iter = list(
                synced_8_div[measure_log : measure_log + self.beats_in_measure]
            )
            measure, note_dur = self.build_measure(measure_iter)
            stream_time_map.append(measure)
            stream_note.append(note_dur)
            measure_log = measure_log + self.beats_in_measure

        remaining_8 = len(synced_8_div) % self.beats_in_measure
        measure, note_dur = self.build_measure(
            list(synced_8_div[-remaining_8:].tolist())
        )
        measure.extend([-1.0] * (self.beats_in_measure - remaining_8))
        note_dur.extend([8.0] * (self.beats_in_measure - remaining_8))

        stream_time_map.append(measure)
        stream_note.append(note_dur)

        for measure in stream_time_map:
            pitch_set: list[list[int | str]] = []
            for note_val in measure:
                if note_val in self.pitch_dict:
                    if len(self.pitch_dict[note_val]) == 0:
                        pitch_set.append(["rest"])
                    else:
                        pitch_set.append(self.pitch_dict[note_val])
                else:
                    pitch_set.append(["rest"])
            stream_pitch.append(pitch_set)
        return stream_time_map, stream_pitch, stream_note

    def get_note_duration(self) -> None:
        """
        A function to calculate different note duration
        """
        self._8_duration = 60 / self.bpm / 2
        self._16_duration = 60 / self.bpm / 4
        self._32_duration = 60 / self.bpm / 8
        self._8_triplet_duration = self._8_duration / 3

    def get_eighth_note_time_grid(
        self, song_duration: float, note_offset: int = 0
    ) -> NDArray[np.float64]:
        """
        A function to calculate the eighth note time grid
        """
        first_note = librosa.samples_to_time(
            self.df.peak_sample.iloc[note_offset], sr=self.sample_rate
        )
        return np.arange(first_note, song_duration, self._8_duration)

    def sync_8(self, _8_div: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        A function to map the eighth note time grid to the onsets
        """
        if len(_8_div) == 0:
            return np.array([])
        # match timing of the first note
        synced_8_div = [_8_div[0]]
        diff_log = 0

        # first, map and sync 8th notes to the onset
        for note_val in _8_div[1:]:
            pos = np.argmin(np.abs(self.note_line - (note_val + diff_log)))
            diff = self.note_line[pos] - (note_val + diff_log)

            if np.abs(diff) > self._32_duration:
                synced_8_div.append(synced_8_div[-1] + self._8_duration)
            else:
                diff_log = diff_log + diff
                synced_8_div.append(note_val + diff_log)
        if self.offset:
            for i in range(self.beats_in_measure):
                synced_8_div.insert(0, synced_8_div[0] - self._8_duration)
        return np.array(synced_8_div)

    def get_note_division(
        self,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """
        A function to calculate the note dividion of various note type and created a numpy array to map on the drum track
        """

        _16_div = self.synced_8_div[:-1] + (np.diff(self.synced_8_div) / 2)

        full_16_div = np.sort(np.concatenate((self.synced_8_div, _16_div), axis=0))
        _32_div = full_16_div[:-1] + (np.diff(full_16_div) / 2)

        _8_triplet_a = self.synced_8_div[:-1] + (np.diff(self.synced_8_div) / 3)
        _8_triplet_b = self.synced_8_div[:-1] + (np.diff(self.synced_8_div) / 3 * 2)
        _8_triplet_div = np.sort(np.concatenate((_8_triplet_a, _8_triplet_b), axis=0))

        full_8_triple_div = np.sort(
            np.concatenate((_8_triplet_div, self.synced_8_div), axis=0)
        )
        _8_sixlet_div = full_8_triple_div[:-1] + (np.diff(full_8_triple_div) / 2)

        return _16_div, _32_div, _8_triplet_div, _8_sixlet_div

    def master_sync(
        self,
        _16_div: NDArray[np.float64],
        _32_div: NDArray[np.float64],
        _8_triplet_div: NDArray[np.float64],
        _8_sixlet_div: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """
        A note quantization function to map 16th, 32th, eighth triplets or eighth sixthlet note to each onset when applicable
        """
        # round the onsets amd synced eighth note position (in the unit of seconds) to 8 decimal places for convinience purpose
        note_line_r = np.round(self.note_line, 8)
        synced_eighth_r = np.round(self.synced_8_div, 8)

        # declare a few variables to store the result
        synced_16_div: list[float] = []
        synced_32_div: list[float] = []
        synced_8_3_div: list[float] = []
        synced_8_6_div: list[float] = []

        # iterate though all synced 8 notes
        for i in range(len(synced_eighth_r) - 1):
            # retrive the current 8th note and the next 8th note (n and n+1)
            eighth_pair = synced_eighth_r[i : i + 2]
            sub_notes = note_line_r[
                (note_line_r > eighth_pair[0]) & (note_line_r < eighth_pair[1])
            ]
            # Check whether there is any detected onset exist between 2 consecuive eighth notes
            if len(sub_notes) > 0:
                # if onsets are deteced between 2 consecuive eighth notes,
                # the below algo will match each note (based on its position in the time domain) to the closest note division (16th, 32th, eighth triplets or eighth sixthlet note)
                dist_dict: dict[str, list[float]] = {
                    "_16": [],
                    "_32": [],
                    "_8_3": [],
                    "_8_6": [],
                }
                sub_notes_dict = {
                    "_16": np.round(
                        np.linspace(self.synced_8_div[i], self.synced_8_div[i + 1], 3),
                        8,
                    )[:-1],
                    "_32": np.round(
                        np.linspace(self.synced_8_div[i], self.synced_8_div[i + 1], 5),
                        8,
                    )[:-1],
                    "_8_3": np.round(
                        np.linspace(self.synced_8_div[i], self.synced_8_div[i + 1], 4),
                        8,
                    )[:-1],
                    "_8_6": np.round(
                        np.linspace(self.synced_8_div[i], self.synced_8_div[i + 1], 7),
                        8,
                    )[:-1],
                }

                for sub_note in sub_notes:
                    diff_16 = np.min(np.abs(_16_div - sub_note))
                    dist_dict["_16"].append(diff_16)
                    _16closest_line = _16_div[np.argmin(np.abs(_16_div - sub_note))]
                    sub_notes_dict["_16"] = np.where(
                        sub_notes_dict["_16"] == np.round(_16closest_line, 8),
                        sub_note,
                        sub_notes_dict["_16"],
                    )

                    diff_32 = np.min(np.abs(_32_div - sub_note))
                    dist_dict["_32"].append(diff_32)
                    _32closest_line = _32_div[np.argmin(np.abs(_32_div - sub_note))]
                    sub_notes_dict["_32"] = np.where(
                        sub_notes_dict["_32"] == np.round(_32closest_line, 8),
                        sub_note,
                        sub_notes_dict["_32"],
                    )

                    diff_8_triplet = np.min(np.abs(_8_triplet_div - sub_note))
                    dist_dict["_8_3"].append(diff_8_triplet)
                    _8_3closest_line = _8_triplet_div[
                        np.argmin(np.abs(_8_triplet_div - sub_note))
                    ]
                    sub_notes_dict["_8_3"] = np.where(
                        sub_notes_dict["_8_3"] == np.round(_8_3closest_line, 8),
                        sub_note,
                        sub_notes_dict["_8_3"],
                    )

                    diff_8_sixlet = np.min(np.abs(_8_sixlet_div - sub_note))
                    dist_dict["_8_6"].append(diff_8_sixlet)
                    _8_6closest_line = _8_sixlet_div[
                        np.argmin(np.abs(_8_sixlet_div - sub_note))
                    ]
                    sub_notes_dict["_8_6"] = np.where(
                        sub_notes_dict["_8_6"] == np.round(_8_6closest_line, 8),
                        sub_note,
                        sub_notes_dict["_8_6"],
                    )

                for key in dist_dict.keys():
                    dist_dict[key] = [sum(dist_dict[key]) / len(dist_dict[key])]
                best_div = min(dist_dict, key=lambda k: dist_dict[k][0])
                if best_div == "_16":
                    synced_16_div.extend(sub_notes_dict["_16"])
                elif best_div == "_32":
                    synced_32_div.extend(sub_notes_dict["_32"])
                elif best_div == "_8_3":
                    synced_8_3_div.extend(sub_notes_dict["_8_3"])
                else:
                    synced_8_6_div.extend(sub_notes_dict["_8_6"])

            else:
                pass

        # If there is any notes living in between 2 consecutive 8th notes, the first 8th note is not an 8th note anymore.
        # Bleow for loop will remove those notes from the synced_8_div variable
        synced_8_div_clean = self.synced_8_div.copy()
        for div in [synced_16_div, synced_32_div, synced_8_3_div, synced_8_6_div]:
            synced_8_div_clean = synced_8_div_clean[
                ~np.in1d(np.around(synced_8_div_clean, 8), np.around(div, 8))
            ]
        return (
            synced_8_div_clean,
            np.array(synced_16_div),
            np.array(synced_32_div),
            np.array(synced_8_3_div),
            np.array(synced_8_6_div),
        )

    def build_measure(
        self, measure_iter: list[float]
    ) -> tuple[list[float], list[float]]:
        """
        A function to clean up note quantization result information in a format that can pass to the build_stream step to build all the required data for sheet music construction step
        """
        synced_16_div = np.around(self.synced_16_div, 8)
        synced_32_div = np.around(self.synced_32_div, 8)
        synced_8_3_div = np.around(self.synced_8_3_div, 8)
        synced_8_6_div = np.around(self.synced_8_6_div, 8)
        measure: list[list[float]] = []
        note_dur: list[float] = []
        for note_val in measure_iter:
            _div = False
            for div in [
                (synced_16_div, 2, 0.25),
                (synced_32_div, 4, 0.125),
                (synced_8_3_div, 3, 1 / 6),
                (synced_8_6_div, 6, 1 / 12),
            ]:
                if note_val in div[0]:
                    pos = np.where(div[0] == note_val)
                    pos = pos[0][0]
                    measure.append(list(div[0][pos : pos + div[1]]))
                    note_dur.extend([div[2]] * div[1])
                    _div = True
            if not _div:
                measure.append([note_val])
                note_dur.append(0.5)

        flat_measure = [item for sublist in measure for item in sublist]
        return flat_measure, note_dur

    def get_pitch_dict(self) -> dict[float, list[int | str]]:
        """
        A function to reformat the prediction result in a format that can pass to the build_stream step to build all the required data for sheet music construction step
        """
        from chart_hero.model_training.transformer_config import DRUM_HIT_MAP

        # Create a reverse mapping from drum hit class to MIDI note
        class_to_midi = {}
        for midi, hit_class in DRUM_HIT_MAP.items():
            if hit_class not in class_to_midi:
                class_to_midi[hit_class] = midi

        pitch_mapping_df = self.df[["peak_sample"] + get_drum_hits()].set_index(
            "peak_sample"
        )
        from typing import cast as _cast

        mapping_dict = _cast(
            dict[int, dict[str, int]], pitch_mapping_df.to_dict(orient="index")
        )
        pitch_dict: dict[float, list[int | str]] = {}
        for p in mapping_dict.keys():
            time = float(round(librosa.samples_to_time(int(p), sr=self.sample_rate), 8))
            pitch_dict[time] = []
            for hit_class, is_hit in mapping_dict[p].items():
                if is_hit == 1:
                    pitch_dict[time].append(class_to_midi[hit_class])
        return pitch_dict
