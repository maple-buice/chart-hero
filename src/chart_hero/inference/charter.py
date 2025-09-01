import os
from typing import Any

import librosa
import numpy as np
import pandas as pd
import torch
from music21 import chord, meter, note, stream
from numpy.typing import NDArray

from chart_hero.model_training.lightning_module import DrumTranscriptionModule
from chart_hero.model_training.transformer_config import get_drum_hits

from .types import PredictionRow, Segment, TransformerConfig


class Charter:
    def __init__(self, config: TransformerConfig, model_path: str | os.PathLike[str]):
        self.config = config
        self.model = DrumTranscriptionModule.load_from_checkpoint(
            str(model_path), config=config
        )
        self.model.eval()

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

                # Thresholds
                thr = self.config.prediction_threshold
                thr_vec = None
                if getattr(self.config, "class_thresholds", None):
                    ct = self.config.class_thresholds
                    if isinstance(ct, (list, tuple)) and len(ct) == len(classes):
                        thr_vec = torch.tensor(ct).view(1, 1, -1)

                for b_idx, seg in enumerate(batch):
                    seg_probs = probs[b_idx]  # [T_patches, C]
                    if thr_vec is not None:
                        active = (seg_probs >= thr_vec).to(torch.bool)
                    else:
                        active = (seg_probs >= thr).to(torch.bool)
                    if active.numel() == 0:
                        continue
                    # Map each patch t to a representative frame
                    T_p = seg_probs.shape[0]
                    for t in range(T_p):
                        cls_mask = active[t]  # [C]
                        if not cls_mask.any():
                            continue
                        # Map to frame center within this patch
                        frame_idx = seg["start_frame"] + t * patch + patch // 2
                        peak_sample = int(frame_idx * hop)
                        row: PredictionRow = {
                            "peak_sample": peak_sample,
                            "0": 0,
                            "1": 0,
                            "2": 0,
                            "3": 0,
                            "4": 0,
                            "66": 0,
                            "67": 0,
                            "68": 0,
                        }
                        for ci, lab in enumerate(classes):
                            row[lab] = int(cls_mask[ci].item())
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
