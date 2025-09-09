from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .types import Segment


def detect_leading_silence_from_segments(
    segments: Sequence[Segment], threshold: float = 1e-3
) -> int:
    """Return the number of silent frames preceding the first audible frame.

    Frame energy is measured as the mean spectrogram magnitude per frame.  If
    the spectrogram is in log scale (e.g., dB), it is converted back to a
    positive magnitude before computing energy.  The first frame whose energy is
    greater than or equal to ``threshold`` is considered the start of audible
    content.  The returned index is in frame units relative to the beginning of
    the audio.  If no frame crosses the threshold, the total number of frames is
    returned.
    """
    if not segments:
        return 0

    last_end = 0
    for seg in segments:
        spec = seg["spec"]
        if isinstance(spec, torch.Tensor):
            spec_arr = spec.detach().cpu().numpy()
        else:
            spec_arr = np.asarray(spec)
        if spec_arr.ndim != 2:
            last_end = int(seg.get("end_frame", last_end))
            continue

        if np.any(spec_arr < 0):
            spec_arr = np.power(10.0, spec_arr / 20.0)

        frame_energies = spec_arr.mean(axis=0)
        idx = np.nonzero(frame_energies >= threshold)[0]
        if idx.size > 0:
            return int(seg["start_frame"]) + int(idx[0])

        last_end = int(seg.get("end_frame", seg["start_frame"] + frame_energies.size))

    return last_end
