from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .types import Segment


def detect_leading_silence_from_segments(
    segments: Sequence[Segment], threshold: float = -1e-3
) -> int:
    """Return the number of silent frames at the start of the audio.

    The function scans the first provided segment and finds the first frame
    whose mean energy exceeds ``threshold``. Spectrogram values are expected to
    be non-negative; values near zero indicate silence. If no such frame exists,
    ``0`` is returned.
    """
    if not segments:
        return 0
    for seg in segments:
        spec = seg["spec"]
        if isinstance(spec, torch.Tensor):
            spec_arr = spec.detach().cpu().numpy()
        else:
            spec_arr = np.asarray(spec)
        if spec_arr.ndim != 2:
            continue
        frame_energies = spec_arr.mean(axis=0)
        idx = np.nonzero(frame_energies < threshold)[0]
        if idx.size > 0:
            return int(seg["start_frame"]) + int(idx[0])
    return 0
