"""Custom mixed precision plugin for Apple's Metal Performance Shaders (MPS).

This plugin wraps :func:`torch.autocast` with ``device_type='mps'`` so that
PyTorch Lightning can run mixed precision training on Apple Silicon without
falling back to CUDA-specific defaults. It supports both ``float16`` and
``bfloat16`` dtypes, mirroring the behaviour of Lightning's built-in mixed
precision plugin but targeting the MPS backend.
"""

from __future__ import annotations

from typing import Literal

import torch
from pytorch_lightning.plugins.precision.amp import MixedPrecision


class MPSPrecisionPlugin(MixedPrecision):
    """Mixed precision plugin for MPS devices."""

    def __init__(self, precision: Literal["16-mixed", "bf16-mixed"]) -> None:
        # Initialise the parent MixedPrecision plugin with ``device='mps'`` so
        # that Lightning uses ``torch.autocast(device_type='mps')``.
        super().__init__(precision=precision, device="mps")

    # Explicitly override the autocast context manager to ensure that the
    # ``device_type`` is set to ``"mps"``.  This prevents Lightning from
    # defaulting to CUDA, which would emit a warning on Apple Silicon.
    def autocast_context_manager(self) -> torch.autocast:
        dtype = torch.bfloat16 if self.precision == "bf16-mixed" else torch.float16
        return torch.autocast(device_type="mps", dtype=dtype, cache_enabled=False)


__all__ = ["MPSPrecisionPlugin"]
