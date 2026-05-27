"""VGGT — feed-forward neural reconstruction. Implemented in PR-B.

This stub keeps the registry happy so the API can advertise the backend
list, and so the UI can show "VGGT" as the default selection from day 1.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from src.features.reconstruction.backends import register
from src.features.reconstruction.backends.base import (
    ReconstructionBackend,
    ReconstructionInput,
    ReconstructionOutput,
)


@register
class VGGTBackend(ReconstructionBackend):
    name = "vggt"
    requires_gpu = True
    implemented = False  # flipped to True in PR-B

    def reconstruct(
        self,
        inp: ReconstructionInput,
        out_dir: Path,
        progress_cb: Callable[[float, str], None],
    ) -> ReconstructionOutput:
        raise NotImplementedError("VGGT backend lands in PR-B of worldscan-v2-video-product.")
