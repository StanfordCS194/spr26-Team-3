"""Gaussian Splatting backend stub. Planned for change `worldscan-v2.1-splat`."""
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
class SplatBackend(ReconstructionBackend):
    name = "splat"
    requires_gpu = True
    implemented = False

    def reconstruct(
        self,
        inp: ReconstructionInput,
        out_dir: Path,
        progress_cb: Callable[[float, str], None],
    ) -> ReconstructionOutput:
        raise NotImplementedError("planned: worldscan-v2.1-splat")
