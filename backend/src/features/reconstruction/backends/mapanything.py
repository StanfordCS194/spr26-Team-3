"""MapAnything — feed-forward METRIC multi-view reconstruction on Replicate.

`vufinder/map-anything` (Meta). Unlike VGGT/π³ (scale-invariant), MapAnything
outputs real-world metric scale — useful when the reconstruction feeds a
physics/RL sim. Same `inputs`→world-points envelope as VGGT; meshing is shared
via `_feedforward.py` (its discontinuity cutoff is percentile-based, so it
handles metric output without special-casing).
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from src.config import get_settings
from src.features.reconstruction.backends import _replicate as rep
from src.features.reconstruction.backends import register
from src.features.reconstruction.backends._feedforward import run_feedforward
from src.features.reconstruction.backends.base import (
    ReconstructionBackend,
    ReconstructionInput,
    ReconstructionOutput,
)


@register
class MapAnythingBackend(ReconstructionBackend):
    name = "mapanything"
    requires_gpu = False  # inference runs on Replicate's GPU
    implemented = rep.replicate_available()

    def reconstruct(
        self,
        inp: ReconstructionInput,
        out_dir: Path,
        progress_cb: Callable[[float, str], None],
    ) -> ReconstructionOutput:
        return run_feedforward(
            get_settings().replicate_mapanything_model, "mapanything", inp, out_dir, progress_cb
        )
