"""π³ (pi3) — feed-forward multi-view reconstruction on Replicate (cloud GPU).

`vufinder/map-anything-pi3`. A VGGT successor: permutation-equivariant (no
reference-frame dependence), so it's more stable across frame ordering.
Scale-invariant output. Same `inputs`→world-points envelope as VGGT — meshing
is shared via `_feedforward.py`.
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
class Pi3Backend(ReconstructionBackend):
    name = "pi3"
    requires_gpu = False  # inference runs on Replicate's GPU
    implemented = rep.replicate_available()

    def reconstruct(
        self,
        inp: ReconstructionInput,
        out_dir: Path,
        progress_cb: Callable[[float, str], None],
    ) -> ReconstructionOutput:
        return run_feedforward(
            get_settings().replicate_pi3_model, "pi3", inp, out_dir, progress_cb
        )
