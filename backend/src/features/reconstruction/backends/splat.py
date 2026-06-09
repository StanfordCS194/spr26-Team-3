"""Gaussian Splatting backend — a two-stage cloud chain.

Stage 1: `jimothyjohn/colmap` recovers camera poses from the scan (reused from
the colmap backend). Stage 2: a Gaussian-Splatting TRAINER consumes those poses
+ images and emits a .ply/.splat of gaussians.

Replicate has no reliable photogrammetric 3DGS trainer hosted as a public
model (only single-image generative splats, which don't reconstruct a
walked-through room). So stage 2's model is configurable via
`settings.replicate_splat_model` — point it at a Replicate trainer (or your own
Modal/Replicate deployment). Until it's set, this backend reports
`implemented=False` so the API never offers a splat it can't run.
"""
from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

from src.config import get_settings
from src.features.reconstruction.backends import _replicate as rep
from src.features.reconstruction.backends import register
from src.features.reconstruction.backends.base import (
    ReconstructionBackend,
    ReconstructionInput,
    ReconstructionOutput,
)
from src.features.reconstruction.backends.colmap import frames_to_video


def _splat_configured() -> bool:
    s = get_settings()
    return bool(s.replicate_api_token and s.replicate_splat_model)


@register
class SplatBackend(ReconstructionBackend):
    name = "splat"
    requires_gpu = False  # training runs on Replicate
    # Needs BOTH a token and a configured trainer model (no default hosted one).
    implemented = _splat_configured()

    def reconstruct(
        self,
        inp: ReconstructionInput,
        out_dir: Path,
        progress_cb: Callable[[float, str], None],
    ) -> ReconstructionOutput:
        out_dir.mkdir(parents=True, exist_ok=True)
        settings = get_settings()
        if not settings.replicate_splat_model:
            raise RuntimeError(
                "No Gaussian-Splatting trainer configured. Set "
                "REPLICATE_SPLAT_MODEL to a Replicate trainer that takes a "
                "scan (video/images + COLMAP poses) and outputs a .ply/.splat. "
                "Replicate has no reliable public multi-view splat trainer, so "
                "this must be supplied (a hosted model or your own deployment)."
            )

        progress_cb(0.05, "encoding frames to video")
        with tempfile.TemporaryDirectory() as td:
            video_path = frames_to_video(inp.frames_dir, inp.fps_sampled, Path(td) / "scan.mp4")

            # Stage 1: COLMAP poses (NeRF-ready archive).
            progress_cb(0.15, f"COLMAP poses on Replicate ({settings.replicate_colmap_model})")
            with open(video_path, "rb") as vf:
                colmap_out = rep.run_model(
                    settings.replicate_colmap_model,
                    {"video": vf, "media": "video", "format": "nerfacto",
                     "quality": "Low", "continuous": True},
                )

            # Stage 2: train gaussians. Input shape depends on the chosen
            # trainer — this passes the scan video + COLMAP archive URI, which
            # most nerfstudio/3DGS trainers accept; align keys with your model.
            progress_cb(0.45, f"training gaussians on Replicate ({settings.replicate_splat_model})")
            with open(video_path, "rb") as vf:
                splat_out = rep.run_model(
                    settings.replicate_splat_model,
                    {"video": vf, "colmap": rep._uri_str(colmap_out)},
                )

            progress_cb(0.85, "downloading splat")
            if isinstance(splat_out, dict):
                splat_uri = splat_out.get("ply") or splat_out.get("output") or splat_out
            else:
                splat_uri = splat_out
            ply_path = rep.download(splat_uri, out_dir / "splat.ply")

        progress_cb(0.98, "done")
        return ReconstructionOutput(
            mesh_path=ply_path,
            point_cloud_path=ply_path,
            camera_poses=None,
            backend_meta={
                "actual_backend": "splat",
                "device": "replicate-cloud",
                "colmap_model": settings.replicate_colmap_model,
                "splat_model": settings.replicate_splat_model,
                "geometry": "gaussian_splat",
            },
        )
