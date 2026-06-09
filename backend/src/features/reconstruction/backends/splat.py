"""Gaussian Splatting backend — depth_fusion + splat, a two-stage chain.

Stage A (consume, read-only): run the `depth_fusion` backend as a black box to
recover camera poses (OpenCV cam-to-world per frame) and an initial colored
point cloud (`points.ply`, already flipped to Y-up). depth_fusion is invoked
ONLY through its public `reconstruct()` call and the returned
`ReconstructionOutput` + the files it writes. This module never edits or imports
internals of depth_fusion.

Stage B (adapt + train): a pure pose-convention adapter turns depth_fusion's
poses into a COLMAP/instant-ngp `transforms.json` in the SAME Y-up frame as
`points.ply`, stages frames + transforms + the init cloud into a trainer bundle,
then (if `settings.replicate_splat_model` is configured) calls the Replicate
Gaussian-Splatting trainer and downloads the trained `.ply`/`.splat`.

Replicate has no reliable public photogrammetric 3DGS trainer that accepts
posed multi-view images + an init cloud and returns a trained splat (only
single-image/text generative splats, which don't reconstruct a walked-through
room). So stage B's model is configurable via `settings.replicate_splat_model`
— point it at a Replicate trainer (or your own Modal/Replicate deployment).
Until it's set, this backend reports `implemented=False` so the API never offers
a splat it can't run, and `reconstruct()` raises a clear, actionable error.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.config import get_settings
from src.features.reconstruction.backends import _replicate as rep
from src.features.reconstruction.backends import register
from src.features.reconstruction.backends.base import (
    ReconstructionBackend,
    ReconstructionInput,
    ReconstructionOutput,
)
from src.features.reconstruction.backends.depth_fusion import DepthFusionBackend

# Same world-side Y/Z flip depth_fusion applies to its geometry
# (depth_fusion.py:344-345: negate Y and Z). Left-multiplying a cam-to-world
# matrix by this puts the camera poses into the identical Y-up frame as the
# exported points.ply, so the trained splat and its init cloud share one frame.
_YUP_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def _splat_configured() -> bool:
    s = get_settings()
    return bool(s.replicate_api_token and s.replicate_splat_model)


def poses_to_transforms_json(
    camera_poses: dict[str, list],
    frames_dir: Path,
    fov_deg: float | None,
    *,
    world_to_camera: bool = False,
) -> dict:
    """Pure adapter: depth_fusion poses -> instant-ngp/COLMAP transforms.json.

    depth_fusion emits 4x4 cam-to-world matrices in OpenCV axes WITHOUT the Y/Z
    flip it applies to the exported geometry, while points.ply HAS the flip
    (depth_fusion.py:344-345). To put poses in the same Y-up frame as the init
    cloud, left-multiply each cam-to-world by `diag(1,-1,-1,1)` on the world
    side. instant-ngp `transforms.json` stores camera-to-world, so we keep it as
    such; set `world_to_camera=True` for trainers that want the inverse.

    Returns a transforms.json dict with one `frames` entry per pose (file_path +
    transform_matrix), plus `camera_angle_x` if a horizontal FOV is known.
    """
    frame_files = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    out_frames: list[dict] = []
    for idx, key in enumerate(sorted(camera_poses)):
        cam_to_world = _YUP_FLIP @ np.asarray(camera_poses[key], dtype=np.float64)
        matrix = np.linalg.inv(cam_to_world) if world_to_camera else cam_to_world
        # Pair each pose with its frame file when available (poses are ordered
        # frame_0000.., matching the sorted frame list).
        file_path = frame_files[idx].name if idx < len(frame_files) else f"{key}.jpg"
        out_frames.append({
            "file_path": file_path,
            "transform_matrix": matrix.tolist(),
        })

    transforms: dict = {"frames": out_frames}
    if fov_deg is not None and np.isfinite(fov_deg) and fov_deg > 0:
        transforms["camera_angle_x"] = float(np.deg2rad(fov_deg))
    return transforms


@register
class SplatBackend(ReconstructionBackend):
    name = "splat"
    requires_gpu = True  # stage A (depth_fusion) needs a GPU; training is cloud
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
                "REPLICATE_SPLAT_MODEL to a Replicate trainer that takes posed "
                "multi-view frames + an init point cloud (from the depth_fusion "
                "upstream stage) and outputs a trained .ply/.splat. Replicate "
                "has no reliable public multi-view splat trainer, so this must "
                "be supplied (a hosted model or your own deployment)."
            )

        # ---- Stage A: consume depth_fusion (read-only black box) ----
        # Run depth_fusion to obtain camera poses + an init point cloud. We only
        # read its returned ReconstructionOutput + the files it writes; we never
        # touch depth_fusion's internals.
        progress_cb(0.05, "running depth_fusion (poses + init cloud)")
        df_out_dir = out_dir / "depth_fusion"
        df_out = DepthFusionBackend().reconstruct(
            inp,
            df_out_dir,
            lambda p, m: progress_cb(0.05 + 0.45 * p, f"depth_fusion: {m}"),
        )
        if not df_out.camera_poses:
            raise RuntimeError(
                "depth_fusion returned no camera poses; cannot build a posed "
                "trainer bundle for the splat stage."
            )
        if df_out.point_cloud_path is None:
            raise RuntimeError("depth_fusion returned no init point cloud (points.ply).")

        # ---- Adapter: poses (OpenCV cam->world) -> Y-up transforms.json ----
        progress_cb(0.55, "adapting poses to transforms.json")
        fov_deg = df_out.backend_meta.get("fov_deg")
        transforms = poses_to_transforms_json(
            df_out.camera_poses, inp.frames_dir, fov_deg, world_to_camera=False
        )

        # Stage the trainer input bundle: frames + transforms.json + init cloud.
        bundle_dir = out_dir / "bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        transforms_path = bundle_dir / "transforms.json"
        transforms_path.write_text(json.dumps(transforms, indent=2))
        init_cloud_path = df_out.point_cloud_path  # points.ply, already Y-up

        # ---- Stage B: train gaussians on Replicate ----
        # Input shape depends on the chosen trainer; align keys with your model.
        progress_cb(0.65, f"training gaussians on Replicate ({settings.replicate_splat_model})")
        with open(transforms_path, "rb") as tf, open(init_cloud_path, "rb") as cf:
            splat_out = rep.run_model(
                settings.replicate_splat_model,
                {"transforms": tf, "point_cloud": cf},
            )

        progress_cb(0.90, "downloading splat")
        if isinstance(splat_out, dict):
            splat_uri = splat_out.get("ply") or splat_out.get("output") or splat_out
        else:
            splat_uri = splat_out
        ply_path = rep.download(splat_uri, out_dir / "splat.ply")

        progress_cb(0.98, "done")
        return ReconstructionOutput(
            mesh_path=ply_path,
            point_cloud_path=ply_path,
            camera_poses=df_out.camera_poses,
            backend_meta={
                "actual_backend": "splat",
                "device": "replicate-cloud",
                "geometry": "gaussian_splat",
                "splat_model": settings.replicate_splat_model,
                # depth_fusion provenance.
                "upstream": "depth_fusion",
                "depth_fusion_meta": df_out.backend_meta,
            },
        )
