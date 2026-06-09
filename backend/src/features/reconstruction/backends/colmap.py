"""Classical COLMAP Structure-from-Motion backend, run on Replicate (cloud).

Uses `jimothyjohn/colmap` (video → NeRF-ready COLMAP output). That model takes
a VIDEO, so we re-encode the sampled frames into an mp4 with ffmpeg and upload
it. COLMAP is an SfM step: it yields camera poses + a SPARSE point cloud, not a
dense mesh. We surface the sparse cloud as the geometry artifact and the poses
in `camera_poses`. (Dense MVS + meshing would be a separate cloud step.)

Reports `implemented` only when a Replicate token is configured.
"""
from __future__ import annotations

import json
import tarfile
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import trimesh

from src.config import get_settings
from src.features.reconstruction.backends import _replicate as rep
from src.features.reconstruction.backends import register
from src.features.reconstruction.backends.base import (
    ReconstructionBackend,
    ReconstructionInput,
    ReconstructionOutput,
)


def frames_to_video(frames_dir: Path, fps: float, dest: Path) -> Path:
    """Re-encode sampled frames into an mp4 for video-input cloud models."""
    import ffmpeg  # ffmpeg-python, declared in pyproject

    pattern = None
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        if next(frames_dir.glob(ext), None) is not None:
            pattern = ext
            break
    if pattern is None:
        raise RuntimeError(f"no frames in {frames_dir} to build a video from")

    dest.parent.mkdir(parents=True, exist_ok=True)
    (
        ffmpeg
        .input(str(frames_dir / pattern), pattern_type="glob", framerate=max(fps, 1.0))
        .output(str(dest), vcodec="libx264", pix_fmt="yuv420p", r=max(fps, 1.0))
        .overwrite_output()
        .run(quiet=True)
    )
    return dest


def _extract_archive(path: Path, dest: Path) -> Path:
    """Extract a downloaded COLMAP archive (zip or tar) into `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(dest)
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            tf.extractall(dest)
    else:
        raise RuntimeError(f"COLMAP output {path} is neither zip nor tar")
    return dest


def _poses_from_transforms(extracted: Path) -> dict | None:
    """Read instant-ngp/nerfstudio transforms.json into a camera_poses dict."""
    tj = next(extracted.rglob("transforms*.json"), None)
    if tj is None:
        return None
    data = json.loads(tj.read_text())
    frames = data.get("frames", [])
    poses: dict[str, list] = {}
    for i, fr in enumerate(sorted(frames, key=lambda f: f.get("file_path", ""))):
        m = fr.get("transform_matrix")
        if m is not None:
            poses[f"frame_{i:04d}"] = m
    return poses or None


def _sparse_cloud_ply(extracted: Path, dest: Path) -> Path | None:
    """Find a point cloud in the COLMAP archive and normalize it to a .ply.

    Prefers an existing .ply; otherwise tries COLMAP's sparse points3D.txt.
    """
    ply = next(extracted.rglob("*.ply"), None)
    if ply is not None:
        return ply
    pts_txt = next(extracted.rglob("points3D.txt"), None)
    if pts_txt is None:
        return None
    xyz, rgb = [], []
    for line in pts_txt.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[...]
        xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
        rgb.append([int(parts[4]), int(parts[5]), int(parts[6])])
    if not xyz:
        return None
    cloud = trimesh.points.PointCloud(
        np.asarray(xyz, dtype=np.float64), colors=np.asarray(rgb, dtype=np.uint8)
    )
    cloud.export(str(dest))
    return dest


@register
class ColmapBackend(ReconstructionBackend):
    name = "colmap"
    requires_gpu = False  # SfM runs on Replicate
    implemented = rep.replicate_available()

    def reconstruct(
        self,
        inp: ReconstructionInput,
        out_dir: Path,
        progress_cb: Callable[[float, str], None],
    ) -> ReconstructionOutput:
        out_dir.mkdir(parents=True, exist_ok=True)
        settings = get_settings()
        model_ref = settings.replicate_colmap_model

        progress_cb(0.05, "encoding frames to video")
        with tempfile.TemporaryDirectory() as td:
            video_path = frames_to_video(inp.frames_dir, inp.fps_sampled, Path(td) / "scan.mp4")

            progress_cb(0.20, f"running COLMAP on Replicate ({model_ref})")
            with open(video_path, "rb") as vf:
                output = rep.run_model(
                    model_ref,
                    {"video": vf, "media": "video", "format": "instant-ngp",
                     "quality": "Low", "continuous": True},
                )

            progress_cb(0.70, "downloading COLMAP output")
            archive = rep.download(output, Path(td) / "colmap_out")
            extracted = _extract_archive(archive, Path(td) / "extracted")

            poses = _poses_from_transforms(extracted)
            pc_src = _sparse_cloud_ply(extracted, out_dir / "point_cloud.ply")

        if pc_src is None:
            raise RuntimeError(
                "COLMAP produced no point cloud (likely failed to register frames). "
                "Try a scan with more overlap / texture."
            )
        pc_path = out_dir / "point_cloud.ply"
        if pc_src != pc_path:
            trimesh.load(str(pc_src)).export(str(pc_path))

        # COLMAP SfM has no mesh; expose the sparse cloud as the geometry artifact.
        mesh_path = pc_path
        progress_cb(0.98, "done")
        return ReconstructionOutput(
            mesh_path=mesh_path,
            point_cloud_path=pc_path,
            camera_poses=poses,
            backend_meta={
                "actual_backend": "colmap",
                "device": "replicate-cloud",
                "model": model_ref,
                "geometry": "sparse_points",  # SfM only — no dense mesh
                "n_poses": len(poses) if poses else 0,
            },
        )
