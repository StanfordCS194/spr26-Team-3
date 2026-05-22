"""VGGT — feed-forward neural reconstruction.

The `vggt` PyPI package wraps Meta's VGGT model. It needs a GPU (CUDA or
Apple MPS) and downloads weights from HuggingFace on first run.

This backend tries to load and run the real model. If the package isn't
installed, weights can't be fetched, or no compute device is available, it
raises a clear error. To demo the rest of the pipeline without GPU, users
select the `demo_fixture` backend instead.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import trimesh

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
    implemented = True

    def reconstruct(
        self,
        inp: ReconstructionInput,
        out_dir: Path,
        progress_cb: Callable[[float, str], None],
    ) -> ReconstructionOutput:
        out_dir.mkdir(parents=True, exist_ok=True)
        progress_cb(0.05, "loading VGGT model")

        try:
            import torch
            from vggt.models.vggt import VGGT
            from vggt.utils.load_fn import load_and_preprocess_images
        except ImportError as e:
            raise RuntimeError(
                f"VGGT not installed: {e}. Run `pip install vggt` in the worker container, "
                "or pick the `demo_fixture` backend to demo the pipeline without GPU."
            ) from e

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            raise RuntimeError(
                "VGGT needs a CUDA or Apple-Metal GPU. None detected on this worker. "
                "Pick the `demo_fixture` backend to demo without GPU."
            )

        progress_cb(0.10, f"using device: {device}")

        frame_paths = sorted(inp.frames_dir.glob("*.jpg")) + sorted(inp.frames_dir.glob("*.png"))
        if not frame_paths:
            raise RuntimeError(f"no frames in {inp.frames_dir}")
        progress_cb(0.15, f"loaded {len(frame_paths)} frames")

        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
        progress_cb(0.40, "running inference")

        with torch.no_grad():
            images = load_and_preprocess_images([str(p) for p in frame_paths]).to(device)
            preds = model(images)

        progress_cb(0.80, "extracting point cloud")
        pts_world = preds["world_points"].squeeze(0).cpu().numpy().reshape(-1, 3)
        # Cap to 200k points so trimesh stays snappy
        if len(pts_world) > 200_000:
            idx = np.random.choice(len(pts_world), 200_000, replace=False)
            pts_world = pts_world[idx]

        cloud = trimesh.PointCloud(pts_world)
        pc_path = out_dir / "point_cloud.ply"
        cloud.export(str(pc_path))

        progress_cb(0.90, "meshing point cloud (ball-pivoting)")
        try:
            import open3d as o3d  # type: ignore

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_world)
            pcd.estimate_normals()
            distances = pcd.compute_nearest_neighbor_distance()
            avg = float(np.mean(distances))
            radii = o3d.utility.DoubleVector([avg * 2, avg * 3])
            mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
            verts = np.asarray(mesh_o3d.vertices)
            faces = np.asarray(mesh_o3d.triangles)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        except ImportError:
            # No open3d — fall back to convex hull as the mesh
            mesh = cloud.convex_hull
        mesh_path = out_dir / "mesh.ply"
        mesh.export(str(mesh_path))

        progress_cb(0.98, "done")
        return ReconstructionOutput(
            mesh_path=mesh_path,
            point_cloud_path=pc_path,
            camera_poses=None,
            backend_meta={
                "actual_backend": "vggt",
                "device": device,
                "n_frames": len(frame_paths),
                "n_points": len(pts_world),
            },
        )
