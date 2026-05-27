"""Demo backend that always produces the procedural sample room.

Lets the full pipeline (upload → reconstruct → validate → build → train →
replay) be exercised end-to-end without a GPU, real model weights, or even
a valid input video. The "reconstruction" produced is the same procedural
mesh `rl_env.sample_room` writes — sized roughly to the number of frames
so different videos give visibly different scenes.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import trimesh

from rl_env.sample_room import make_sample_room
from src.features.reconstruction.backends import register
from src.features.reconstruction.backends.base import (
    ReconstructionBackend,
    ReconstructionInput,
    ReconstructionOutput,
)


@register
class DemoFixtureBackend(ReconstructionBackend):
    name = "demo_fixture"
    requires_gpu = False
    implemented = True

    def reconstruct(
        self,
        inp: ReconstructionInput,
        out_dir: Path,
        progress_cb: Callable[[float, str], None],
    ) -> ReconstructionOutput:
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = list(inp.frames_dir.iterdir())
        n_frames = len(frames)

        progress_cb(0.20, f"saw {n_frames} frames; building fixture room")
        time.sleep(0.3)  # so the progress bar is visible

        # Vary room size with frame count so different uploads produce
        # visibly different scenes.
        size_x = 4.0 + min(n_frames, 32) * 0.05
        size_z = 3.0 + min(n_frames, 32) * 0.04
        mesh = make_sample_room(size=(size_x, 3.0, size_z), seed=n_frames)

        mesh_path = out_dir / "mesh.ply"
        mesh.export(str(mesh_path))
        progress_cb(0.80, "exporting point cloud")

        cloud = trimesh.PointCloud(mesh.vertices)
        pc_path = out_dir / "point_cloud.ply"
        cloud.export(str(pc_path))

        progress_cb(0.98, "done")
        return ReconstructionOutput(
            mesh_path=mesh_path,
            point_cloud_path=pc_path,
            camera_poses=None,
            backend_meta={
                "actual_backend": "demo_fixture",
                "n_frames_input": n_frames,
                "size": [size_x, 3.0, size_z],
            },
        )
