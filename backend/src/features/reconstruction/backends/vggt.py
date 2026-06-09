"""VGGT — feed-forward neural reconstruction, run on Replicate (cloud GPU).

VGGT is a 1B-param model whose global attention is O(n²) over every frame's
patches at once. Loading it locally OOMs and crashes laptops, so this backend
NEVER runs it on the worker. Instead it ships frames to Replicate
(`vufinder/vggt-1b`, an L40S), downloads the per-frame world-point grids, and
does the lightweight numpy/trimesh meshing locally.

The backend reports `implemented` only when a Replicate token is configured
(see `_replicate.replicate_available`). To demo the pipeline without cloud
access, select the `demo_fixture` backend.
"""
from __future__ import annotations

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

# Cap the back-projected grid's longer side per frame so the concatenated
# whole-scan mesh stays viewer-loadable (matches the depth_fusion budget).
_MAX_GRID_SIDE = 240
# Frames sent to VGGT in one run. The model resizes to ≤518px and runs on a
# cloud L40S, so this is a cost/latency knob, not a local-memory limit.
_MAX_FRAMES = 32


def _grid_to_mesh(
    points: np.ndarray,
    color01: np.ndarray,
    conf: np.ndarray,
    conf_thresh: float,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate a VGGT (H,W,3) world-point grid into a colored mesh.

    Skips low-confidence pixels and the longest ~10% of edges (depth
    discontinuities → stretched ghost faces). Pure numpy/trimesh — no open3d,
    which has no Python 3.13 wheels. `color01` is (H,W,3) in [0,1].
    """
    if stride > 1:
        points = points[::stride, ::stride]
        color01 = color01[::stride, ::stride]
        conf = conf[::stride, ::stride]
    H, W, _ = points.shape
    verts = points.reshape(-1, 3).astype(np.float64)
    cols = np.clip(color01.reshape(-1, 3) * 255.0, 0, 255).astype(np.uint8)

    valid = conf >= conf_thresh
    idx = np.arange(H * W, dtype=np.int64).reshape(H, W)
    i0, i1 = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    i2, i3 = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    v0, v1 = valid[:-1, :-1].ravel(), valid[:-1, 1:].ravel()
    v2, v3 = valid[1:, :-1].ravel(), valid[1:, 1:].ravel()

    p = verts

    def edge(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.linalg.norm(p[a] - p[b], axis=1)

    span = np.maximum.reduce([
        edge(i0, i1), edge(i0, i2), edge(i0, i3),
        edge(i1, i2), edge(i1, i3), edge(i2, i3),
    ])
    # Scale-free discontinuity cutoff: VGGT world units are arbitrary, so use a
    # percentile of edge lengths rather than a fixed metric threshold.
    finite = span[np.isfinite(span)]
    span_thresh = float(np.quantile(finite, 0.90)) if finite.size else np.inf
    ok = v0 & v1 & v2 & v3 & (span <= span_thresh)
    faces = np.concatenate([
        np.stack([i0[ok], i2[ok], i1[ok]], axis=1),
        np.stack([i1[ok], i2[ok], i3[ok]], axis=1),
    ], axis=0)
    return verts.astype(np.float32), faces.astype(np.int64), cols


def _as_hw3_color(image: np.ndarray) -> np.ndarray:
    """Normalize a decoded VGGT image array to (H,W,3) float in [0,1]."""
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = arr.transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:  # uint8-style 0..255
        arr = arr / 255.0
    return arr[..., :3]


@register
class VGGTBackend(ReconstructionBackend):
    name = "vggt"
    requires_gpu = False  # inference runs on Replicate's GPU, not the worker
    implemented = rep.replicate_available()

    def reconstruct(
        self,
        inp: ReconstructionInput,
        out_dir: Path,
        progress_cb: Callable[[float, str], None],
    ) -> ReconstructionOutput:
        out_dir.mkdir(parents=True, exist_ok=True)
        settings = get_settings()
        model_ref = settings.replicate_vggt_model

        frame_paths = sorted(inp.frames_dir.glob("*.jpg")) + sorted(inp.frames_dir.glob("*.png"))
        if not frame_paths:
            raise RuntimeError(f"no frames in {inp.frames_dir}")
        if len(frame_paths) > _MAX_FRAMES:
            sel = np.linspace(0, len(frame_paths) - 1, _MAX_FRAMES).round().astype(int)
            frame_paths = [frame_paths[i] for i in sorted(set(sel.tolist()))]

        progress_cb(0.10, f"uploading {len(frame_paths)} frames to {model_ref}")
        progress_cb(0.20, "running VGGT on Replicate (cloud GPU)")
        # VETTED UPGRADE PATH (not applied here, deliberately deferred):
        # `vufinder/map-anything` is a metric, same-owner hosted upgrade. The
        # swap is ~a per-frame key rename in the download loop below:
        #   world_points -> pts3d, world_points_conf -> conf (image unchanged).
        # BUT before switching, handle these caveats:
        #   (a) map-anything resizes inputs to a fixed 518x518, changing grid
        #       dimensions and thus the _MAX_GRID_SIDE / stride math above;
        #   (b) it is METRIC (vs vggt-1b's scale-invariant) output, so
        #       _grid_to_mesh's percentile edge-length discontinuity cutoff
        #       (vggt.py span_thresh) should become a real metric threshold —
        #       a blind rename would silently regress mesh quality;
        #   (c) it returns camera_poses + intrinsics the current code ignores.
        # Verify each against live output before flipping config.py's default.
        output = rep.run_model(
            model_ref,
            {
                "inputs": list(frame_paths),
                "to_base64": True,
                "return_pcd": True,
                "pcd_source": "point_head",
            },
        )

        # Output: {"data": [uri per image], "point_cloud": uri}. Each data JSON
        # carries base64 world_points (H,W,3), world_points_conf (H,W), image.
        data_uris = output.get("data") if isinstance(output, dict) else None
        if not data_uris:
            raise RuntimeError(f"VGGT (Replicate) returned no per-image data: {output!r}")

        progress_cb(0.55, f"downloading {len(data_uris)} world-point grids")
        per_frame = []
        for uri in data_uris:
            d = rep.fetch_json(uri)
            wp = rep.decode_array(d["world_points"])
            conf = rep.decode_array(d["world_points_conf"])
            img = _as_hw3_color(rep.decode_array(d["image"]))
            per_frame.append((wp.astype(np.float32), conf.astype(np.float32), img))

        # Confidence threshold + stride shared across frames so the whole-scan
        # mesh stays viewer-loadable.
        all_conf = np.concatenate([c.ravel() for _, c, _ in per_frame])
        conf_thresh = float(np.quantile(all_conf, 0.30))
        max_side = max(max(wp.shape[0], wp.shape[1]) for wp, _, _ in per_frame)
        stride = max(1, -(-max_side // _MAX_GRID_SIDE))  # ceil division

        progress_cb(0.70, "building mesh from world points")
        all_v: list[np.ndarray] = []
        all_f: list[np.ndarray] = []
        all_c: list[np.ndarray] = []
        offset = 0
        n = len(per_frame)
        for s, (wp, conf, img) in enumerate(per_frame):
            v, f, c = _grid_to_mesh(wp, img, conf, conf_thresh=conf_thresh, stride=stride)
            if f.shape[0]:
                all_v.append(v)
                all_c.append(c)
                all_f.append(f + offset)
                offset += v.shape[0]
            progress_cb(0.70 + 0.25 * (s + 1) / n, f"meshing frame {s + 1}/{n}")

        if not all_v:
            raise RuntimeError("VGGT produced no confident geometry")
        verts = np.concatenate(all_v, axis=0)
        faces = np.concatenate(all_f, axis=0)
        colors = np.concatenate(all_c, axis=0)

        mesh_path = out_dir / "mesh.ply"
        pc_path = out_dir / "point_cloud.ply"
        trimesh.Trimesh(
            vertices=verts.astype(np.float64), faces=faces,
            vertex_colors=colors, process=False,
        ).export(str(mesh_path))
        trimesh.points.PointCloud(verts.astype(np.float64), colors=colors).export(str(pc_path))

        progress_cb(0.98, "done")
        return ReconstructionOutput(
            mesh_path=mesh_path,
            point_cloud_path=pc_path,
            camera_poses=None,
            backend_meta={
                "actual_backend": "vggt",
                "device": "replicate-cloud",
                "model": model_ref,
                "n_frames": len(frame_paths),
                "vertices": int(verts.shape[0]),
                "faces": int(faces.shape[0]),
                "conf_thresh": conf_thresh,
            },
        )
