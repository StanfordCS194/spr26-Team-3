"""Shared pipeline for feed-forward multi-view reconstructors on Replicate.

VGGT, π³ (map-anything-pi3) and MapAnything are all `vufinder` models with the
same envelope: take an `inputs` array of frames, run one feed-forward pass on a
cloud GPU, and return per-image JSONs with a world-point grid + confidence +
image. They differ only in the JSON key names and whether output is metric, so
one routine serves all three — each backend is a thin wrapper that passes its
model slug.

The heavy model runs on Replicate (never locally); only the lightweight
numpy/trimesh meshing runs on the worker. The discontinuity cutoff is a
percentile of edge lengths (scale-free), so it works for both metric
(MapAnything) and scale-invariant (VGGT/π³) output without special-casing.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import trimesh

from src.features.reconstruction.backends import _replicate as rep
from src.features.reconstruction.backends.base import (
    ReconstructionInput,
    ReconstructionOutput,
)

# Cap the back-projected grid's longer side per frame so the concatenated
# whole-scan mesh stays viewer-loadable.
_MAX_GRID_SIDE = 240
# Frames sent in one run. Models resize to ≤518px and run on a cloud GPU, so
# this is a cost/latency knob, not a local-memory limit.
_MAX_FRAMES = 32

# JSON key candidates across the vufinder models (vggt uses world_points/
# world_points_conf; map-anything/π³ use pts3d/conf). Tried in order.
_POINT_KEYS = ("world_points", "pts3d", "points", "world_pts", "pointmap", "point_map")
_CONF_KEYS = ("world_points_conf", "conf", "confidence", "pts3d_conf", "point_conf")
_IMAGE_KEYS = ("image", "rgb", "original_image")


def _grid_to_mesh(
    points: np.ndarray,
    color01: np.ndarray,
    conf: np.ndarray,
    conf_thresh: float,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate an (H,W,3) world-point grid into a colored mesh.

    Skips low-confidence pixels and the longest ~10% of edges (depth
    discontinuities → stretched ghost faces). Pure numpy/trimesh. `color01` is
    (H,W,3) in [0,1].
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
    finite = span[np.isfinite(span)]
    span_thresh = float(np.quantile(finite, 0.90)) if finite.size else np.inf
    ok = v0 & v1 & v2 & v3 & (span <= span_thresh)
    faces = np.concatenate([
        np.stack([i0[ok], i2[ok], i1[ok]], axis=1),
        np.stack([i1[ok], i2[ok], i3[ok]], axis=1),
    ], axis=0)
    return verts.astype(np.float32), faces.astype(np.int64), cols


def _as_hw3_color(image: np.ndarray) -> np.ndarray:
    """Normalize a decoded image array to (H,W,3) float in [0,1]."""
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = arr.transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:  # uint8-style 0..255
        arr = arr / 255.0
    return arr[..., :3]


def _squeeze_leading(arr: np.ndarray, ndim: int) -> np.ndarray:
    """Drop leading singleton dims until arr has `ndim` dims (e.g. (1,H,W,3))."""
    while arr.ndim > ndim and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _find_points(d: dict) -> np.ndarray:
    for k in _POINT_KEYS:
        if k in d:
            arr = _squeeze_leading(rep.decode_array(d[k]), 3)
            if arr.ndim == 3 and arr.shape[-1] == 3:
                return arr.astype(np.float32)
    raise RuntimeError(f"no (H,W,3) world-point grid in output keys: {list(d)}")


def _find_conf(d: dict, hw: tuple[int, int]) -> np.ndarray:
    for k in _CONF_KEYS:
        if k in d:
            arr = _squeeze_leading(rep.decode_array(d[k]), 2)
            if arr.shape == hw:
                return arr.astype(np.float32)
    # Some models omit confidence — treat every pixel as valid.
    return np.ones(hw, dtype=np.float32)


def _find_image(d: dict, hw: tuple[int, int]) -> np.ndarray:
    for k in _IMAGE_KEYS:
        if k in d:
            return _as_hw3_color(rep.decode_array(d[k]))
    # No image channel — fall back to flat gray so meshing still works.
    return np.full((*hw, 3), 0.6, dtype=np.float32)


def run_feedforward(
    model_ref: str,
    backend_name: str,
    inp: ReconstructionInput,
    out_dir: Path,
    progress_cb: Callable[[float, str], None],
) -> ReconstructionOutput:
    """Run a vufinder feed-forward reconstructor and mesh its world points."""
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(inp.frames_dir.glob("*.jpg")) + sorted(inp.frames_dir.glob("*.png"))
    if not frame_paths:
        raise RuntimeError(f"no frames in {inp.frames_dir}")
    if len(frame_paths) > _MAX_FRAMES:
        sel = np.linspace(0, len(frame_paths) - 1, _MAX_FRAMES).round().astype(int)
        frame_paths = [frame_paths[i] for i in sorted(set(sel.tolist()))]

    progress_cb(0.10, f"uploading {len(frame_paths)} frames to {model_ref}")
    progress_cb(0.20, f"running {backend_name} on Replicate (cloud GPU)")
    output = rep.run_model(
        model_ref,
        {"inputs": list(frame_paths), "to_base64": True, "return_pcd": True},
    )

    data_uris = output.get("data") if isinstance(output, dict) else None
    if not data_uris:
        raise RuntimeError(f"{backend_name} (Replicate) returned no per-image data: {output!r}")

    progress_cb(0.55, f"downloading {len(data_uris)} world-point grids")
    per_frame = []
    for uri in data_uris:
        d = rep.fetch_json(uri)
        wp = _find_points(d)
        hw = (wp.shape[0], wp.shape[1])
        conf = _find_conf(d, hw)
        img = _find_image(d, hw)
        per_frame.append((wp, conf, img))

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
        raise RuntimeError(f"{backend_name} produced no confident geometry")
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
            "actual_backend": backend_name,
            "device": "replicate-cloud",
            "model": model_ref,
            "n_frames": len(frame_paths),
            "vertices": int(verts.shape[0]),
            "faces": int(faces.shape[0]),
            "conf_thresh": conf_thresh,
        },
    )


__all__ = ["run_feedforward"]
