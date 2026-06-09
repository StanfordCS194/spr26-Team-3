"""Depth-fusion reconstruction backend — server-side port of matthew's
prototype/v4.2.html pipeline.

Pass 1 (per frame):
  1. Monocular metric depth (Apple Depth-Pro by default; it also emits a
     per-frame horizontal FOV estimate). Falls back to the lighter
     Depth-Anything-V2-Metric-Indoor model if Depth-Pro can't be loaded.
  2. SuperPoint keypoints + descriptors (ONNX).
  3. Collect FOV estimates; the shared intrinsics use their median (v4.2's
     trick — a single, self-tuned FOV beats a hardcoded guess).

Pass 2 — cross-frame fusion (for each new frame i, find best previous frame j
to fuse against):
  4. LightGlue match (i, j); skip if < 8 matches.
  5. Depth calibration: robust scale-only fit of frame i's depths onto frame
     j's, correcting monocular drift before pose alignment.
  6. Pose: RANSAC + rigid Umeyama on matched keypoint 3D points
     (rigid since depth is metric — preserves world scale).
  7. Back-project each frame (on a decimated grid) into world space and
     concatenate into one colored mesh.

Outputs:
  - mesh.ply: concatenated per-frame meshes (triangles, vertex colors)
  - points.ply: same vertex cloud unindexed

Reuses matthew's depth weights via `rl_env.server` so the legacy Flask
`/api/depth` and this backend stay in sync.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from src.features.reconstruction.backends import register
from src.features.reconstruction.backends._geometry import (
    assume_intrinsics,
    back_project,
    ransac_umeyama,
    robust_linear_fit,
)
from src.features.reconstruction.backends._models import (
    extract_superpoint,
    get_depth_model,
    infer_depth,
    match_lightglue,
)
from src.features.reconstruction.backends.base import (
    ReconstructionBackend,
    ReconstructionInput,
    ReconstructionOutput,
)

log = logging.getLogger(__name__)

# Defaults match matthew's JS heuristics; tunable via ReconstructionInput.intrinsics_hint.
_MIN_MATCHES = 8
_GOOD_MATCHES = 500  # stop searching backward once a frame has this many matches
_FOV_DEG_DEFAULT = 60.0
_DISCONTINUITY_M = 0.30
_RANSAC_ITERS = 200
_RANSAC_THRESH_M = 0.15
_RANSAC_MIN_INLIERS = 10
# Cap the *total* meshing budget so a fused mesh stays renderable in the
# browser regardless of frame count (a full-res 24-frame fuse is ~100M faces,
# and even a per-frame 640px cap balloons to ~300MB at 24 frames). We spread a
# fixed vertex budget across frames; metric 3D points are preserved when
# decimating — only the triangle density drops.
_MESH_VERT_BUDGET = 1_800_000  # total vertices across all frames
_MESH_MIN_PIX = 20_000  # never decimate a single frame below this (≈170x118)


def _sample_depth_at(depth: np.ndarray, kp: np.ndarray) -> np.ndarray:
    """Sample depth at keypoint pixel coordinates with nearest-neighbor.
    Returns 0 for out-of-bounds or invalid (≤0) samples so callers can mask."""
    h, w = depth.shape
    u = np.clip(np.round(kp[:, 0]).astype(np.int64), 0, w - 1)
    v = np.clip(np.round(kp[:, 1]).astype(np.int64), 0, h - 1)
    return depth[v, u]


def _kp_to_camera_points(kp: np.ndarray, z: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Lift 2D keypoints + per-keypoint depth into camera-frame 3D points."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (kp[:, 0] - cx) * z / fx
    Y = (kp[:, 1] - cy) * z / fy
    return np.stack([X, Y, z], axis=1)


def _downsample_for_mesh(
    depth: np.ndarray, color: np.ndarray, fov_deg: float, max_pixels: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Down-sample a (calibrated) depth map + color so the grid has at most
    `max_pixels` cells, and return (depth_ds, color_ds, K_ds).

    The back-projected metric 3D points are unchanged because the intrinsics
    scale with resolution (fx, cx ∝ width); only the triangle grid coarsens.
    Depth uses nearest-neighbor to keep object edges crisp for the
    discontinuity filter; color uses bilinear.
    """
    h, w = depth.shape
    s = min(1.0, (float(max_pixels) / float(w * h)) ** 0.5)
    if s < 1.0:
        new_w = max(2, int(round(w * s)))
        new_h = max(2, int(round(h * s)))
        depth_ds = np.asarray(
            Image.fromarray(depth.astype(np.float32)).resize((new_w, new_h), Image.NEAREST),
            dtype=np.float32,
        )
        color_ds = np.asarray(
            Image.fromarray(color.astype(np.uint8)).resize((new_w, new_h), Image.BILINEAR),
            dtype=np.uint8,
        )
    else:
        new_w, new_h = w, h
        depth_ds = depth.astype(np.float32)
        color_ds = color.astype(np.uint8)
    K_ds = assume_intrinsics(new_w, new_h, fov_deg=fov_deg)
    return depth_ds, color_ds, K_ds


def _write_ply_mesh(verts: np.ndarray, faces: np.ndarray, colors: np.ndarray, path: Path) -> None:
    mesh = trimesh.Trimesh(
        vertices=verts.astype(np.float64),
        faces=faces.astype(np.int64),
        vertex_colors=colors.astype(np.uint8),
        process=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def _write_ply_points(verts: np.ndarray, colors: np.ndarray, path: Path) -> None:
    cloud = trimesh.points.PointCloud(verts.astype(np.float64), colors=colors.astype(np.uint8))
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud.export(path)


@register
class DepthFusionBackend(ReconstructionBackend):
    name = "depth_fusion"
    requires_gpu = True
    implemented = True

    def reconstruct(
        self,
        inp: ReconstructionInput,
        out_dir: Path,
        progress_cb: Callable[[float, str], None],
    ) -> ReconstructionOutput:
        frames_dir = inp.frames_dir
        frame_paths = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if not frame_paths:
            raise RuntimeError(f"no frames found under {frames_dir}")

        out_dir.mkdir(parents=True, exist_ok=True)

        hint = inp.intrinsics_hint or {}
        depth_model = hint.get("depth_model", "pro")
        # Resolve the depth model once; fall back to the lighter indoor model if
        # Depth-Pro can't be loaded (download/disk/OOM on a GPU-less host).
        try:
            get_depth_model(depth_model)
        except Exception as e:  # noqa: BLE001 — any load failure → fallback
            log.warning(
                "[depth_fusion] depth model %r unavailable (%s); falling back to 'indoor'",
                depth_model, e,
            )
            depth_model = "indoor"

        n = len(frame_paths)
        # Spread the total vertex budget across frames so the fused mesh stays
        # renderable whether there are 1 or 24 of them.
        mesh_max_pixels = max(_MESH_MIN_PIX, _MESH_VERT_BUDGET // n)

        # ---- Pass 1: per-frame depth + SuperPoint features + FOV estimate ----
        frames: list[dict] = []
        fov_estimates: list[float] = []
        for i, fp in enumerate(frame_paths):
            img = Image.open(fp).convert("RGB")
            w, h = img.size
            depth, depth_meta = infer_depth(img, name=depth_model)
            if depth.shape != (h, w):
                # Resize depth onto the image grid if the model emitted a different size.
                depth = np.asarray(
                    Image.fromarray(depth).resize((w, h), Image.BILINEAR),
                    dtype=np.float32,
                )
            feat = extract_superpoint(img)
            fov_i = depth_meta.get("fov_deg")
            if fov_i is not None and np.isfinite(fov_i) and fov_i > 0:
                fov_estimates.append(float(fov_i))
            frames.append({"color": np.asarray(img), "w": w, "h": h, "depth": depth, "feat": feat})
            progress_cb(0.5 * (i + 1) / n, f"depth {i + 1}/{n}")

        # Shared horizontal FOV: median of per-frame estimates (Depth-Pro), else
        # an explicit hint, else 60° (matches the prototype's slider default).
        if fov_estimates:
            fov_deg = float(np.median(fov_estimates))
            fov_src = f"median of {len(fov_estimates)} estimate(s)"
        else:
            fov_deg = float(hint.get("fov_deg", _FOV_DEG_DEFAULT))
            fov_src = "hint/default (no per-frame estimate)"
        log.info("[depth_fusion] %d frames, depth=%s, fov=%.1f° (%s)", n, depth_model, fov_deg, fov_src)

        # ---- Pass 2: fuse frames into a common world frame ----
        per_frame: list[dict] = []
        all_verts: list[np.ndarray] = []
        all_faces: list[np.ndarray] = []
        all_colors: list[np.ndarray] = []
        vertex_offset = 0
        n_fused = 0
        n_failed_fusion = 0
        inlier_counts: list[int] = []

        for i, fr in enumerate(frames):
            t_frame = time.time()
            w, h = fr["w"], fr["h"]
            depth = fr["depth"]
            feat = fr["feat"]
            K = assume_intrinsics(w, h, fov_deg=fov_deg)

            world_T = np.eye(4, dtype=np.float64)  # frame 0 anchors world at identity
            if i > 0:
                # Search backward for the best previously-fused frame to fuse onto.
                best_j: int | None = None
                best_matches: np.ndarray | None = None
                for j in range(i - 1, -1, -1):
                    matches = match_lightglue(per_frame[j]["feat"], feat, image_size=(w, h))
                    if matches.shape[0] < _MIN_MATCHES:
                        continue
                    if best_matches is None or matches.shape[0] > best_matches.shape[0]:
                        best_matches = matches
                        best_j = j
                    if matches.shape[0] >= _GOOD_MATCHES:
                        break

                if best_j is None or best_matches is None:
                    log.warning("[depth_fusion] frame %d: <%d matches against any previous; placing offset", i, _MIN_MATCHES)
                    world_T[:3, 3] = np.array([0.0, 0.0, 1.0 * i])  # naive lateral offset
                    n_failed_fusion += 1
                else:
                    ref = per_frame[best_j]
                    idx_j, idx_i = best_matches[:, 0], best_matches[:, 1]
                    kp_j = ref["feat"]["keypoints"][idx_j]
                    kp_i = feat["keypoints"][idx_i]
                    z_j_at_kp = _sample_depth_at(ref["depth"], kp_j)
                    z_i_at_kp = _sample_depth_at(depth, kp_i)

                    # Depth calibration: rescale frame i's depths into frame j's frame.
                    k_scale, c_off = robust_linear_fit(z_j_at_kp, z_i_at_kp)
                    depth = np.clip(k_scale * depth + c_off, 0.0, None)
                    z_i_at_kp = k_scale * z_i_at_kp + c_off

                    # Build paired 3D points: frame j in world, frame i in camera_i.
                    pts_i = _kp_to_camera_points(kp_i, z_i_at_kp, K)
                    pts_j_cam = _kp_to_camera_points(kp_j, z_j_at_kp, K)
                    pts_j_world = (ref["world_T"][:3, :3] @ pts_j_cam.T).T + ref["world_T"][:3, 3]

                    # Mask invalid depths (≤0 at keypoints).
                    valid = (z_i_at_kp > 0) & (z_j_at_kp > 0)
                    if valid.sum() < _RANSAC_MIN_INLIERS:
                        log.warning(
                            "[depth_fusion] frame %d → %d: only %d valid matched depths; offset fallback",
                            i, best_j, int(valid.sum()),
                        )
                        world_T = ref["world_T"].copy()
                        world_T[:3, 3] = world_T[:3, 3] + np.array([0.3, 0.0, 0.0])
                        n_failed_fusion += 1
                    else:
                        T, inliers = ransac_umeyama(
                            pts_i[valid], pts_j_world[valid],
                            iters=_RANSAC_ITERS, inlier_thresh_m=_RANSAC_THRESH_M, min_inliers=_RANSAC_MIN_INLIERS,
                        )
                        if T is None:
                            log.warning(
                                "[depth_fusion] frame %d → %d: RANSAC failed (%d valid pts); offset fallback",
                                i, best_j, int(valid.sum()),
                            )
                            world_T = ref["world_T"].copy()
                            world_T[:3, 3] = world_T[:3, 3] + np.array([0.3, 0.0, 0.0])
                            n_failed_fusion += 1
                        else:
                            world_T = T
                            n_fused += 1
                            inlier_counts.append(int(inliers.sum()))
                            log.info(
                                "[depth_fusion] frame %d → ref %d: %d matches, %d inliers, k=%.3f",
                                i, best_j, best_matches.shape[0], int(inliers.sum()), k_scale,
                            )

            # Back-project this frame into world space on a decimated grid.
            depth_ds, color_ds, K_ds = _downsample_for_mesh(depth, fr["color"], fov_deg, mesh_max_pixels)
            verts, faces, colors = back_project(
                depth=depth_ds,
                color=color_ds,
                intrinsics=K_ds,
                world_transform=world_T,
                discontinuity_m=_DISCONTINUITY_M,
            )
            if faces.shape[0] > 0:
                all_verts.append(verts)
                all_colors.append(colors)
                all_faces.append(faces + vertex_offset)
                vertex_offset += verts.shape[0]

            per_frame.append({
                "feat": feat,
                "depth": depth,
                "world_T": world_T,
            })

            elapsed = time.time() - t_frame
            log.info("[depth_fusion] frame %d/%d fused in %.2fs (%d verts)", i + 1, n, elapsed, verts.shape[0])
            progress_cb(0.5 + 0.5 * (i + 1) / n, f"fuse {i + 1}/{n}")

        # With multiple frames, zero successful fusions means the pipeline failed
        # (bad matches / wrong intrinsics). Don't emit a giant pile of unaligned
        # per-frame meshes marked "ok" — surface a real failure instead.
        if n > 1 and n_fused == 0:
            raise RuntimeError(
                f"reconstruction failed: 0/{n - 1} frames fused (no inlier pose found). "
                "Use a video/photos with more overlap and texture, or try the indoor model."
            )

        if not all_verts:
            raise RuntimeError("depth_fusion produced no geometry — every frame failed back-projection")

        verts = np.concatenate(all_verts, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        faces = np.concatenate(all_faces, axis=0)

        # back_project works in OpenCV camera axes (X right, Y down, Z forward).
        # The viewer and the downstream physics build both expect Y up, so flip
        # the whole scene into that convention (matches prototype/v4.2). Negating
        # Y *and* Z is a 180° rotation about X — it preserves face winding, so no
        # normals/winding fix is needed.
        verts[:, 1] *= -1.0
        verts[:, 2] *= -1.0

        mesh_path = out_dir / "mesh.ply"
        points_path = out_dir / "points.ply"
        _write_ply_mesh(verts, faces, colors, mesh_path)
        _write_ply_points(verts, colors, points_path)

        backend_meta = {
            "depth_model": depth_model,
            "fov_deg": fov_deg,
            "n_frames": n,
            "n_fused": n_fused,
            "n_failed_fusion": n_failed_fusion,
            "avg_inliers": float(np.mean(inlier_counts)) if inlier_counts else 0.0,
            "vertices": int(verts.shape[0]),
            "faces": int(faces.shape[0]),
        }
        log.info("[depth_fusion] done: %s", backend_meta)

        return ReconstructionOutput(
            mesh_path=mesh_path,
            point_cloud_path=points_path,
            camera_poses={f"frame_{i:04d}": f["world_T"].tolist() for i, f in enumerate(per_frame)},
            backend_meta=backend_meta,
        )
