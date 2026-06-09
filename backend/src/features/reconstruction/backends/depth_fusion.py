"""Depth-fusion reconstruction backend — server-side port of matthew's
prototype/v4.html pipeline.

Pipeline per frame:
  1. Monocular metric depth (Depth-Anything-V2-Metric-Indoor by default).
  2. SuperPoint keypoints + descriptors (ONNX).

Cross-frame fusion (for each new frame i, find best previous frame j to fuse
against):
  3. LightGlue match (i, j); skip if < 8 matches.
  4. Depth calibration: linear robust fit of frame i's depths onto frame j's,
     correcting monocular drift before pose alignment.
  5. Pose: RANSAC + rigid Umeyama on matched keypoint 3D points
     (rigid=True since depth is metric — preserves world scale).
  6. Compose into a per-frame back-projected colored mesh in world space.

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

from src.features.reconstruction.backends import _replicate as rep
from src.features.reconstruction.backends import register
from src.features.reconstruction.backends._geometry import (
    assume_intrinsics,
    back_project,
    ransac_umeyama,
    robust_linear_fit,
)
from src.features.reconstruction.backends._models import (
    extract_superpoint,
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
# 0.15m was too tight for room-scale metric depth — frames with 80+ valid
# matched points still failed to find a consensus pose. 0.5m lets honest
# correspondences survive monocular-depth noise while still rejecting outliers.
_RANSAC_THRESH_M = 0.50
_RANSAC_MIN_INLIERS = 10
# Cap the back-projected grid's longer side. A full 1080×1920 frame is ~2M
# verts; across a 24-frame scan that's a ~2GB PLY no viewer can load. Striding
# to ≤ this many samples/side keeps the whole-scan mesh in the low-millions.
_MAX_GRID_SIDE = 240


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
    requires_gpu = False  # depth runs on Replicate; only ONNX + numpy run locally
    implemented = rep.replicate_available()

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

        depth_model = (inp.intrinsics_hint or {}).get("depth_model", "indoor")
        fov_deg = float((inp.intrinsics_hint or {}).get("fov_deg", _FOV_DEG_DEFAULT))

        per_frame: list[dict] = []
        all_verts: list[np.ndarray] = []
        all_faces: list[np.ndarray] = []
        all_colors: list[np.ndarray] = []
        vertex_offset = 0
        n_fused = 0
        n_failed_fusion = 0
        inlier_counts: list[int] = []

        log.info("[depth_fusion] %d frames, depth=%s, fov=%.1f°", len(frame_paths), depth_model, fov_deg)

        for i, fp in enumerate(frame_paths):
            t_frame = time.time()
            img = Image.open(fp).convert("RGB")
            w, h = img.size
            depth, _depth_meta = infer_depth(img, name=depth_model, fov_deg=fov_deg)
            if depth.shape != (h, w):
                # Resize depth onto the image grid if the model emitted a different size.
                depth = np.asarray(
                    Image.fromarray(depth).resize((w, h), Image.BILINEAR),
                    dtype=np.float32,
                )
            feat = extract_superpoint(img)
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
                                "[depth_fusion] frame %d → ref %d: %d matches, %d inliers, k=%.3f c=%.3f",
                                i, best_j, best_matches.shape[0], int(inliers.sum()), k_scale, c_off,
                            )

            # Back-project this frame into world space and collect its mesh.
            stride = max(1, -(-max(h, w) // _MAX_GRID_SIDE))  # ceil division
            verts, faces, colors = back_project(
                depth=depth,
                color=np.asarray(img),
                intrinsics=K,
                world_transform=world_T,
                discontinuity_m=_DISCONTINUITY_M,
                stride=stride,
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
            log.info("[depth_fusion] frame %d/%d done in %.2fs (%d verts)", i + 1, len(frame_paths), elapsed, verts.shape[0])
            progress_cb((i + 1) / len(frame_paths), f"frame {i + 1}/{len(frame_paths)}")

        if not all_verts:
            raise RuntimeError("depth_fusion produced no geometry — every frame failed back-projection")

        verts = np.concatenate(all_verts, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        faces = np.concatenate(all_faces, axis=0)

        mesh_path = out_dir / "mesh.ply"
        points_path = out_dir / "points.ply"
        _write_ply_mesh(verts, faces, colors, mesh_path)
        _write_ply_points(verts, colors, points_path)

        backend_meta = {
            "depth_model": depth_model,
            "fov_deg": fov_deg,
            "n_frames": len(frame_paths),
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
