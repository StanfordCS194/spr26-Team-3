"""Pure geometry helpers for the depth-fusion reconstruction backend.

All functions are NumPy-only and pure. They mirror the math in matthew's
browser-side pipeline (`prototype/v4.html`) so the server-side port produces
equivalent reconstructions. Kept separate from depth_fusion.py for unit
testability — no model loads, no I/O.
"""
from __future__ import annotations

import numpy as np


def robust_linear_fit(z_ref: np.ndarray, z_new: np.ndarray) -> tuple[float, float]:
    """Robust scale-only fit z_ref ≈ k * z_new on matched keypoint depths.

    Even with metric depth, per-image monocular depth has scale drift; this
    corrects new-image depths into the reference frame before pose alignment.

    Scale-only (offset c fixed at 0), via the median of per-match depth ratios
    (z_ref / z_new), `k` clamped to [0.5, 2.0]. This mirrors matthew's prototype
    (`robustLinearFit` in prototype/v4.2.html): a *free* offset was tried and
    dropped because it produced extreme values when matched depths had narrow
    variance. Returns (k, 0.0).
    """
    z_ref = np.asarray(z_ref, dtype=np.float64).ravel()
    z_new = np.asarray(z_new, dtype=np.float64).ravel()
    # Prototype filters at 0.05 (metres) on both sides before taking ratios.
    mask = (z_ref > 0.05) & (z_new > 0.05)
    if mask.sum() < 4:
        return 1.0, 0.0
    ratios = z_ref[mask] / z_new[mask]
    k = float(np.clip(np.median(ratios), 0.5, 2.0))
    return k, 0.0


def umeyama_rigid(pts_a: np.ndarray, pts_b: np.ndarray) -> np.ndarray:
    """Best rigid (rotation + translation, scale fixed to 1) transform mapping
    `pts_a` to `pts_b`. Returns a 4x4 homogeneous matrix.

    Uses the Umeyama SVD solution. Forcing scale=1 is correct when both point
    sets are in metric units (Depth-Anything-V2-Metric); allowing the SVD's
    natural scale on tight keypoint clusters collapses photos into a doll
    house, per matthew's notes.
    """
    a = np.asarray(pts_a, dtype=np.float64)
    b = np.asarray(pts_b, dtype=np.float64)
    if a.shape != b.shape or a.shape[1] != 3 or a.shape[0] < 3:
        raise ValueError(f"need ≥3 paired 3D points, got {a.shape} and {b.shape}")
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    aa, bb = a - mu_a, b - mu_b
    H = aa.T @ bb / aa.shape[0]
    U, _, Vt = np.linalg.svd(H)
    # Reflection fix: ensure right-handed rotation.
    d = np.sign(np.linalg.det(U @ Vt))
    S = np.diag([1.0, 1.0, d])
    R = (U @ S @ Vt).T
    t = mu_b - R @ mu_a
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def ransac_umeyama(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    iters: int = 200,
    inlier_thresh_m: float = 0.15,
    min_inliers: int = 10,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray | None, np.ndarray]:
    """RANSAC over rigid Umeyama. Returns (4x4 transform, inlier mask) or
    (None, empty mask) if no consensus set reaches `min_inliers`.

    Defaults match matthew's JS (200 iters, 0.15 m, ≥10 inliers).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    a = np.asarray(pts_a, dtype=np.float64)
    b = np.asarray(pts_b, dtype=np.float64)
    n = a.shape[0]
    if n < 3:
        return None, np.zeros(n, dtype=bool)

    best_inliers = np.zeros(n, dtype=bool)
    best_T: np.ndarray | None = None
    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        try:
            T = umeyama_rigid(a[idx], b[idx])
        except (ValueError, np.linalg.LinAlgError):
            continue
        a_t = (T[:3, :3] @ a.T).T + T[:3, 3]
        err = np.linalg.norm(a_t - b, axis=1)
        inliers = err < inlier_thresh_m
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_T = T

    if best_inliers.sum() < min_inliers or best_T is None:
        return None, np.zeros(n, dtype=bool)

    # Refit on the full inlier set for a tighter transform.
    refit = umeyama_rigid(a[best_inliers], b[best_inliers])
    return refit, best_inliers


def back_project(
    depth: np.ndarray,
    color: np.ndarray,
    intrinsics: np.ndarray,
    world_transform: np.ndarray | None = None,
    discontinuity_m: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lift a depth grid to a colored triangle mesh.

    Returns (vertices Nx3, faces Mx3, colors Nx3 uint8). Faces are skipped at
    depth discontinuities (>= discontinuity_m between adjacent pixels) to
    avoid stretched ghost geometry around object edges.

    Coordinate convention: camera looks down +Z; X right, Y down (OpenCV).
    `intrinsics` is a 3x3 K matrix. If `world_transform` (4x4) is given,
    vertices are pre-multiplied into world frame.
    """
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth.shape}")
    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError(f"color must be HxWx3, got {color.shape}")
    if color.shape[:2] != depth.shape:
        raise ValueError(f"color {color.shape[:2]} != depth {depth.shape}")

    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    us, vs = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    z = depth.astype(np.float64)
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy

    verts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    cols = color.reshape(-1, 3).astype(np.uint8)

    if world_transform is not None:
        R = world_transform[:3, :3]
        t = world_transform[:3, 3]
        verts = (R @ verts.T).T + t

    # Triangulate the depth grid. Two triangles per cell; skip both if any
    # corner's depth is invalid or any edge crosses a discontinuity.
    faces: list[np.ndarray] = []
    idx = np.arange(H * W, dtype=np.int64).reshape(H, W)
    z_grid = z  # alias for clarity
    valid = z_grid > 0.0

    i0 = idx[:-1, :-1].ravel()  # top-left
    i1 = idx[:-1, 1:].ravel()   # top-right
    i2 = idx[1:, :-1].ravel()   # bot-left
    i3 = idx[1:, 1:].ravel()    # bot-right

    z0 = z_grid[:-1, :-1].ravel()
    z1 = z_grid[:-1, 1:].ravel()
    z2 = z_grid[1:, :-1].ravel()
    z3 = z_grid[1:, 1:].ravel()

    v0 = valid[:-1, :-1].ravel()
    v1 = valid[:-1, 1:].ravel()
    v2 = valid[1:, :-1].ravel()
    v3 = valid[1:, 1:].ravel()

    span = np.maximum.reduce([
        np.abs(z0 - z1), np.abs(z0 - z2), np.abs(z0 - z3),
        np.abs(z1 - z2), np.abs(z1 - z3), np.abs(z2 - z3),
    ])
    ok = v0 & v1 & v2 & v3 & (span < discontinuity_m)

    faces.append(np.stack([i0[ok], i2[ok], i1[ok]], axis=1))
    faces.append(np.stack([i1[ok], i2[ok], i3[ok]], axis=1))
    faces_arr = np.concatenate(faces, axis=0) if faces else np.zeros((0, 3), dtype=np.int64)

    return verts.astype(np.float32), faces_arr.astype(np.int64), cols


def assume_intrinsics(width: int, height: int, fov_deg: float = 60.0) -> np.ndarray:
    """Build a 3x3 K matrix from image size + horizontal FOV (degrees).

    Used when `ReconstructionInput.intrinsics_hint` is None — picks a sensible
    default that matches matthew's prototype assumptions (~60° hFOV, principal
    point at image center).
    """
    fx = width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    fy = fx  # assume square pixels
    cx = width / 2.0
    cy = height / 2.0
    K = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return K
