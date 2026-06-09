"""Unit tests for depth-fusion geometry helpers. Pure math, no GPU/models."""
from __future__ import annotations

import numpy as np
import pytest

from src.features.reconstruction.backends._geometry import (
    assume_intrinsics,
    back_project,
    ransac_umeyama,
    robust_linear_fit,
    umeyama_rigid,
)


def test_robust_linear_fit_recovers_scale() -> None:
    # Scale-only contract (matches prototype v4.2): recover k, offset fixed at 0.
    rng = np.random.default_rng(0)
    z_new = rng.uniform(0.5, 5.0, size=50)
    k_true = 1.3
    z_ref = k_true * z_new + rng.normal(0, 0.01, size=50)
    k, c = robust_linear_fit(z_ref, z_new)
    assert abs(k - k_true) < 0.05
    assert c == 0.0


def test_robust_linear_fit_survives_outliers() -> None:
    rng = np.random.default_rng(1)
    z_new = rng.uniform(0.5, 5.0, size=50)
    z_ref = 1.1 * z_new
    # Corrupt 30% with wild outliers — the median ratio shrugs them off.
    z_ref[::3] = rng.uniform(-10, 50, size=z_ref[::3].shape)
    k, c = robust_linear_fit(z_ref, z_new)
    assert 0.9 < k < 1.3
    assert c == 0.0


def test_robust_linear_fit_clamps_scale() -> None:
    # Extreme scale is clamped to [0.5, 2.0] to guard ill-conditioned matches.
    z_new = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    k, c = robust_linear_fit(10.0 * z_new, z_new)
    assert k == 2.0 and c == 0.0


def test_robust_linear_fit_falls_back_when_too_few_points() -> None:
    k, c = robust_linear_fit(np.array([1.0, 2.0]), np.array([0.5, 1.0]))
    assert k == 1.0 and c == 0.0


def test_umeyama_rigid_recovers_known_transform() -> None:
    rng = np.random.default_rng(2)
    pts = rng.normal(size=(20, 3))
    theta = 0.5
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    t = np.array([0.3, -0.7, 1.1])
    pts_t = (R @ pts.T).T + t

    T = umeyama_rigid(pts, pts_t)
    R_est, t_est = T[:3, :3], T[:3, 3]
    np.testing.assert_allclose(R_est, R, atol=1e-6)
    np.testing.assert_allclose(t_est, t, atol=1e-6)
    # Apply and check it round-trips.
    pts_back = (R_est @ pts.T).T + t_est
    np.testing.assert_allclose(pts_back, pts_t, atol=1e-6)


def test_umeyama_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError):
        umeyama_rigid(np.zeros((2, 3)), np.zeros((2, 3)))


def test_ransac_umeyama_picks_inliers_around_outliers() -> None:
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(60, 3))
    theta = 0.4
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    t = np.array([0.2, 0.5, -0.3])
    pts_t = (R @ pts.T).T + t
    # Corrupt 40% with random offsets.
    bad = slice(0, 24)
    pts_t[bad] = pts_t[bad] + rng.uniform(-5.0, 5.0, size=(24, 3))

    T, inliers = ransac_umeyama(pts, pts_t, iters=300, inlier_thresh_m=0.1, min_inliers=10, rng=rng)
    assert T is not None
    # Inliers should be predominantly from the clean half (idx >= 24).
    inlier_idxs = np.where(inliers)[0]
    assert inliers.sum() >= 20
    assert (inlier_idxs >= 24).mean() > 0.9


def test_ransac_returns_none_when_no_consensus() -> None:
    rng = np.random.default_rng(4)
    pts = rng.normal(size=(20, 3))
    noise = rng.uniform(-100, 100, size=(20, 3))
    T, inliers = ransac_umeyama(pts, noise, iters=100, inlier_thresh_m=0.05, min_inliers=10, rng=rng)
    assert T is None
    assert inliers.sum() == 0


def test_back_project_produces_mesh_for_simple_plane() -> None:
    # 4x4 flat depth grid at z=1m.
    depth = np.ones((4, 4), dtype=np.float32)
    color = (np.ones((4, 4, 3)) * 255).astype(np.uint8)
    K = assume_intrinsics(width=4, height=4, fov_deg=60.0)
    verts, faces, cols = back_project(depth, color, K, world_transform=None, discontinuity_m=0.1)
    assert verts.shape == (16, 3)
    assert cols.shape == (16, 3)
    # 3 cells x 3 cells x 2 triangles = 18 faces.
    assert faces.shape == (18, 3)
    # All z values are 1.
    np.testing.assert_allclose(verts[:, 2], 1.0, atol=1e-6)


def test_back_project_skips_discontinuities() -> None:
    depth = np.ones((4, 4), dtype=np.float32)
    depth[2:, :] = 5.0  # half the grid jumps 4 m away
    color = (np.ones((4, 4, 3)) * 128).astype(np.uint8)
    K = assume_intrinsics(width=4, height=4, fov_deg=60.0)
    verts, faces, _ = back_project(depth, color, K, world_transform=None, discontinuity_m=0.5)
    # Cells crossing the row-2 boundary get culled; only top and bottom 3-cell rows survive,
    # so faces = (1 + 1) rows × 3 cols × 2 triangles = 12.
    assert faces.shape[0] == 12


def test_back_project_applies_world_transform() -> None:
    depth = np.ones((2, 2), dtype=np.float32)
    color = np.zeros((2, 2, 3), dtype=np.uint8)
    K = assume_intrinsics(2, 2, fov_deg=60.0)
    T = np.eye(4)
    T[:3, 3] = [10.0, 20.0, 30.0]
    verts, _, _ = back_project(depth, color, K, world_transform=T)
    # Every vertex's z = 1 + 30 = 31.
    np.testing.assert_allclose(verts[:, 2], 31.0, atol=1e-5)
