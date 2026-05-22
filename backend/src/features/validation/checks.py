"""Catalog of mesh sanity checks.

Each check is a pure function `(mesh: trimesh.Trimesh) -> CheckResult` that
the validate route runs through and aggregates into a report.

To add a new check, write the function and register it at the bottom of the
file. The route picks the catalog up automatically.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass
class CheckResult:
    name: str
    status: str  # 'pass' | 'warn' | 'fail'
    message: str
    fix: str


# ---------------------------------------------------------- watertight

def check_watertight(mesh: trimesh.Trimesh) -> CheckResult:
    """Watertightness for a multi-part scene is measured per component. A
    scene of N closed boxes is "watertight enough" even though the union
    isn't a single closed surface."""
    parts = mesh.split(only_watertight=False)
    if not parts:
        return CheckResult("watertight", "fail", "empty mesh", "no faces to analyze")
    closed = sum(1 for p in parts if p.is_watertight)
    ratio = closed / len(parts)
    if ratio >= 0.95:
        return CheckResult(
            "watertight",
            "pass",
            f"{closed}/{len(parts)} components are closed",
            "",
        )
    if ratio >= 0.5:
        return CheckResult(
            "watertight",
            "warn",
            f"only {closed}/{len(parts)} components closed",
            "Half-open meshes still work for navigation but the agent can wander past open faces.",
        )
    return CheckResult(
        "watertight",
        "fail",
        f"{closed}/{len(parts)} components closed — mesh has holes",
        "Single-photo reconstructions are 2.5D and open at the back. Use multi-frame video or LiDAR.",
    )


# ---------------------------------------------------------- connected components

def check_connected_components(mesh: trimesh.Trimesh) -> CheckResult:
    """Rooms legitimately have many parts (walls + floor + each obstacle is
    its own watertight box). The signal we want: is there ONE dominant
    component, or is the mesh shattered into uniform small pieces?
    """
    parts = mesh.split(only_watertight=False)
    n = len(parts)
    if n <= 1:
        return CheckResult("connected_components", "pass", "single connected mesh", "")

    sizes = sorted((len(p.faces) for p in parts), reverse=True)
    largest_share = sizes[0] / max(sum(sizes), 1)

    if largest_share >= 0.5 or n <= 20:
        return CheckResult(
            "connected_components",
            "pass",
            f"{n} pieces; largest covers {largest_share:.0%}",
            "",
        )
    if n <= 50:
        return CheckResult(
            "connected_components",
            "warn",
            f"{n} pieces; no dominant component",
            "Small islands are usually scan noise. Raise the reconstruction confidence threshold.",
        )
    return CheckResult(
        "connected_components",
        "fail",
        f"{n} pieces — mesh is fragmented",
        "Reconstruction failed to produce a coherent scene. Re-record with more frames / better coverage.",
    )


# ---------------------------------------------------------- bbox plausibility

def check_bbox_plausibility(mesh: trimesh.Trimesh) -> CheckResult:
    extents = mesh.extents  # (x, y, z) lengths
    longest = float(np.max(extents))
    if 1.0 <= longest <= 30.0:
        return CheckResult(
            "bbox_plausibility",
            "pass",
            f"longest dim {longest:.2f} m — plausible room size",
            "",
        )
    if 0.5 <= longest <= 100.0:
        return CheckResult(
            "bbox_plausibility",
            "warn",
            f"longest dim {longest:.2f} m — unusual for a room",
            "If the scan is unitless, set --target-diagonal-m in Build settings.",
        )
    return CheckResult(
        "bbox_plausibility",
        "fail",
        f"longest dim {longest:.2f} m — implausible",
        "Reconstruction scale is wrong. Set --target-diagonal-m or re-run with known intrinsics.",
    )


# ---------------------------------------------------------- floor detected

def check_floor_detected(mesh: trimesh.Trimesh) -> CheckResult:
    """Find faces with normal close to +Y (Y-up convention from photo depth)
    or +Z (Z-up convention from LiDAR), in the lowest 10% of mesh height."""
    if len(mesh.faces) == 0:
        return CheckResult("floor_detected", "fail", "empty mesh", "no faces to analyze")

    normals = mesh.face_normals
    centroids = mesh.triangles_center
    bbox = mesh.bounds
    height_axis = int(np.argmin(bbox[0]))  # not perfect but reasonable
    # try both Y-up and Z-up
    best_area = 0.0
    for axis in (1, 2):
        up_dot = normals[:, axis]
        is_up = up_dot > 0.9
        if not is_up.any():
            continue
        face_heights = centroids[:, axis]
        h_min = float(face_heights.min())
        h_max = float(face_heights.max())
        near_floor = face_heights < (h_min + (h_max - h_min) * 0.15)
        floor_mask = is_up & near_floor
        if floor_mask.any():
            areas = mesh.area_faces[floor_mask]
            best_area = max(best_area, float(np.sum(areas)))

    if best_area >= 1.0:
        return CheckResult(
            "floor_detected",
            "pass",
            f"floor patch ~{best_area:.2f} m²",
            "",
        )
    if best_area >= 0.25:
        return CheckResult(
            "floor_detected",
            "warn",
            f"small floor patch ({best_area:.2f} m²)",
            "Agent spawn region may be limited. Capture more of the floor.",
        )
    return CheckResult(
        "floor_detected",
        "fail",
        "no flat ground plane found",
        "Without a floor, the agent has nowhere to spawn. Record with the camera tilted slightly down.",
    )


# ---------------------------------------------------------- convex decomp quality

def check_convex_decomp_quality(mesh: trimesh.Trimesh) -> CheckResult:
    """Decompose into convex components (per-connected-component hulls) and
    check volume preservation. VHACD would be more accurate; we use the
    simpler fallback that rl_env.build also uses by default."""
    parts = mesh.split(only_watertight=False)
    n_hulls = len(parts)
    if n_hulls == 0:
        return CheckResult("convex_decomp_quality", "fail", "no hulls produced", "empty mesh")

    try:
        original_vol = abs(float(mesh.volume))
    except Exception:
        original_vol = 0.0
    hull_vol = 0.0
    for p in parts:
        try:
            hull_vol += abs(float(p.convex_hull.volume))
        except Exception:
            pass

    vol_ratio = hull_vol / max(original_vol, 1e-6) if original_vol > 1e-3 else 1.0

    if 3 <= n_hulls <= 64 and vol_ratio >= 0.7:
        return CheckResult(
            "convex_decomp_quality",
            "pass",
            f"{n_hulls} hulls, vol preservation {vol_ratio:.0%}",
            "",
        )
    if n_hulls < 3:
        return CheckResult(
            "convex_decomp_quality",
            "warn" if n_hulls >= 2 else "fail",
            f"only {n_hulls} hulls — scene collapsed into a blob",
            "Install vhacdx (pip install vhacdx) for proper convex decomposition, "
            "or capture more obstacles in the scene.",
        )
    if vol_ratio < 0.5:
        return CheckResult(
            "convex_decomp_quality",
            "fail",
            f"vol preservation only {vol_ratio:.0%} — hulls overshoot scene",
            "Mesh has deep concavities; reduce max_hulls or use VHACD.",
        )
    return CheckResult(
        "convex_decomp_quality",
        "warn",
        f"{n_hulls} hulls, vol preservation {vol_ratio:.0%}",
        "Acceptable but not ideal.",
    )


# ---------------------------------------------------------- scale calibration

def check_scale_calibration(mesh: trimesh.Trimesh) -> CheckResult:
    extents = mesh.extents
    if (extents > 0).all():
        ratios = extents / extents.min()
        if ratios.max() < 20:
            return CheckResult(
                "scale_calibration",
                "pass",
                f"aspect ratios reasonable (max {ratios.max():.1f}×)",
                "",
            )
        return CheckResult(
            "scale_calibration",
            "warn",
            f"extreme aspect ratio {ratios.max():.1f}×",
            "One dimension is much larger than the others — likely a degenerate scan.",
        )
    return CheckResult(
        "scale_calibration",
        "fail",
        "degenerate (zero-extent) dimension",
        "Mesh is flat in at least one axis.",
    )


# ---------------------------------------------------------- registry

CATALOG: dict[str, Callable[[trimesh.Trimesh], CheckResult]] = {
    "watertight": check_watertight,
    "connected_components": check_connected_components,
    "bbox_plausibility": check_bbox_plausibility,
    "floor_detected": check_floor_detected,
    "convex_decomp_quality": check_convex_decomp_quality,
    "scale_calibration": check_scale_calibration,
}


def run_all(mesh_path: str | Path) -> dict:
    """Load a mesh, run every check, return the report dict that gets
    persisted to the validation table."""
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        return {
            "checks": [],
            "overall": "fail",
            "error": f"could not load mesh as trimesh.Trimesh (got {type(mesh).__name__})",
        }
    results = [check(mesh) for check in CATALOG.values()]
    statuses = {r.status for r in results}
    overall = "fail" if "fail" in statuses else ("warn" if "warn" in statuses else "pass")
    return {
        "checks": [r.__dict__ for r in results],
        "overall": overall,
    }
