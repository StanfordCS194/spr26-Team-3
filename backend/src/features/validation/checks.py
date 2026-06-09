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


# ---------------------------------------------------------- component analysis
#
# A back-projected reconstruction is often millions of faces split into
# thousands of tiny disconnected "noise islands". `mesh.split()` materialises a
# full Trimesh *per component*, which exhausts memory and hangs/kills the
# process. Instead we label components cheaply with face-adjacency (scipy) and
# only materialise the largest few to run the per-component watertight / hull
# tests. The three split-based checks read this precomputed summary.

_MAX_SUBMESHES = 48  # cap how many of the largest components we copy out


@dataclass
class _Components:
    count: int            # total number of connected components
    face_sizes: list[int]  # face count per component (all of them — cheap)
    sampled: int          # how many of the largest were materialised
    closed: int           # of those sampled, how many are watertight
    hull_vol: float       # summed convex-hull volume of the sampled components
    mesh_vol: float       # |mesh.volume|


def _analyze_components(mesh: trimesh.Trimesh) -> _Components:
    """Connected-component stats WITHOUT mesh.split() (which copies a full
    Trimesh per component → OOM on a multi-million-face reconstruction).
    Best-effort: never raises."""
    nf = len(mesh.faces)
    if nf == 0:
        return _Components(0, [], 0, 0, 0.0, 0.0)
    try:
        from trimesh.graph import connected_components

        comps = connected_components(
            mesh.face_adjacency, min_len=1, nodes=np.arange(nf)
        )
    except Exception:
        # Fall back to treating the whole mesh as a single component.
        comps = [np.arange(nf)]

    face_sizes = [int(len(c)) for c in comps]
    order = np.argsort(face_sizes)[::-1][:_MAX_SUBMESHES]  # largest first
    closed = 0
    hull_vol = 0.0
    for i in order:
        try:
            sub = mesh.submesh([comps[i]], append=True, repair=False)
            if sub.is_watertight:
                closed += 1
            hull_vol += abs(float(sub.convex_hull.volume))
        except Exception:
            pass
    try:
        mesh_vol = abs(float(mesh.volume))
    except Exception:
        mesh_vol = 0.0
    return _Components(len(comps), face_sizes, int(len(order)), closed, hull_vol, mesh_vol)


def _components(mesh: trimesh.Trimesh) -> _Components:
    cached = mesh.metadata.get("_wv_comps") if hasattr(mesh, "metadata") else None
    return cached if isinstance(cached, _Components) else _analyze_components(mesh)


# ---------------------------------------------------------- watertight

def check_watertight(mesh: trimesh.Trimesh) -> CheckResult:
    """Watertightness for a multi-part scene is measured per component. A
    scene of N closed boxes is "watertight enough" even though the union
    isn't a single closed surface."""
    comps = _components(mesh)
    if comps.count == 0:
        return CheckResult("watertight", "fail", "empty mesh", "no faces to analyze")
    denom = max(comps.sampled, 1)  # measured over the largest components
    closed = comps.closed
    ratio = closed / denom
    if ratio >= 0.95:
        return CheckResult(
            "watertight",
            "pass",
            f"{closed}/{denom} largest components are closed",
            "",
        )
    if ratio >= 0.5:
        return CheckResult(
            "watertight",
            "warn",
            f"only {closed}/{denom} largest components closed",
            "Half-open meshes still work for navigation but the agent can wander past open faces.",
        )
    return CheckResult(
        "watertight",
        "warn",
        f"{closed}/{denom} largest components closed — open/partial scan",
        "Partial captures (e.g. half a room) are open at the back. Build wraps "
        "the scene in invisible boundary walls so the agent stays inside; "
        "multi-frame video or LiDAR gives a more complete mesh.",
    )


# ---------------------------------------------------------- connected components

def check_connected_components(mesh: trimesh.Trimesh) -> CheckResult:
    """Rooms legitimately have many parts (walls + floor + each obstacle is
    its own watertight box). The signal we want: is there ONE dominant
    component, or is the mesh shattered into uniform small pieces?
    """
    comps = _components(mesh)
    n = comps.count
    if n <= 1:
        return CheckResult("connected_components", "pass", "single connected mesh", "")

    sizes = sorted(comps.face_sizes, reverse=True)
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
        "warn",
        "little/no floor captured in the scan",
        "Build adds a ground plane under the scene, so the agent can still "
        "spawn — but capturing more floor (camera tilted slightly down) improves "
        "coverage and the spawn area.",
    )


# ---------------------------------------------------------- convex decomp quality

def check_convex_decomp_quality(mesh: trimesh.Trimesh) -> CheckResult:
    """Decompose into convex components (per-connected-component hulls) and
    check volume preservation. VHACD would be more accurate; we use the
    simpler fallback that rl_env.build also uses by default."""
    comps = _components(mesh)
    n_hulls = comps.count
    if n_hulls == 0:
        return CheckResult("convex_decomp_quality", "fail", "no hulls produced", "empty mesh")

    # hull_vol is summed over the largest sampled components; for open
    # reconstructions mesh.volume is ~0 so vol_ratio defaults to 1.0 and the
    # hull-count thresholds drive the verdict.
    original_vol = comps.mesh_vol
    hull_vol = comps.hull_vol
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
    # Analyse connected components ONCE (memory-bounded) and cache it so the
    # split-based checks don't each re-walk a multi-million-face mesh.
    try:
        mesh.metadata["_wv_comps"] = _analyze_components(mesh)
    except Exception:
        pass
    results = [check(mesh) for check in CATALOG.values()]
    statuses = {r.status for r in results}
    overall = "fail" if "fail" in statuses else ("warn" if "warn" in statuses else "pass")
    return {
        "checks": [r.__dict__ for r in results],
        "overall": overall,
    }
