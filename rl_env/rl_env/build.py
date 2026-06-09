"""Tier 1 pipeline: mesh -> MuJoCo MJCF navigation environment.

Mirrors PRD Feature 2 (Tier 1, weeks 2-8): preprocess -> whole-scene convex
decomposition -> material estimation -> MJCF export. The whole scene is one
static collision body; per-object segmentation is Tier 2 and out of scope here.

Usage:
    cfg = BuildConfig(mesh_path="scan.obj", out_dir="build/")
    artifacts = build_environment(cfg)
    # artifacts.mjcf_path -> ready to load with mujoco.MjModel.from_xml_path
"""
from __future__ import annotations

import json
import shutil
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"trimesh\.triangles")


@dataclass
class BuildConfig:
    mesh_path: str | Path
    out_dir: str | Path
    target_diagonal_m: float | None = 6.0
    """If set, scale the input mesh so its xy-diagonal matches this length (meters).

    The prototype's photo-to-3D output is unitless. Real Polycam/iPhone LiDAR
    scans are metric; pass None to skip rescaling for those.
    """
    up_axis: str = "auto"
    """'y', 'z', or 'auto'. Photo-depth meshes are typically Y-up; LiDAR is Z-up."""
    max_hulls: int = 64
    """Cap on convex hulls when decomposing. More hulls = more accurate collisions, slower sim."""
    decompose: bool = True
    """If False, use connected-component hulls (fast); skips per-shape decomposition."""
    enclose: bool = True
    """Wrap the scene in invisible boundary walls so the agent can't wander out
    of partial/open reconstructions (a half-recorded room has open sides). The
    walls sit at the mesh's XY bounds + `wall_margin` and are collidable +
    lidar-visible but not rendered."""
    wall_margin: float = 0.25
    """Gap (meters) between the reconstructed geometry and the boundary walls."""
    wall_thickness: float = 0.08
    """Half-thickness (meters) of the invisible boundary walls."""


@dataclass
class BuildArtifacts:
    mjcf_path: Path
    mesh_dir: Path
    n_hulls: int
    bounds: np.ndarray  # (2, 3) AABB in MJCF coords (Z-up)
    floor_z: float
    spawn_region: tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
    materials: dict[str, dict] = field(default_factory=dict)
    # 4x4 raw-mesh -> sim/MJCF transform (rotate up-axis, center+ground, scale).
    # Lets viewers map sim-frame trajectories back onto the raw textured mesh.
    raw_to_sim: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------------


def _detect_up_axis(mesh: trimesh.Trimesh) -> str:
    """Find the vertical axis of a room-like mesh.

    A real floor (and usually a ceiling) is a dense, flat plane sitting at one
    extreme of the vertical axis; walls are not. So the vertical axis is the one
    whose extreme planes hold the largest fraction of vertices. This is far more
    reliable than comparing bounding-box extents (a room is often wider/longer
    than it is tall, which fooled the old extent heuristic into picking a
    horizontal axis as "up").

    Falls back to the extent heuristic when the floor signal is weak/ambiguous
    (e.g. a closed synthetic box with uniform vertex density).
    """
    v = np.asarray(mesh.vertices)

    def floor_conc(ax: int) -> float:
        lo, hi = float(v[:, ax].min()), float(v[:, ax].max())
        span = hi - lo
        if span < 1e-6:
            return 0.0
        near_lo = float(np.mean(v[:, ax] < lo + 0.08 * span))
        near_hi = float(np.mean(v[:, ax] > hi - 0.08 * span))
        return max(near_lo, near_hi)

    cy, cz = floor_conc(1), floor_conc(2)
    # One axis clearly dominating the other is the signal — partial scans may
    # only put ~15-20% of vertices in the floor slab, so judge by ratio, not by
    # an absolute cutoff.
    if max(cy, cz) >= 0.10 and max(cy, cz) > 1.5 * min(cy, cz):
        return "y" if cy > cz else "z"

    extents = mesh.extents
    if extents[2] > extents[1] * 1.2 and extents[2] > extents[0] * 0.5:
        return "z"
    return "y"


def _to_z_up(mesh: trimesh.Trimesh, up_axis: str) -> trimesh.Trimesh:
    if up_axis == "z":
        return mesh
    if up_axis == "y":
        R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        mesh = mesh.copy()
        mesh.apply_transform(R)
        return mesh
    raise ValueError(f"unknown up axis: {up_axis}")


def preprocess(mesh: trimesh.Trimesh, cfg: BuildConfig) -> tuple[trimesh.Trimesh, float, np.ndarray]:
    """Return (mesh in MuJoCo Z-up coords, floor_z, raw_to_sim 4x4).

    raw_to_sim maps a point in the original mesh's coordinates to the
    sim/MJCF frame, composed as scale @ translate @ rotate.
    """
    up = cfg.up_axis if cfg.up_axis != "auto" else _detect_up_axis(mesh)
    if up == "y":
        R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    elif up == "z":
        R = np.eye(4)
    else:
        raise ValueError(f"unknown up axis: {up}")
    mesh = mesh.copy()
    mesh.apply_transform(R)

    mins = mesh.bounds[0]
    maxs = mesh.bounds[1]
    center_xy = (mins[:2] + maxs[:2]) / 2
    tvec = np.array([-center_xy[0], -center_xy[1], -mins[2]], dtype=float)
    Tm = trimesh.transformations.translation_matrix(tvec)
    mesh.apply_translation(tvec)

    Sm = np.eye(4)
    if cfg.target_diagonal_m is not None:
        ext = mesh.extents
        diag_xy = float(np.hypot(ext[0], ext[1]))
        if diag_xy > 1e-6:
            scale = cfg.target_diagonal_m / diag_xy
            Sm = trimesh.transformations.scale_matrix(scale)
            mesh.apply_scale(scale)

    floor_z = float(mesh.bounds[0][2])
    raw_to_sim = Sm @ Tm @ R  # apply R first, then T, then S
    return mesh, floor_z, raw_to_sim


# ---------------------------------------------------------------------------
# Convex decomposition
# ---------------------------------------------------------------------------


# A reconstruction mesh can be millions of faces; coarse collision geometry
# doesn't need that, and the heavy ops below choke on it. Decimate first.
_COLLISION_MAX_FACES = 120_000


def _decimate_for_collision(mesh: trimesh.Trimesh, max_faces: int = _COLLISION_MAX_FACES) -> trimesh.Trimesh:
    """Best-effort reduce face count before decomposition. If no decimation
    backend is available, return the mesh unchanged — the component fallback
    below is memory-safe either way."""
    if len(mesh.faces) <= max_faces:
        return mesh
    try:
        out = mesh.simplify_quadric_decimation(face_count=max_faces)
        if out is not None and len(out.faces) > 0:
            return out
    except Exception:
        pass
    return mesh


def _largest_components(mesh: trimesh.Trimesh, k: int) -> list[trimesh.Trimesh]:
    """Up to `k` largest connected components, materialised one at a time.

    Avoids `mesh.split()` — which builds a full Trimesh for *every* component,
    OOMing on a back-projected mesh with thousands of tiny noise islands. We
    label components cheaply (face adjacency) and only copy out the biggest few.
    """
    nf = len(mesh.faces)
    if nf == 0:
        return [mesh]
    try:
        from trimesh.graph import connected_components

        comps = connected_components(mesh.face_adjacency, min_len=1, nodes=np.arange(nf))
    except Exception:
        return [mesh]
    comps = sorted(comps, key=len, reverse=True)[: max(k, 1)]
    out: list[trimesh.Trimesh] = []
    for fi in comps:
        try:
            out.append(mesh.submesh([fi], append=True, repair=False))
        except Exception:
            pass
    return out or [mesh]


def _real_floor_z(mesh: trimesh.Trimesh) -> float:
    """The visible floor height: the densest z-slab in the lower 40% of the
    height range. Phone scans have noise *below* the real floor, so the mesh
    minimum sits under the surface the robot should roll on."""
    zs = np.asarray(mesh.vertices)[:, 2]
    hist, edges = np.histogram(zs, bins=80)
    i = int(np.argmax(hist[: max(1, int(80 * 0.4))]))
    return float((edges[i] + edges[i + 1]) / 2)


def _occupancy_columns(
    mesh: trimesh.Trimesh,
    floor_z: float,
    cell: float = 0.17,
    band_lo: float = 0.14,
    band_hi: float = 1.4,
    max_boxes: int = 700,
) -> list[tuple[float, float, float, float, float, float]]:
    """2.5D costmap collision: one box column per XY cell that contains mesh
    points in the obstacle band above the floor (walls, furniture).

    Convex hulls of a curved scan shell each wrap huge swaths of *free* space —
    the agent ends up "in collision" everywhere and lidar sees phantom walls.
    Box columns hug the actual geometry instead. Returns (cx, cy, hx, hy, z0, z1).
    """
    v = np.asarray(mesh.vertices)
    zs = v[:, 2]
    pts = v[(zs > floor_z + band_lo) & (zs < floor_z + band_hi)]
    if not len(pts):
        return []
    while True:
        origin = pts[:, :2].min(axis=0)
        ij = np.floor((pts[:, :2] - origin) / cell).astype(int)
        from collections import defaultdict

        zmax: dict[tuple[int, int], float] = defaultdict(float)
        count: dict[tuple[int, int], int] = defaultdict(int)
        for (i, j), z in zip(map(tuple, ij), pts[:, 2]):
            count[(i, j)] += 1
            zmax[(i, j)] = max(zmax[(i, j)], z - floor_z)
        thr = max(4, int(np.percentile(np.array(list(count.values())), 30) * 0.4))
        cells = [c for c, n in count.items() if n >= thr]
        if len(cells) <= max_boxes:
            break
        cell *= 1.4  # too fine for this scan — coarsen and retry
    half = cell * 0.55  # slight overlap so diagonal gaps don't leak
    out = []
    for c in cells:
        h = max(0.12, float(zmax[c]) / 2)
        out.append((
            float(origin[0] + (c[0] + 0.5) * cell),
            float(origin[1] + (c[1] + 0.5) * cell),
            half, half,
            floor_z, floor_z + 2 * h,
        ))
    return out


def decompose(mesh: trimesh.Trimesh, cfg: BuildConfig) -> list[trimesh.Trimesh]:
    """Return a list of convex hull meshes covering `mesh`.

    Strategy:
      1. Decimate huge meshes (coarse collision geometry needs few faces).
      2. If trimesh has a working VHACD/CoACD backend, use it.
      3. Else take convex hulls of the largest connected components.
      4. If that yields nothing, a single convex hull of the whole scene.

    The MVP intentionally accepts coarse collision geometry; tightening this
    is a known follow-up (PRD: "balance collision fidelity against simulation
    speed", Key technical challenge #1).
    """
    work = _decimate_for_collision(mesh)

    # Only attempt VHACD if the mesh is a sane size — it's slow/heavy on a
    # multi-million-face mesh, and the component fallback is always safe.
    if cfg.decompose and len(work.faces) <= _COLLISION_MAX_FACES * 4:
        try:
            hulls = trimesh.decomposition.convex_decomposition(
                work, maxNumVerticesPerCH=64, resolution=100_000
            )
            if hulls:
                hulls = [h for h in hulls if h.is_volume and h.volume > 1e-6]
                if hulls:
                    return hulls[: cfg.max_hulls]
        except Exception:
            pass

    hulls: list[trimesh.Trimesh] = []
    for c in _largest_components(work, cfg.max_hulls):
        try:
            h = c.convex_hull
            if h.is_volume and h.volume > 1e-6:
                hulls.append(h)
        except Exception:
            continue

    if not hulls:
        try:
            hulls = [work.convex_hull]
        except Exception:
            hulls = [mesh.convex_hull]

    return hulls


# ---------------------------------------------------------------------------
# Material lookup (CLIP stub)
# ---------------------------------------------------------------------------


# Simple lookup table — class -> (friction triplet, rgba). PRD calls for CLIP
# classification feeding this table; for the MVP we infer class from geometric
# heuristics (low geom -> floor, otherwise wall/object). The CLIP hook is
# `classify_hull` below.
MATERIAL_TABLE: dict[str, dict] = {
    "floor": {"friction": (1.5, 0.01, 0.0001), "rgba": (0.45, 0.45, 0.5, 1.0)},
    "wall": {"friction": (1.0, 0.005, 0.0001), "rgba": (0.78, 0.78, 0.78, 1.0)},
    "object": {"friction": (0.9, 0.005, 0.0001), "rgba": (0.55, 0.65, 0.85, 1.0)},
}


def classify_hull(hull: trimesh.Trimesh, scene_bounds: np.ndarray) -> str:
    """Heuristic class: lowest-thin slabs -> floor; tall/wide -> wall; else object.

    Replaceable with a CLIP-based classifier once we render per-hull crops.
    """
    z_min, z_max = scene_bounds[0][2], scene_bounds[1][2]
    scene_h = max(z_max - z_min, 1e-6)

    h = hull.extents[2]
    z_centroid = float(hull.centroid[2])
    rel_z = (z_centroid - z_min) / scene_h

    if rel_z < 0.08 and h < 0.3:
        return "floor"
    if h / scene_h > 0.5:
        return "wall"
    return "object"


# ---------------------------------------------------------------------------
# MJCF generation
# ---------------------------------------------------------------------------


def _xml_indent(elem: ET.Element, level: int = 0) -> None:
    pad = "\n" + "  " * level
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = pad + "  "
        if not (elem.tail or "").strip():
            elem.tail = pad
        for child in elem:
            _xml_indent(child, level + 1)
        if not (child.tail or "").strip():
            child.tail = pad
    else:
        if level and not (elem.tail or "").strip():
            elem.tail = pad


def write_mjcf(
    hulls: list[trimesh.Trimesh],
    classes: list[str],
    out_dir: Path,
    floor_z: float,
    spawn_region: tuple[float, float, float, float],
    bounds: np.ndarray | None = None,
    enclose: bool = True,
    wall_margin: float = 0.25,
    wall_thickness: float = 0.08,
    boxes: list[tuple[float, float, float, float, float, float]] | None = None,
) -> Path:
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    mesh_files: list[str] = []
    for i, hull in enumerate(hulls):
        fname = f"hull_{i:04d}.stl"
        hull.export(mesh_dir / fname)
        mesh_files.append(fname)

    mujoco = ET.Element("mujoco", model="worldscan_nav")
    ET.SubElement(mujoco, "option", timestep="0.01", integrator="implicitfast")
    ET.SubElement(
        mujoco,
        "compiler",
        angle="radian",
        autolimits="true",
        meshdir="meshes",
    )

    default = ET.SubElement(mujoco, "default")
    d_scene = ET.SubElement(default, "default", attrib={"class": "scene"})
    ET.SubElement(
        d_scene,
        "geom",
        group="1",
        type="mesh",
        contype="1",
        conaffinity="1",
        condim="3",
        rgba="0.78 0.78 0.78 1",
        friction="1.0 0.005 0.0001",
    )
    d_floor = ET.SubElement(default, "default", attrib={"class": "nav_floor"})
    ET.SubElement(
        d_floor,
        "geom",
        type="plane",
        rgba="0.25 0.25 0.28 1",
        friction="1.5 0.01 0.0001",
    )
    d_agent = ET.SubElement(default, "default", attrib={"class": "agent"})
    ET.SubElement(
        d_agent,
        "geom",
        type="sphere",
        size="0.15",
        rgba="0.4 0.9 0.6 1",
        mass="1.0",
        friction="0.8 0.005 0.0001",
    )

    asset = ET.SubElement(mujoco, "asset")
    ET.SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1="0.3 0.5 0.7",
        rgb2="0 0 0",
        width="32",
        height="512",
    )
    for fname in mesh_files:
        # refpos/refquat at identity stops MuJoCo from re-centering and
        # rotating mesh vertices to its principal-inertia frame; otherwise
        # asymmetric hulls (walls!) get reoriented and leave gaps.
        ET.SubElement(
            asset,
            "mesh",
            name=Path(fname).stem,
            file=fname,
            refpos="0 0 0",
            refquat="1 0 0 0",
        )

    worldbody = ET.SubElement(mujoco, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        pos="0 0 5",
        dir="0 0 -1",
        diffuse="0.8 0.8 0.8",
    )
    ET.SubElement(
        worldbody,
        "geom",
        attrib={"class": "nav_floor"},
        name="floor",
        size="20 20 0.05",
        pos=f"0 0 {floor_z:.4f}",
    )

    scene_body = ET.SubElement(worldbody, "body", name="scene", pos="0 0 0")
    for fname, cls in zip(mesh_files, classes):
        mat = MATERIAL_TABLE[cls]
        ET.SubElement(
            scene_body,
            "geom",
            attrib={"class": "scene"},
            name=Path(fname).stem,
            mesh=Path(fname).stem,
            rgba=" ".join(f"{v:.3f}" for v in mat["rgba"]),
            friction=" ".join(f"{v:.4f}" for v in mat["friction"]),
        )
    # Occupancy-column collision (scan meshes). Named hull_box_* so NavEnv
    # counts touches as scene collisions and lidar sees them (env matches the
    # `hull_` prefix).
    for i, (cx, cy, hx, hy, z0, z1) in enumerate(boxes or []):
        ET.SubElement(
            scene_body,
            "geom",
            name=f"hull_box_{i:04d}",
            type="box",
            size=f"{hx:.4f} {hy:.4f} {(z1 - z0) / 2:.4f}",
            pos=f"{cx:.4f} {cy:.4f} {(z0 + z1) / 2:.4f}",
            contype="1",
            conaffinity="1",
            condim="3",
            group="1",
            rgba="0.78 0.78 0.78 1",
            friction="1.0 0.005 0.0001",
        )

    # Invisible boundary walls — enclose the navigable footprint so the agent
    # can't escape a partial/open reconstruction. Sit at the mesh XY bounds +
    # margin, floor-to-ceiling. Collidable + lidar-visible, but rgba alpha 0 so
    # they don't render. Named `boundary_*` so NavEnv counts touches as
    # collisions (see env._collect_scene_geoms).
    if enclose and bounds is not None:
        bx0, by0, bz0 = float(bounds[0][0]), float(bounds[0][1]), float(bounds[0][2])
        bx1, by1, bz1 = float(bounds[1][0]), float(bounds[1][1]), float(bounds[1][2])
        x0, x1 = bx0 - wall_margin, bx1 + wall_margin
        y0, y1 = by0 - wall_margin, by1 + wall_margin
        wall_h = max(bz1 - bz0, 1.0)  # at least 1 m so the agent can't roll over
        zc = floor_z + wall_h / 2.0
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        hx, hy = (x1 - x0) / 2.0, (y1 - y0) / 2.0
        t = wall_thickness
        bf = "1.0 0.005 0.0001"
        boundary = ET.SubElement(worldbody, "body", name="boundary", pos="0 0 0")
        walls = [
            ("boundary_xmin", f"{t} {hy + t} {wall_h / 2:.4f}", f"{x0:.4f} {cy:.4f} {zc:.4f}"),
            ("boundary_xmax", f"{t} {hy + t} {wall_h / 2:.4f}", f"{x1:.4f} {cy:.4f} {zc:.4f}"),
            ("boundary_ymin", f"{hx + t} {t} {wall_h / 2:.4f}", f"{cx:.4f} {y0:.4f} {zc:.4f}"),
            ("boundary_ymax", f"{hx + t} {t} {wall_h / 2:.4f}", f"{cx:.4f} {y1:.4f} {zc:.4f}"),
        ]
        for name, size, pos in walls:
            ET.SubElement(
                boundary,
                "geom",
                name=name,
                type="box",
                size=size,
                pos=pos,
                contype="1",
                conaffinity="1",
                condim="3",
                group="3",
                rgba="0 0 0 0",
                friction=bf,
            )

    xmin, xmax, ymin, ymax = spawn_region
    spawn_x = (xmin + xmax) / 2
    spawn_y = (ymin + ymax) / 2
    agent_z = floor_z + 0.16

    agent = ET.SubElement(
        worldbody, "body", name="agent", pos=f"{spawn_x:.4f} {spawn_y:.4f} {agent_z:.4f}"
    )
    ET.SubElement(agent, "joint", name="agent_x", type="slide", axis="1 0 0", damping="0.5")
    ET.SubElement(agent, "joint", name="agent_y", type="slide", axis="0 1 0", damping="0.5")
    ET.SubElement(agent, "geom", attrib={"class": "agent"}, name="agent_geom")
    ET.SubElement(agent, "site", name="agent_site", pos="0 0 0", size="0.02")

    ET.SubElement(
        worldbody,
        "site",
        name="goal",
        pos=f"{xmax - 0.5:.4f} {ymax - 0.5:.4f} {floor_z + 0.05:.4f}",
        size="0.18",
        rgba="1 0.7 0.1 0.45",
        type="sphere",
    )

    actuator = ET.SubElement(mujoco, "actuator")
    ET.SubElement(
        actuator,
        "velocity",
        name="vx",
        joint="agent_x",
        kv="6",
        ctrlrange="-2 2",
    )
    ET.SubElement(
        actuator,
        "velocity",
        name="vy",
        joint="agent_y",
        kv="6",
        ctrlrange="-2 2",
    )

    sensor = ET.SubElement(mujoco, "sensor")
    ET.SubElement(sensor, "framepos", name="agent_pos", objtype="site", objname="agent_site")
    ET.SubElement(sensor, "framepos", name="goal_pos", objtype="site", objname="goal")

    _xml_indent(mujoco)
    tree = ET.ElementTree(mujoco)
    mjcf_path = out_dir / "scene.xml"
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)
    return mjcf_path


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def _spawn_region_from_bounds(bounds: np.ndarray, margin: float = 0.4) -> tuple[float, float, float, float]:
    xmin = float(bounds[0][0]) + margin
    xmax = float(bounds[1][0]) - margin
    ymin = float(bounds[0][1]) + margin
    ymax = float(bounds[1][1]) - margin
    if xmax <= xmin:
        xmin, xmax = float(bounds[0][0]), float(bounds[1][0])
    if ymax <= ymin:
        ymin, ymax = float(bounds[0][1]), float(bounds[1][1])
    return xmin, xmax, ymin, ymax


def build_environment(cfg: BuildConfig) -> BuildArtifacts:
    out_dir = Path(cfg.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    raw = trimesh.load(cfg.mesh_path, force="mesh")
    if isinstance(raw, trimesh.Scene):
        raw = trimesh.util.concatenate(tuple(raw.geometry.values()))

    mesh, floor_z, raw_to_sim = preprocess(raw, cfg)
    bounds = mesh.bounds
    spawn_region = _spawn_region_from_bounds(bounds)

    # Phone scans (dense, noisy, partial): ground the nav world at the REAL
    # floor and collide against 2.5D occupancy columns. Convex hulls of a
    # curved shell wrap free space and leave the agent "in collision"
    # everywhere; box columns hug the actual walls/furniture.
    is_scan = len(mesh.vertices) >= 20_000
    boxes: list[tuple[float, float, float, float, float, float]] = []
    if is_scan:
        floor_z = _real_floor_z(mesh)
        boxes = _occupancy_columns(mesh, floor_z)
        hulls, classes = [], []
    else:
        hulls = decompose(mesh, cfg)
        classes = [classify_hull(h, bounds) for h in hulls]
        keep_idx = [i for i, c in enumerate(classes) if c != "floor"]
        hulls = [hulls[i] for i in keep_idx]
        classes = [classes[i] for i in keep_idx]

    mjcf_path = write_mjcf(
        hulls=hulls,
        classes=classes,
        out_dir=out_dir,
        floor_z=floor_z,
        spawn_region=spawn_region,
        bounds=bounds,
        enclose=cfg.enclose,
        wall_margin=cfg.wall_margin,
        wall_thickness=cfg.wall_thickness,
        boxes=boxes,
    )

    materials = {f"hull_{i:04d}": MATERIAL_TABLE[c] for i, c in enumerate(classes)}

    metadata = {
        "bounds_min": bounds[0].tolist(),
        "bounds_max": bounds[1].tolist(),
        "floor_z": float(floor_z),
        "spawn_region": list(spawn_region),
        "n_hulls": len(hulls) + len(boxes),
        "raw_to_sim": raw_to_sim.tolist(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return BuildArtifacts(
        mjcf_path=mjcf_path,
        mesh_dir=out_dir / "meshes",
        n_hulls=len(hulls) + len(boxes),
        bounds=bounds,
        floor_z=floor_z,
        spawn_region=spawn_region,
        materials=materials,
        raw_to_sim=raw_to_sim,
    )
