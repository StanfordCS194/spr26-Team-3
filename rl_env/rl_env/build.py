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


@dataclass
class BuildArtifacts:
    mjcf_path: Path
    mesh_dir: Path
    n_hulls: int
    bounds: np.ndarray  # (2, 3) AABB in MJCF coords (Z-up)
    floor_z: float
    spawn_region: tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
    materials: dict[str, dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------------


def _detect_up_axis(mesh: trimesh.Trimesh) -> str:
    """Heuristic: the up axis is whichever has the largest vertical-extent / horizontal-extent ratio.

    For room-like meshes this is unreliable, so callers should override when known.
    Falls back to 'y' (most photo-depth pipelines).
    """
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


def preprocess(mesh: trimesh.Trimesh, cfg: BuildConfig) -> tuple[trimesh.Trimesh, float]:
    """Return (mesh in MuJoCo Z-up coords, floor_z)."""
    up = cfg.up_axis if cfg.up_axis != "auto" else _detect_up_axis(mesh)
    mesh = _to_z_up(mesh, up)

    mins = mesh.bounds[0]
    maxs = mesh.bounds[1]
    center_xy = (mins[:2] + maxs[:2]) / 2
    mesh.apply_translation((-center_xy[0], -center_xy[1], -mins[2]))

    if cfg.target_diagonal_m is not None:
        ext = mesh.extents
        diag_xy = float(np.hypot(ext[0], ext[1]))
        if diag_xy > 1e-6:
            scale = cfg.target_diagonal_m / diag_xy
            mesh.apply_scale(scale)

    floor_z = float(mesh.bounds[0][2])
    return mesh, floor_z


# ---------------------------------------------------------------------------
# Convex decomposition
# ---------------------------------------------------------------------------


def decompose(mesh: trimesh.Trimesh, cfg: BuildConfig) -> list[trimesh.Trimesh]:
    """Return a list of convex hull meshes covering `mesh`.

    Strategy:
      1. If trimesh has a working VHACD/CoACD backend, use it.
      2. Else split into connected components and take convex hull of each.
      3. If that yields too few hulls (e.g. one big merged mesh), fall back to
         a single convex hull of the entire scene — coarse but always works.

    The MVP intentionally accepts coarse collision geometry; tightening this
    is a known follow-up (PRD: "balance collision fidelity against simulation
    speed", Key technical challenge #1).
    """
    if cfg.decompose:
        try:
            hulls = trimesh.decomposition.convex_decomposition(
                mesh, maxNumVerticesPerCH=64, resolution=100_000
            )
            if hulls:
                hulls = [h for h in hulls if h.is_volume and h.volume > 1e-6]
                if hulls:
                    return hulls[: cfg.max_hulls]
        except Exception:
            pass

    components = mesh.split(only_watertight=False)
    if len(components) == 0:
        components = [mesh]

    hulls: list[trimesh.Trimesh] = []
    for c in components[: cfg.max_hulls]:
        try:
            h = c.convex_hull
            if h.is_volume and h.volume > 1e-6:
                hulls.append(h)
        except Exception:
            continue

    if not hulls:
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

    mesh, floor_z = preprocess(raw, cfg)
    bounds = mesh.bounds
    spawn_region = _spawn_region_from_bounds(bounds)

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
    )

    materials = {f"hull_{i:04d}": MATERIAL_TABLE[c] for i, c in enumerate(classes)}

    metadata = {
        "bounds_min": bounds[0].tolist(),
        "bounds_max": bounds[1].tolist(),
        "floor_z": float(floor_z),
        "spawn_region": list(spawn_region),
        "n_hulls": len(hulls),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return BuildArtifacts(
        mjcf_path=mjcf_path,
        mesh_dir=out_dir / "meshes",
        n_hulls=len(hulls),
        bounds=bounds,
        floor_z=floor_z,
        spawn_region=spawn_region,
        materials=materials,
    )
