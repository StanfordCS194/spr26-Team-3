"""Procedural sample room mesh — a fallback for users without a scan.

Produces an OBJ file shaped like a small room with a floor, four walls,
and two box obstacles. Used by the `demo` CLI subcommand and as a fixture
for tests that don't want to depend on the prototype's photo-to-3D output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def make_sample_room(
    size: tuple[float, float, float] = (5.0, 3.0, 4.0),
    wall_thickness: float = 0.1,
    seed: int = 0,
) -> trimesh.Trimesh:
    """Generate a closed-room mesh with two interior obstacles.

    size: (x_extent, y_extent_ceiling_height, z_extent) in meters.
    Floor is at y=0, ceiling at y=size[1].
    """
    sx, sy, sz = size
    t = wall_thickness

    parts: list[trimesh.Trimesh] = []

    floor = trimesh.creation.box(extents=(sx, t, sz))
    floor.apply_translation((0, -t / 2, 0))
    parts.append(floor)

    ceiling = trimesh.creation.box(extents=(sx, t, sz))
    ceiling.apply_translation((0, sy + t / 2, 0))
    parts.append(ceiling)

    wall_n = trimesh.creation.box(extents=(sx, sy, t))
    wall_n.apply_translation((0, sy / 2, -sz / 2 - t / 2))
    parts.append(wall_n)

    wall_s = trimesh.creation.box(extents=(sx, sy, t))
    wall_s.apply_translation((0, sy / 2, sz / 2 + t / 2))
    parts.append(wall_s)

    wall_e = trimesh.creation.box(extents=(t, sy, sz))
    wall_e.apply_translation((sx / 2 + t / 2, sy / 2, 0))
    parts.append(wall_e)

    wall_w = trimesh.creation.box(extents=(t, sy, sz))
    wall_w.apply_translation((-sx / 2 - t / 2, sy / 2, 0))
    parts.append(wall_w)

    rng = np.random.default_rng(seed)
    margin = 0.6
    for _ in range(2):
        ext = rng.uniform(0.4, 0.9, size=3)
        ext[1] = rng.uniform(0.4, 1.2)
        x = rng.uniform(-sx / 2 + margin, sx / 2 - margin)
        z = rng.uniform(-sz / 2 + margin, sz / 2 - margin)
        obs = trimesh.creation.box(extents=ext)
        obs.apply_translation((x, ext[1] / 2, z))
        parts.append(obs)

    # Chair: seat, backrest, four legs. Centered roughly mid-room, rotated
    # slightly off-axis so it doesn't lattice with the box obstacles.
    cx = float(rng.uniform(-sx / 2 + margin, sx / 2 - margin))
    cz = float(rng.uniform(-sz / 2 + margin, sz / 2 - margin))
    chair_yaw = float(rng.uniform(0, np.pi))
    parts.extend(_make_chair(center=(cx, 0.0, cz), yaw=chair_yaw))

    return trimesh.util.concatenate(parts)


def _make_chair(
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    yaw: float = 0.0,
    seat_w: float = 0.45,
    seat_d: float = 0.45,
    seat_h: float = 0.06,
    seat_z_off_floor: float = 0.45,
    leg_w: float = 0.05,
    back_h: float = 0.45,
) -> list[trimesh.Trimesh]:
    """Build a simple chair from boxes. Floor is at y=0, chair stands on it."""
    parts: list[trimesh.Trimesh] = []

    # Seat
    seat = trimesh.creation.box(extents=(seat_w, seat_h, seat_d))
    seat.apply_translation((0, seat_z_off_floor + seat_h / 2, 0))
    parts.append(seat)

    # Backrest
    back = trimesh.creation.box(extents=(seat_w, back_h, seat_h))
    back.apply_translation(
        (0, seat_z_off_floor + seat_h + back_h / 2, -seat_d / 2 + seat_h / 2)
    )
    parts.append(back)

    # Four legs
    leg_h = seat_z_off_floor
    leg_offsets = [
        (seat_w / 2 - leg_w / 2, seat_d / 2 - leg_w / 2),
        (-(seat_w / 2 - leg_w / 2), seat_d / 2 - leg_w / 2),
        (seat_w / 2 - leg_w / 2, -(seat_d / 2 - leg_w / 2)),
        (-(seat_w / 2 - leg_w / 2), -(seat_d / 2 - leg_w / 2)),
    ]
    for lx, lz in leg_offsets:
        leg = trimesh.creation.box(extents=(leg_w, leg_h, leg_w))
        leg.apply_translation((lx, leg_h / 2, lz))
        parts.append(leg)

    # Rotate chair around Y, then translate to center.
    rot = trimesh.transformations.rotation_matrix(yaw, [0, 1, 0])
    cx, cy, cz = center
    trans = trimesh.transformations.translation_matrix([cx, cy, cz])
    transform = trans @ rot
    for p in parts:
        p.apply_transform(transform)
    return parts


def write_sample_room(path: str | Path, **kwargs) -> Path:
    mesh = make_sample_room(**kwargs)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out)
    return out


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "rl_env/data/sample_room.obj"
    p = write_sample_room(out)
    m = trimesh.load(p)
    print(f"wrote {p}: {len(m.vertices)} verts, {len(m.faces)} faces")
