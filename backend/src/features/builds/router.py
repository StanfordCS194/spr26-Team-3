"""Build routes — call into rl_env.build.

In PR-A the project doesn't have a real reconstruction yet, so we accept a
`fixture` flag that uses the procedural sample room from `rl_env.sample_room`.
PR-B switches the default path to use the project's latest reconstruction.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from rl_env.build import BuildConfig, build_environment
from rl_env.sample_room import write_sample_room
from src.config import get_settings
from src.deps import DbSession, ProjectDep
from src.models import Build, Reconstruction
from src.schemas import BuildOut, BuildRequest

router = APIRouter()


@router.post("/{project_id}/build", response_model=BuildOut)
def build_project(
    project: ProjectDep,
    body: BuildRequest,
    db: DbSession,
) -> Build:
    settings = get_settings()
    project_dir = settings.data_dir / "projects" / project.id
    project_dir.mkdir(parents=True, exist_ok=True)

    # Resolve mesh: latest successful reconstruction, OR the fixture sample
    # room if no reconstruction exists yet (PR-A pause state).
    recon: Reconstruction | None = None
    if body.reconstruction_id:
        recon = db.get(Reconstruction, body.reconstruction_id)
    else:
        recon = db.scalars(
            select(Reconstruction)
            .where(Reconstruction.project_id == project.id, Reconstruction.status == "ok")
            .order_by(Reconstruction.created_at.desc())
        ).first()

    if recon and recon.mesh_path:
        mesh_path = Path(recon.mesh_path)
    else:
        # Fixture path so PR-A demo works without any reconstruction.
        mesh_path = project_dir / "fixture_sample_room.obj"
        if not mesh_path.exists():
            write_sample_room(mesh_path)

    out_dir = project_dir / "build"
    artifacts = build_environment(
        BuildConfig(
            mesh_path=str(mesh_path),
            out_dir=str(out_dir),
            target_diagonal_m=body.target_diagonal_m or settings.default_target_diagonal_m,
            up_axis=body.up_axis,
            max_hulls=body.max_hulls or settings.default_max_hulls,
        )
    )

    build = Build(
        project_id=project.id,
        reconstruction_id=recon.id if recon else None,
        mjcf_path=str(artifacts.mjcf_path),
        n_hulls=artifacts.n_hulls,
        bounds={
            "min": artifacts.bounds[0].tolist(),
            "max": artifacts.bounds[1].tolist(),
        },
        spawn_region={
            "xmin": artifacts.spawn_region[0],
            "xmax": artifacts.spawn_region[1],
            "ymin": artifacts.spawn_region[2],
            "ymax": artifacts.spawn_region[3],
        },
    )
    db.add(build)
    db.commit()
    db.refresh(build)
    return build
