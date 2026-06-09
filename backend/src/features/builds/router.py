"""Build route: queues a Build row + emits Inngest event."""
from __future__ import annotations

import inngest
from fastapi import APIRouter, HTTPException
from nanoid import generate as nanoid
from sqlalchemy import select

from src.deps import DbSession, ProjectDep
from src.inngest_client import inngest_client
from src.models import Build, Reconstruction
from src.schemas import BuildOut, BuildRequest

router = APIRouter()


@router.post("/{project_id}/build", response_model=BuildOut)
async def build_project(
    project: ProjectDep,
    body: BuildRequest,
    db: DbSession,
) -> Build:
    recon: Reconstruction | None = None
    if body.reconstruction_id:
        recon = db.get(Reconstruction, body.reconstruction_id)
    else:
        recon = db.scalars(
            select(Reconstruction)
            .where(Reconstruction.project_id == project.id, Reconstruction.status == "ok")
            .order_by(Reconstruction.created_at.desc())
        ).first()

    if not recon or not recon.mesh_path:
        raise HTTPException(
            400, "no successful reconstruction — run Reconstruct first"
        )

    build = Build(
        id=nanoid(size=12),
        project_id=project.id,
        reconstruction_id=recon.id,
        status="pending",
    )
    db.add(build)
    db.commit()
    db.refresh(build)

    await inngest_client.send(
        events=[
            inngest.Event(
                name="build/requested",
                data={
                    "build_id": build.id,
                    "target_diagonal_m": body.target_diagonal_m,
                    "max_hulls": body.max_hulls,
                    "up_axis": body.up_axis,
                    "enclose": body.enclose,
                },
            )
        ]
    )

    return build


@router.get("/{project_id}/build/latest", response_model=BuildOut | None)
def latest_build(project: ProjectDep, db: DbSession) -> Build | None:
    return db.scalars(
        select(Build)
        .where(Build.project_id == project.id)
        .order_by(Build.created_at.desc())
    ).first()


@router.post("/{project_id}/build/cancel", response_model=BuildOut | None)
def cancel_build(project: ProjectDep, db: DbSession) -> Build | None:
    build = db.scalars(
        select(Build)
        .where(
            Build.project_id == project.id,
            Build.status.in_(["pending", "running"]),
        )
        .order_by(Build.created_at.desc())
    ).first()
    if build is None:
        return None
    build.status = "cancelled"
    db.commit()
    db.refresh(build)
    return build
