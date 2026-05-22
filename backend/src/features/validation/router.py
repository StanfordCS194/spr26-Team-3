"""Validation route: queues a Validation row + emits Inngest event.
The function runs the 6-check catalog and persists `report`.
"""
from __future__ import annotations

import inngest
from fastapi import APIRouter, HTTPException
from nanoid import generate as nanoid
from sqlalchemy import select

from src.deps import DbSession, ProjectDep
from src.inngest_client import inngest_client
from src.models import Reconstruction, Validation
from src.schemas import ValidationOut

router = APIRouter()


@router.post("/{project_id}/validate", response_model=ValidationOut)
async def validate_project(project: ProjectDep, db: DbSession) -> Validation:
    recon = db.scalars(
        select(Reconstruction)
        .where(Reconstruction.project_id == project.id, Reconstruction.status == "ok")
        .order_by(Reconstruction.created_at.desc())
    ).first()
    if not recon:
        raise HTTPException(
            400, "no successful reconstruction for this project — run Reconstruct first"
        )

    v = Validation(
        id=nanoid(size=12),
        reconstruction_id=recon.id,
        status="pending",
    )
    db.add(v)
    db.commit()
    db.refresh(v)

    await inngest_client.send(
        events=[
            inngest.Event(
                name="validation/requested",
                data={"validation_id": v.id},
            )
        ]
    )
    return v


@router.get("/{project_id}/validate/latest", response_model=ValidationOut | None)
def latest_validation(project: ProjectDep, db: DbSession) -> Validation | None:
    recon = db.scalars(
        select(Reconstruction)
        .where(Reconstruction.project_id == project.id)
        .order_by(Reconstruction.created_at.desc())
    ).first()
    if not recon:
        return None
    return db.scalars(
        select(Validation)
        .where(Validation.reconstruction_id == recon.id)
        .order_by(Validation.created_at.desc())
    ).first()
