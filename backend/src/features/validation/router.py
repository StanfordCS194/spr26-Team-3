"""Validation route: queues a Validation row + emits Inngest event.
The function runs the 6-check catalog and persists `report`.
"""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException
from nanoid import generate as nanoid
from sqlalchemy import select

from src.deps import DbSession, ProjectDep
from src.models import Reconstruction, Validation
from src.schemas import ValidationOut

router = APIRouter()


@router.post("/{project_id}/validate", response_model=ValidationOut)
async def validate_project(
    project: ProjectDep, db: DbSession, background_tasks: BackgroundTasks
) -> Validation:
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

    background_tasks.add_task(_run_validation_blocking, v.id)
    return v


def _run_validation_blocking(validation_id: str) -> None:
    """In-process mesh validation (host mode, no Inngest). Runs the 6-check
    catalog and persists the report."""
    import traceback

    from src.db import SessionLocal
    from src.features.validation.checks import run_all

    try:
        with SessionLocal() as db:
            v = db.get(Validation, validation_id)
            if v is None:
                return
            v.status = "running"
            db.commit()
            recon = db.get(Reconstruction, v.reconstruction_id)
            if recon is None or not recon.mesh_path:
                raise RuntimeError("upstream reconstruction has no mesh")
            mesh_path = recon.mesh_path

        report = run_all(mesh_path)

        with SessionLocal() as db:
            v = db.get(Validation, validation_id)
            assert v is not None
            v.report = report
            v.status = "ok"
            db.commit()
    except Exception as exc:
        with SessionLocal() as db:
            v = db.get(Validation, validation_id)
            if v is not None:
                v.status = "failed"
                v.error = f"{exc.__class__.__name__}: {exc}"[:1000]
                db.commit()
        traceback.print_exc()


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


@router.post("/{project_id}/validate/cancel", response_model=ValidationOut | None)
def cancel_validation(project: ProjectDep, db: DbSession) -> Validation | None:
    recon = db.scalars(
        select(Reconstruction)
        .where(Reconstruction.project_id == project.id)
        .order_by(Reconstruction.created_at.desc())
    ).first()
    if not recon:
        return None
    v = db.scalars(
        select(Validation)
        .where(
            Validation.reconstruction_id == recon.id,
            Validation.status.in_(["pending", "running"]),
        )
        .order_by(Validation.created_at.desc())
    ).first()
    if v is None:
        return None
    v.status = "cancelled"
    db.commit()
    db.refresh(v)
    return v
