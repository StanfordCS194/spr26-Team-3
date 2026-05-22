"""Validation route: run the mesh check catalog against a project's latest
reconstruction.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from src.deps import DbSession, ProjectDep
from src.features.validation.checks import run_all
from src.models import Reconstruction, Validation
from src.schemas import ValidationOut

router = APIRouter()


@router.post("/{project_id}/validate", response_model=ValidationOut)
def validate_project(project: ProjectDep, db: DbSession) -> Validation:
    recon = db.scalars(
        select(Reconstruction)
        .where(Reconstruction.project_id == project.id, Reconstruction.status == "ok")
        .order_by(Reconstruction.created_at.desc())
    ).first()
    if not recon:
        raise HTTPException(
            400, "no successful reconstruction for this project — run Reconstruct first"
        )
    if not recon.mesh_path:
        raise HTTPException(500, f"reconstruction {recon.id} has no mesh_path")

    report = run_all(recon.mesh_path)
    v = Validation(reconstruction_id=recon.id, report=report)
    db.add(v)
    db.commit()
    db.refresh(v)
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
