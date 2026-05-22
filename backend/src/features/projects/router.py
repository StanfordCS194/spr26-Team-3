"""Project CRUD + project-level state derivation."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select

from src.deps import DbSession, ProjectDep
from src.models import Build, Policy, Project, Reconstruction, Validation
from src.schemas import ProjectCreate, ProjectOut

router = APIRouter()


class StageState(BaseModel):
    complete: bool
    reason: str | None = None  # human-readable explanation when not complete


class ProjectState(BaseModel):
    capture: StageState
    reconstruct: StageState
    validate: StageState
    build: StageState
    train: StageState
    replay: StageState


@router.get("", response_model=list[ProjectOut])
def list_projects(db: DbSession) -> list[Project]:
    return list(db.scalars(select(Project).order_by(Project.created_at.desc())))


@router.post("", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
def create_project(body: ProjectCreate, db: DbSession) -> Project:
    p = Project(name=body.name)
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


@router.get("/{project_id}", response_model=ProjectOut)
def get_project_route(project: ProjectDep) -> Project:
    return project


@router.patch("/{project_id}", response_model=ProjectOut)
def update_project(project: ProjectDep, body: ProjectCreate, db: DbSession) -> Project:
    project.name = body.name
    db.commit()
    db.refresh(project)
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project: ProjectDep, db: DbSession) -> None:
    db.delete(project)
    db.commit()


@router.get("/{project_id}/state", response_model=ProjectState)
def get_project_state(project: ProjectDep, db: DbSession) -> ProjectState:
    """Derive per-stage completion from existing artifacts.

    Used by the frontend's StepNav to gate locked stages — a user can't open
    Replay until they've built, can't open Train until reconstruction +
    validation pass, etc. PR-A only the Build chain has real artifacts; the
    other stages report `complete=False` with a "lands in PR-X" reason.
    """
    has_video = project.video_path is not None
    last_recon = db.scalars(
        select(Reconstruction)
        .where(Reconstruction.project_id == project.id, Reconstruction.status == "ok")
        .order_by(Reconstruction.created_at.desc())
    ).first()
    last_validation = (
        db.scalars(
            select(Validation)
            .where(Validation.reconstruction_id == last_recon.id)
            .order_by(Validation.created_at.desc())
        ).first()
        if last_recon
        else None
    )
    validation_passed = last_validation is not None and (
        last_validation.user_override
        or (last_validation.report or {}).get("overall") in ("pass", "warn")
    )
    last_build = db.scalars(
        select(Build).where(Build.project_id == project.id).order_by(Build.created_at.desc())
    ).first()
    last_policy = (
        db.scalars(
            select(Policy).where(Policy.build_id == last_build.id).order_by(Policy.created_at.desc())
        ).first()
        if last_build
        else None
    )

    return ProjectState(
        capture=StageState(
            complete=has_video,
            reason=None if has_video else "Capture lands in PR-B. No video uploaded yet.",
        ),
        reconstruct=StageState(
            complete=last_recon is not None,
            reason=None if last_recon else "Reconstruct lands in PR-B.",
        ),
        validate=StageState(
            complete=validation_passed,
            reason=None if validation_passed else "Validate lands in PR-B.",
        ),
        build=StageState(
            complete=last_build is not None,
            reason=None if last_build else "No build yet. Click 'Build env'.",
        ),
        train=StageState(
            complete=last_policy is not None,
            reason=None if last_policy else "Train lands in PR-C.",
        ),
        replay=StageState(
            complete=last_build is not None,
            reason=None if last_build else "Complete Build first.",
        ),
    )
