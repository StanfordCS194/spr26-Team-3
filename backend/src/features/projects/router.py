"""Project CRUD + project-level state derivation + run history + export."""
from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select

from src.config import get_settings
from src.deps import DbSession, ProjectDep
from src.models import Build, Policy, Project, Reconstruction, Run, Validation
from src.schemas import ProjectCreate, ProjectOut, RunOut

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


# NOTE: Static paths must be declared BEFORE the dynamic /{project_id}
# routes below, otherwise FastAPI's matcher treats "summary" as a project id.


class ProjectSummary(BaseModel):
    id: str
    name: str
    created_at: str
    thumbnail_path: str | None
    status_pill: str  # "New" | "Captured" | "Reconstructed" | "Validated" | "Built" | "Trained — N%"
    n_runs: int


@router.get("/summary", response_model=list[ProjectSummary])
def list_projects_summary(db: DbSession) -> list[ProjectSummary]:
    """Sidebar query: lists all projects with derived status pill in one
    round-trip. Joins across child tables so the frontend doesn't N+1.
    """
    projects = list(db.scalars(select(Project).order_by(Project.created_at.desc())))
    summaries: list[ProjectSummary] = []
    for p in projects:
        has_video = bool(p.video_path)
        ok_recon = db.scalars(
            select(Reconstruction)
            .where(Reconstruction.project_id == p.id, Reconstruction.status == "ok")
            .order_by(Reconstruction.created_at.desc())
        ).first()
        latest_validation = None
        if ok_recon:
            latest_validation = db.scalars(
                select(Validation)
                .where(Validation.reconstruction_id == ok_recon.id)
                .order_by(Validation.created_at.desc())
            ).first()
        latest_build = db.scalars(
            select(Build).where(Build.project_id == p.id).order_by(Build.created_at.desc())
        ).first()
        latest_policy: Policy | None = None
        if latest_build:
            latest_policy = db.scalars(
                select(Policy)
                .where(Policy.build_id == latest_build.id)
                .order_by(Policy.created_at.desc())
            ).first()
        best_run: Run | None = None
        if latest_policy:
            best_run = db.scalars(
                select(Run)
                .where(Run.policy_id == latest_policy.id)
                .order_by(Run.successes.desc(), Run.created_at.desc())
            ).first()

        if latest_policy:
            if best_run and best_run.episodes > 0:
                pct = int(100 * best_run.successes / best_run.episodes)
                pill = f"Trained — {pct}%"
            else:
                pill = "Trained"
        elif latest_build:
            pill = "Built"
        elif latest_validation:
            ov = (latest_validation.report or {}).get("overall")
            pill = "Validated" if ov in ("pass", "warn") else "Validation failed"
        elif ok_recon:
            pill = "Reconstructed"
        elif has_video:
            pill = "Captured"
        else:
            pill = "New"

        n_runs = 0
        if latest_build:
            policies = list(db.scalars(select(Policy).where(Policy.build_id == latest_build.id)))
            if policies:
                policy_ids = [pol.id for pol in policies]
                n_runs = (
                    db.query(Run).filter(Run.policy_id.in_(policy_ids)).count()
                    if policy_ids
                    else 0
                )

        summaries.append(
            ProjectSummary(
                id=p.id,
                name=p.name,
                created_at=p.created_at.isoformat() if p.created_at else "",
                thumbnail_path=p.thumbnail_path,
                status_pill=pill,
                n_runs=n_runs,
            )
        )
    return summaries


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
    """Removes the project row (cascades child rows) and the on-disk
    folder under data/projects/<id>/."""
    settings = get_settings()
    project_id = project.id
    db.delete(project)
    db.commit()
    folder = settings.data_dir / "projects" / project_id
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)


@router.get("/{project_id}/runs", response_model=list[RunOut])
def list_project_runs(project: ProjectDep, db: DbSession) -> list[Run]:
    builds = list(db.scalars(select(Build).where(Build.project_id == project.id)))
    if not builds:
        return []
    policy_ids = [
        pol.id for pol in db.scalars(select(Policy).where(Policy.build_id.in_([b.id for b in builds])))
    ]
    # Include both PPO runs (policy_id set) and baselines (policy_id NULL but build relevant)
    # Baselines don't have a build link in the schema currently — we keep it simple and just
    # return policy-linked runs. Baselines are still visible inline in Replay.
    if not policy_ids:
        return []
    return list(
        db.scalars(
            select(Run).where(Run.policy_id.in_(policy_ids)).order_by(Run.created_at.desc())
        )
    )


@router.post("/{project_id}/export")
def export_project(project: ProjectDep) -> StreamingResponse:
    """Zip data/projects/<id>/ and stream it. Self-contained: mesh +
    scene.xml + policy.zip + trajectories under one root."""
    settings = get_settings()
    folder = settings.data_dir / "projects" / project.id
    if not folder.exists():
        raise HTTPException(404, "no data on disk for this project")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in folder.rglob("*"):
            if path.is_file():
                arcname = Path(project.id) / path.relative_to(folder)
                zf.write(path, str(arcname))
    buf.seek(0)
    safe_name = project.name.replace(" ", "_")
    filename = f"{safe_name}-{project.id}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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
