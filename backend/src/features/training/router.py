"""Training routes — emits Inngest event, function does the work."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException
from nanoid import generate as nanoid
from sqlalchemy import select

from src.deps import DbSession, ProjectDep
from src.models import Build, Policy
from src.schemas import PolicyOut, TrainRequest

router = APIRouter()


@router.post("/{project_id}/train", response_model=PolicyOut)
async def queue_training(
    project: ProjectDep,
    body: TrainRequest,
    db: DbSession,
    background_tasks: BackgroundTasks,
) -> Policy:
    build = db.scalars(
        select(Build).where(Build.project_id == project.id).order_by(Build.created_at.desc())
    ).first()
    if not build:
        raise HTTPException(400, "no build for this project — call POST /build first")

    policy = Policy(
        id=nanoid(size=12),
        build_id=build.id,
        algo="ppo",
        ckpt_path="",
        total_steps=body.total_steps,
        metrics={"progress": 0.0, "steps": 0, "queued": True},
    )
    db.add(policy)
    db.commit()
    db.refresh(policy)

    # Train spawning on the SCANNED floor when we can profile it — a partial
    # scan's bounding-box corners are empty space the demo never uses, and a
    # policy trained there can't navigate the real corridor.
    spawn_cells = None
    try:
        from src.features.replay.router import _nav_footprint
        from src.models import Reconstruction

        recon = (
            db.get(Reconstruction, build.reconstruction_id)
            if build.reconstruction_id
            else None
        )
        fp = _nav_footprint(build, recon.mesh_path if recon else None)
        spawn_cells = fp["free_cells"] if fp else None
    except Exception:
        spawn_cells = None

    background_tasks.add_task(
        _run_training_blocking, policy.id, build.mjcf_path, body.total_steps,
        body.n_envs, 300, body.seed, spawn_cells,
    )

    return policy


def _run_training_blocking(
    policy_id: str, mjcf_path: str, total_steps: int, n_envs: int, max_steps: int, seed: int,
    spawn_cells: list | None = None,
) -> None:
    """In-process PPO training (host mode, no Inngest). run_training updates
    the Policy row's metrics/status as it goes."""
    import traceback

    from src.db import SessionLocal
    from src.features.training.service import run_training

    try:
        run_training(
            policy_id=policy_id, mjcf_path=mjcf_path, total_steps=total_steps,
            n_envs=n_envs, max_steps=max_steps, seed=seed, spawn_cells=spawn_cells,
        )
    except Exception as exc:
        from src.models import Policy

        with SessionLocal() as db:
            p = db.get(Policy, policy_id)
            if p is not None:
                p.metrics = {**(p.metrics or {}), "error": f"{exc.__class__.__name__}: {exc}"[:500]}
                db.commit()
        traceback.print_exc()


@router.get("/{project_id}/policies", response_model=list[PolicyOut])
def list_policies(project: ProjectDep, db: DbSession) -> list[Policy]:
    builds = list(db.scalars(select(Build).where(Build.project_id == project.id)))
    if not builds:
        return []
    build_ids = [b.id for b in builds]
    return list(
        db.scalars(
            select(Policy)
            .where(Policy.build_id.in_(build_ids))
            .order_by(Policy.created_at.desc())
        )
    )


@router.get("/{project_id}/policies/{policy_id}", response_model=PolicyOut)
def get_policy(project: ProjectDep, policy_id: str, db: DbSession) -> Policy:
    pol = db.get(Policy, policy_id)
    if pol is None:
        raise HTTPException(404, f"unknown policy {policy_id}")
    return pol


@router.post("/{project_id}/policies/{policy_id}/cancel", response_model=PolicyOut)
def cancel_policy(project: ProjectDep, policy_id: str, db: DbSession) -> Policy:
    """Flag a running training job to stop. The training loop checks this flag
    each logging interval and halts (the partially-trained policy is kept)."""
    pol = db.get(Policy, policy_id)
    if pol is None:
        raise HTTPException(404, f"unknown policy {policy_id}")
    m = dict(pol.metrics or {})
    if not m.get("done") and not m.get("error"):
        m["cancelled"] = True
        pol.metrics = m  # reassign so SQLAlchemy tracks the JSONB change
        db.commit()
        db.refresh(pol)
    return pol
