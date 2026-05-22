"""Training routes: kick off PPO + poll progress."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from nanoid import generate as nanoid
from sqlalchemy import select

from src.deps import DbSession, ProjectDep
from src.features.training.service import start_training
from src.models import Build, Policy
from src.schemas import PolicyOut, TrainRequest

router = APIRouter()


@router.post("/{project_id}/train", response_model=PolicyOut)
def queue_training(
    project: ProjectDep,
    body: TrainRequest,
    db: DbSession,
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
        ckpt_path="",  # set when training completes
        total_steps=body.total_steps,
        metrics={"progress": 0.0, "steps": 0},
    )
    db.add(policy)
    db.commit()
    db.refresh(policy)

    start_training(
        policy.id,
        build.mjcf_path,
        total_steps=body.total_steps,
        n_envs=body.n_envs,
        max_steps=300,
        seed=body.seed,
    )
    return policy


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
