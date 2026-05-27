"""Replay route: queues a Run + emits Inngest event. Returns the run row
with status='pending'; the frontend polls /runs/{id} until status='ok' and
then loads the trajectories JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import inngest
from fastapi import APIRouter, HTTPException
from nanoid import generate as nanoid
from pydantic import BaseModel
from sqlalchemy import select

from src.deps import DbSession, ProjectDep
from src.inngest_client import inngest_client
from src.models import Build, Policy, Run
from src.schemas import RunOut

router = APIRouter()


class ReplayRequest(BaseModel):
    policy: str = "greedy"  # 'random' | 'greedy' | 'ppo'
    episodes: int = 5
    max_steps: int = 300
    seed: int = 0
    policy_id: str | None = None


@router.post("/{project_id}/replay", response_model=RunOut)
async def replay(project: ProjectDep, body: ReplayRequest, db: DbSession) -> Run:
    build = db.scalars(
        select(Build)
        .where(Build.project_id == project.id, Build.status == "ok")
        .order_by(Build.created_at.desc())
    ).first()
    if not build:
        raise HTTPException(404, "no built scene — run Build first")

    policy_id = body.policy_id
    if body.policy == "ppo" and policy_id is None:
        policy = db.scalars(
            select(Policy).where(Policy.build_id == build.id).order_by(Policy.created_at.desc())
        ).first()
        if policy is None or not policy.ckpt_path or not Path(policy.ckpt_path).exists():
            raise HTTPException(400, "no trained PPO policy — train one first")
        policy_id = policy.id

    run = Run(
        id=nanoid(size=12),
        policy_id=policy_id if body.policy == "ppo" else None,
        baseline=body.policy if body.policy in ("random", "greedy") else None,
        status="pending",
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    await inngest_client.send(
        events=[
            inngest.Event(
                name="replay/requested",
                data={
                    "run_id": run.id,
                    "policy": body.policy,
                    "policy_id": policy_id,
                    "project_id": project.id,
                    "episodes": body.episodes,
                    "max_steps": body.max_steps,
                    "seed": body.seed,
                },
            )
        ]
    )
    return run


@router.get("/{project_id}/runs/{run_id}", response_model=RunOut)
def get_run(project: ProjectDep, run_id: str, db: DbSession) -> Run:
    r = db.get(Run, run_id)
    if r is None:
        raise HTTPException(404, f"unknown run {run_id}")
    return r


@router.get("/{project_id}/runs/{run_id}/trajectories")
def get_run_trajectories(project: ProjectDep, run_id: str, db: DbSession) -> dict:
    r = db.get(Run, run_id)
    if r is None or not r.trajectories_path:
        raise HTTPException(404, "run has no trajectories yet")
    p = Path(r.trajectories_path)
    if not p.exists():
        raise HTTPException(404, "trajectories file missing on disk")
    return json.loads(p.read_text())
