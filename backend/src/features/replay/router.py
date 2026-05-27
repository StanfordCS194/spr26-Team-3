"""Replay — rollout against the project's latest build with baseline policies.

PR-A: random + greedy baselines so the pause-state demo works without any
trained policy. PPO baseline added in PR-C.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from rl_env.env import NavEnv, TaskConfig
from src.deps import DbSession, ProjectDep
from src.models import Build, Run

router = APIRouter()


class ReplayRequest(BaseModel):
    policy: str = "greedy"  # 'random' | 'greedy' | 'ppo'
    episodes: int = 5
    max_steps: int = 300
    seed: int = 0


class EpisodeOut(BaseModel):
    steps: int
    reward: float
    distance: float
    success: bool


class ReplayResponse(BaseModel):
    policy: str
    successes: int
    n_episodes: int
    avg_reward: float
    episodes: list[EpisodeOut]


def _policy_random(obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


def _policy_greedy(obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    goal_vec = obs[4:6]
    norm = float(np.linalg.norm(goal_vec))
    if norm < 1e-6:
        return np.zeros(2, dtype=np.float32)
    return (goal_vec / norm).astype(np.float32)


@router.post("/{project_id}/replay", response_model=ReplayResponse)
def replay(project: ProjectDep, body: ReplayRequest, db: DbSession) -> ReplayResponse:
    build = db.scalars(
        select(Build).where(Build.project_id == project.id).order_by(Build.created_at.desc())
    ).first()
    if not build:
        raise HTTPException(404, "no build for this project — call POST /build first")

    if body.policy == "ppo":
        raise HTTPException(501, "ppo baseline lands in PR-C")

    policy_fn = _policy_greedy if body.policy == "greedy" else _policy_random

    env = NavEnv(build.mjcf_path, task=TaskConfig(max_steps=body.max_steps))
    rng = np.random.default_rng(body.seed)

    episodes: list[EpisodeOut] = []
    for ep in range(body.episodes):
        obs, _ = env.reset(seed=body.seed + ep)
        ep_reward = 0.0
        ep_steps = 0
        info: dict = {}
        for _ in range(body.max_steps):
            a = policy_fn(obs, rng)
            obs, r, term, trunc, info = env.step(a)
            ep_reward += r
            ep_steps += 1
            if term or trunc:
                break
        episodes.append(
            EpisodeOut(
                steps=ep_steps,
                reward=round(float(ep_reward), 3),
                distance=round(float(info.get("distance", -1)), 3),
                success=bool(info.get("success", False)),
            )
        )
    env.close()

    successes = sum(int(e.success) for e in episodes)
    avg_reward = float(np.mean([e.reward for e in episodes]))

    run = Run(
        policy_id=None,
        baseline=body.policy,
        episodes=body.episodes,
        successes=successes,
        avg_reward=avg_reward,
    )
    db.add(run)
    db.commit()

    return ReplayResponse(
        policy=body.policy,
        successes=successes,
        n_episodes=body.episodes,
        avg_reward=round(avg_reward, 3),
        episodes=episodes,
    )
