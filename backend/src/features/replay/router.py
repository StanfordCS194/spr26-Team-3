"""Replay — rollout against the project's latest build with baseline policies
or a trained PPO checkpoint. Returns per-step trajectories so the frontend
can render them spatially.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from rl_env.env import NavEnv, TaskConfig
from src.deps import DbSession, ProjectDep
from src.models import Build, Policy, Run

router = APIRouter()


class ReplayRequest(BaseModel):
    policy: str = "greedy"  # 'random' | 'greedy' | 'ppo'
    episodes: int = 5
    max_steps: int = 300
    seed: int = 0
    policy_id: str | None = None  # for 'ppo'; defaults to latest
    include_trajectories: bool = True


class TrajectoryPoint(BaseModel):
    step: int
    x: float
    y: float
    collision: bool


class EpisodeOut(BaseModel):
    steps: int
    reward: float
    distance: float
    success: bool
    failure_class: str  # 'success' | 'timeout' | 'stuck' | 'collided' | 'near-miss'
    spawn: list[float]
    goal: list[float]
    trajectory: list[TrajectoryPoint] | None = None


class ReplayResponse(BaseModel):
    policy: str
    successes: int
    n_episodes: int
    avg_reward: float
    bounds: dict
    spawn_region: dict
    episodes: list[EpisodeOut]


def _policy_random(obs, rng):
    return rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


def _policy_greedy(obs, rng):
    goal_vec = obs[4:6]
    norm = float(np.linalg.norm(goal_vec))
    if norm < 1e-6:
        return np.zeros(2, dtype=np.float32)
    return (goal_vec / norm).astype(np.float32)


def _classify(steps: int, max_steps: int, distance: float, success: bool, collisions: int, last_speeds: list[float]) -> str:
    if success:
        return "success"
    if collisions >= 3:
        return "collided"
    if distance < 1.0:
        return "near-miss"
    if last_speeds and max(last_speeds) < 0.05:
        return "stuck"
    if steps >= max_steps:
        return "timeout"
    return "timeout"


@router.post("/{project_id}/replay", response_model=ReplayResponse)
def replay(project: ProjectDep, body: ReplayRequest, db: DbSession) -> ReplayResponse:
    build = db.scalars(
        select(Build).where(Build.project_id == project.id).order_by(Build.created_at.desc())
    ).first()
    if not build:
        raise HTTPException(404, "no build for this project — call POST /build first")

    if body.policy == "ppo":
        if body.policy_id:
            policy = db.get(Policy, body.policy_id)
        else:
            policy = db.scalars(
                select(Policy)
                .where(Policy.build_id == build.id, Policy.algo == "ppo")
                .order_by(Policy.created_at.desc())
            ).first()
        if not policy or not policy.ckpt_path or not Path(policy.ckpt_path).exists():
            raise HTTPException(400, "no trained PPO policy — train one first via POST /train")
        from stable_baselines3 import PPO
        sb3_model = PPO.load(policy.ckpt_path, device="cpu")
        def policy_fn(obs, _rng):
            action, _ = sb3_model.predict(obs, deterministic=True)
            return action
    else:
        policy_fn = _policy_greedy if body.policy == "greedy" else _policy_random
        policy = None

    env = NavEnv(build.mjcf_path, task=TaskConfig(max_steps=body.max_steps))
    rng = np.random.default_rng(body.seed)

    episodes: list[EpisodeOut] = []
    successes = 0
    for ep in range(body.episodes):
        obs, _ = env.reset(seed=body.seed + ep)
        ep_reward = 0.0
        ep_steps = 0
        collisions = 0
        traj: list[TrajectoryPoint] = []
        speeds: list[float] = []
        info: dict = {}
        spawn_xy = env._agent_xy().astype(float).tolist()
        goal_xy = env._goal_xy().astype(float).tolist()

        prev_xy = np.array(spawn_xy)
        for _ in range(body.max_steps):
            a = policy_fn(obs, rng)
            obs, r, term, trunc, info = env.step(a)
            ep_reward += r
            ep_steps += 1
            curr = env._agent_xy().astype(float)
            speed = float(np.linalg.norm(curr - prev_xy))
            prev_xy = curr
            speeds.append(speed)
            had_collision = bool(env._has_scene_collision())
            if had_collision:
                collisions += 1
            if body.include_trajectories:
                traj.append(TrajectoryPoint(step=ep_steps, x=float(curr[0]), y=float(curr[1]), collision=had_collision))
            if term or trunc:
                break

        success = bool(info.get("success", False))
        distance = float(info.get("distance", -1.0))
        last_speeds = speeds[-50:]
        failure_class = _classify(ep_steps, body.max_steps, distance, success, collisions, last_speeds)
        if success:
            successes += 1
        episodes.append(
            EpisodeOut(
                steps=ep_steps,
                reward=round(float(ep_reward), 3),
                distance=round(distance, 3),
                success=success,
                failure_class=failure_class,
                spawn=spawn_xy,
                goal=goal_xy,
                trajectory=traj if body.include_trajectories else None,
            )
        )
    env.close()

    avg_reward = float(np.mean([e.reward for e in episodes]))
    run = Run(
        policy_id=policy.id if policy else None,
        baseline=body.policy if body.policy in ("random", "greedy") else None,
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
        bounds=build.bounds,
        spawn_region=build.spawn_region,
        episodes=episodes,
    )
