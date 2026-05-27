"""Inngest function for policy/baseline rollouts. Writes trajectories to
disk under data/projects/<id>/runs/<run_id>/trajectories.json and updates
the run row with the summary.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import inngest
import numpy as np

from rl_env.env import NavEnv, TaskConfig
from src.config import get_settings
from src.db import SessionLocal
from src.inngest_client import inngest_client, register
from src.models import Build, Policy, Run

log = logging.getLogger(__name__)


def _policy_random(obs, rng):
    return rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


def _policy_greedy(obs, rng):
    goal_vec = obs[4:6]
    norm = float(np.linalg.norm(goal_vec))
    if norm < 1e-6:
        return np.zeros(2, dtype=np.float32)
    return (goal_vec / norm).astype(np.float32)


def _classify(steps, max_steps, distance, success, collisions, last_speeds):
    if success:
        return "success"
    if collisions >= 3:
        return "collided"
    if distance < 1.0:
        return "near-miss"
    if last_speeds and max(last_speeds) < 0.05:
        return "stuck"
    return "timeout"


@register
@inngest_client.create_function(
    fn_id="run-replay",
    trigger=inngest.TriggerEvent(event="replay/requested"),
    retries=0,
)
async def run_replay(ctx: inngest.Context) -> dict:
    step = ctx.step
    payload = ctx.event.data or {}
    run_id: str = payload["run_id"]
    policy_name: str = payload.get("policy", "greedy")
    episodes: int = int(payload.get("episodes", 5))
    max_steps: int = int(payload.get("max_steps", 300))
    seed: int = int(payload.get("seed", 0))
    policy_id: str | None = payload.get("policy_id")

    log.info("run-replay: %s (%s, %d eps)", run_id, policy_name, episodes)

    async def _run() -> dict:
        settings = get_settings()
        with SessionLocal() as db:
            r = db.get(Run, run_id)
            if r is None:
                raise RuntimeError(f"run {run_id} disappeared")
            r.status = "running"
            db.commit()
            policy = db.get(Policy, policy_id) if policy_id else None
            if policy:
                build = db.get(Build, policy.build_id)
            else:
                # Latest build on the project — pulled via the event's project ref
                project_id = payload["project_id"]
                build = (
                    db.query(Build)
                    .filter(Build.project_id == project_id)
                    .order_by(Build.created_at.desc())
                    .first()
                )
            if build is None or not build.mjcf_path:
                raise RuntimeError("no built scene for this project")
            mjcf_path = build.mjcf_path
            bounds = build.bounds
            spawn_region = build.spawn_region

        if policy_name == "ppo":
            if not policy or not policy.ckpt_path or not Path(policy.ckpt_path).exists():
                raise RuntimeError("no trained PPO policy")
            from stable_baselines3 import PPO

            sb3 = PPO.load(policy.ckpt_path, device="cpu")

            def policy_fn(obs, _rng):
                action, _ = sb3.predict(obs, deterministic=True)
                return action

        else:
            policy_fn = _policy_greedy if policy_name == "greedy" else _policy_random

        env = NavEnv(mjcf_path, task=TaskConfig(max_steps=max_steps))
        rng = np.random.default_rng(seed)
        successes = 0
        rewards = []
        episodes_out = []
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            ep_reward = 0.0
            ep_steps = 0
            collisions = 0
            traj = []
            speeds = []
            info: dict = {}
            spawn_xy = env._agent_xy().astype(float).tolist()
            goal_xy = env._goal_xy().astype(float).tolist()
            prev_xy = np.array(spawn_xy)
            for _ in range(max_steps):
                a = policy_fn(obs, rng)
                obs, rew, term, trunc, info = env.step(a)
                ep_reward += rew
                ep_steps += 1
                curr = env._agent_xy().astype(float)
                speeds.append(float(np.linalg.norm(curr - prev_xy)))
                prev_xy = curr
                if env._has_scene_collision():
                    collisions += 1
                traj.append(
                    {
                        "step": ep_steps,
                        "x": float(curr[0]),
                        "y": float(curr[1]),
                        "collision": bool(env._has_scene_collision()),
                    }
                )
                if term or trunc:
                    break
            success = bool(info.get("success", False))
            distance = float(info.get("distance", -1.0))
            failure_class = _classify(ep_steps, max_steps, distance, success, collisions, speeds[-50:])
            if success:
                successes += 1
            rewards.append(ep_reward)
            episodes_out.append(
                {
                    "steps": ep_steps,
                    "reward": round(float(ep_reward), 3),
                    "distance": round(distance, 3),
                    "success": success,
                    "failure_class": failure_class,
                    "spawn": spawn_xy,
                    "goal": goal_xy,
                    "trajectory": traj,
                }
            )
        env.close()

        avg_reward = float(np.mean(rewards)) if rewards else 0.0

        # Persist trajectories JSON to disk; keep DB row small
        traj_dir = settings.data_dir / "projects" / build.project_id / "runs" / run_id
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_path = traj_dir / "trajectories.json"
        traj_path.write_text(
            json.dumps(
                {
                    "bounds": bounds,
                    "spawn_region": spawn_region,
                    "episodes": episodes_out,
                }
            )
        )

        with SessionLocal() as db:
            r = db.get(Run, run_id)
            assert r is not None
            r.episodes = episodes
            r.successes = successes
            r.avg_reward = round(avg_reward, 3)
            r.trajectories_path = str(traj_path)
            r.status = "ok"
            db.commit()
        return {"run_id": run_id, "successes": successes, "n_episodes": episodes}

    return await step.run("rollout", _run)
