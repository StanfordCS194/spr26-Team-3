"""PPO training + evaluation for NavEnv.

Minimum-viable RL: 4 vectorized envs, PPO with library defaults, MLP policy
over the (agent_xy, goal_xy, goal_vec, dist, lidar) observation. Saves a
.zip checkpoint that `play_policy` (and `python -m rl_env play`) can load.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_env.env import NavEnv, TaskConfig


def _make_env_fn(mjcf_path: str, max_steps: int, seed: int):
    def _thunk():
        env = NavEnv(mjcf_path, task=TaskConfig(max_steps=max_steps, seed=seed))
        return env

    return _thunk


def train_ppo(
    mjcf_path: str | Path,
    total_timesteps: int = 200_000,
    n_envs: int = 4,
    max_steps: int = 300,
    save_path: str | Path = "policy.zip",
    seed: int = 0,
    device: str = "auto",
    verbose: int = 1,
) -> Path:
    mjcf_path = str(mjcf_path)
    env_fns = [_make_env_fn(mjcf_path, max_steps, seed + i) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=verbose,
        seed=seed,
        device=device,
    )

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    elapsed = time.time() - t0

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    vec_env.close()

    if verbose:
        print(f"\ntrained {total_timesteps} steps in {elapsed:.1f}s "
              f"({total_timesteps / elapsed:.0f} steps/s); saved {save_path}")
    return save_path


def play_policy(
    mjcf_path: str | Path,
    ckpt_path: str | Path,
    episodes: int = 10,
    max_steps: int = 500,
    seed: int = 0,
    deterministic: bool = True,
) -> dict:
    env = NavEnv(str(mjcf_path), task=TaskConfig(max_steps=max_steps))
    model = PPO.load(str(ckpt_path), device="auto")

    eps: list[dict] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_steps = 0
        info: dict = {}
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(action)
            ep_reward += float(r)
            ep_steps += 1
            if term or trunc:
                break
        eps.append(
            {
                "steps": ep_steps,
                "reward": float(round(ep_reward, 3)),
                "distance": float(round(info.get("distance", -1), 3)),
                "success": bool(info.get("success", False)),
            }
        )
    env.close()

    successes = sum(int(e["success"]) for e in eps)
    avg_reward = float(np.mean([e["reward"] for e in eps]))
    return {
        "episodes": eps,
        "successes": successes,
        "n_episodes": len(eps),
        "avg_reward": round(avg_reward, 3),
        "avg_steps": float(round(np.mean([e["steps"] for e in eps]), 1)),
    }
