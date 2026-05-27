"""Training worker: PPO loop chunked into ~5 progress snapshots."""
from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_env.env import NavEnv, TaskConfig
from src.config import get_settings
from src.db import SessionLocal
from src.models import Policy


def _make_env_fn(mjcf_path: str, max_steps: int, seed: int):
    def _t():
        return NavEnv(mjcf_path, task=TaskConfig(max_steps=max_steps, seed=seed))
    return _t


class ProgressCB(BaseCallback):
    """Snapshots metrics into the DB every N steps."""

    def __init__(self, policy_id: str, total: int, period: int = 2000):
        super().__init__()
        self.policy_id = policy_id
        self.total = total
        self.period = period
        self.last = 0
        self.start = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last < self.period:
            return True
        self.last = self.num_timesteps
        elapsed = time.time() - self.start
        ep_info = self.locals.get("infos", [{}])[0] if self.locals.get("infos") else {}
        recent_returns = [e.get("r", 0.0) for e in self.model.ep_info_buffer or []]
        avg_r = float(np.mean(recent_returns)) if recent_returns else 0.0
        with SessionLocal() as db:
            p = db.get(Policy, self.policy_id)
            if p is None:
                return False
            p.metrics = {
                **(p.metrics or {}),
                "progress": min(1.0, self.num_timesteps / max(self.total, 1)),
                "steps": int(self.num_timesteps),
                "avg_reward": round(avg_r, 3),
                "fps": int(self.num_timesteps / max(elapsed, 1e-6)),
                "elapsed_s": round(elapsed, 2),
                "trace": (p.metrics or {}).get("trace", []) + [
                    {"step": int(self.num_timesteps), "reward": round(avg_r, 3)}
                ],
            }
            db.commit()
        return True


def run_training(policy_id: str, mjcf_path: str, total_steps: int, n_envs: int, max_steps: int, seed: int) -> None:
    try:
        with SessionLocal() as db:
            p = db.get(Policy, policy_id)
            assert p is not None
            p.metrics = {"progress": 0.0, "steps": 0, "started_at": time.time()}
            db.commit()

        vec = DummyVecEnv([_make_env_fn(mjcf_path, max_steps, seed + i) for i in range(n_envs)])
        model = PPO(
            "MlpPolicy",
            vec,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
            seed=seed,
            device="cpu",
        )
        cb = ProgressCB(policy_id, total_steps, period=max(2000, total_steps // 20))
        t0 = time.time()
        model.learn(total_timesteps=total_steps, callback=cb, progress_bar=False)
        elapsed = time.time() - t0

        ckpt_path = Path(mjcf_path).parent / f"policy_{policy_id}.zip"
        model.save(ckpt_path)
        vec.close()

        with SessionLocal() as db:
            p = db.get(Policy, policy_id)
            assert p is not None
            p.ckpt_path = str(ckpt_path)
            p.metrics = {
                **(p.metrics or {}),
                "progress": 1.0,
                "elapsed_s": round(elapsed, 2),
                "done": True,
            }
            db.commit()
    except Exception as exc:
        traceback.print_exc()
        with SessionLocal() as db:
            p = db.get(Policy, policy_id)
            if p is not None:
                p.metrics = {**(p.metrics or {}), "error": str(exc), "done": True}
                db.commit()


def start_training(policy_id: str, mjcf_path: str, total_steps: int, n_envs: int, max_steps: int, seed: int) -> threading.Thread:
    t = threading.Thread(
        target=run_training,
        args=(policy_id, mjcf_path, total_steps, n_envs, max_steps, seed),
        daemon=True,
        name=f"train-{policy_id}",
    )
    t.start()
    return t
