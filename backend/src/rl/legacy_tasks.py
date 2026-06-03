"""Reference task implementations — included in LLM prompts during codegen (PR-2).

`NavToGoalTask` mirrors the hardcoded behavior in `rl_env.env.NavEnv` today.
Training still uses `NavEnv` directly until PR-4 wires `TaskEnv`.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.random import Generator

from src.rl.task_abc import Task, TaskContext, nav_action_space, nav_obs_space


@dataclass
class NavToGoalConfig:
    success_radius: float = 0.4
    max_steps: int = 500
    collision_penalty: float = 0.05
    success_bonus: float = 10.0
    step_penalty: float = 0.001
    reward_scale: float = 0.1
    n_lidar: int = 16
    lidar_max: float = 6.0
    min_start_goal_dist: float = 1.5


class NavToGoalTask(Task):
    """Point-to-goal navigation with lidar — the current production default."""

    obs_space = nav_obs_space()
    action_space = nav_action_space()
    horizon = 500

    def __init__(self, ctx: TaskContext, cfg: NavToGoalConfig | None = None) -> None:
        super().__init__(ctx)
        self.cfg = cfg or NavToGoalConfig()
        self.horizon = self.cfg.max_steps
        self.obs_space = nav_obs_space(self.cfg.n_lidar)
        self._step = 0

    def reset(self, mj_data: mujoco.MjData, rng: Generator) -> np.ndarray:
        self._step = 0
        return self.observe(mj_data)

    def observe(self, mj_data: mujoco.MjData) -> np.ndarray:
        # Placeholder — PR-4 TaskEnv will call into mujoco helpers shared with NavEnv.
        return np.zeros(int(self.obs_space.shape[0]), dtype=np.float32)

    def reward(self, mj_data: mujoco.MjData, action: np.ndarray) -> float:
        return -self.cfg.step_penalty

    def terminated(self, mj_data: mujoco.MjData) -> bool:
        return False

    def truncated(self, mj_data: mujoco.MjData, step: int) -> bool:
        return step >= self.cfg.max_steps
