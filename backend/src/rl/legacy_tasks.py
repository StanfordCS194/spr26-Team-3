"""Reference task implementations — included in LLM prompts during codegen (PR-2)."""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.random import Generator

from src.rl import mujoco_view
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
        self._model: mujoco.MjModel | None = None
        self._step = 0

    def _spawn_region(self) -> tuple[float, float, float, float]:
        sr = self.ctx.spawn_region
        if sr:
            return (sr["xmin"], sr["xmax"], sr["ymin"], sr["ymax"])
        return (0.5, 3.5, 0.5, 3.5)

    def _m(self, mj_data: mujoco.MjData) -> mujoco.MjModel:
        if self._model is None:
            raise RuntimeError("task._model not set — runtime must inject before reset/step")
        return self._model

    def reset(self, mj_data: mujoco.MjData, rng: Generator) -> np.ndarray:
        self._step = 0
        xmin, xmax, ymin, ymax = self._spawn_region()
        agent = rng.uniform([xmin, ymin], [xmax, ymax])
        goal = rng.uniform([xmin, ymin], [xmax, ymax])
        for _ in range(64):
            agent = rng.uniform([xmin, ymin], [xmax, ymax])
            goal = rng.uniform([xmin, ymin], [xmax, ymax])
            if float(np.linalg.norm(agent - goal)) >= self.cfg.min_start_goal_dist:
                break
        if self.ctx.goal_3d:
            goal = np.array([self.ctx.goal_3d["x"], self.ctx.goal_3d["y"]], dtype=np.float32)

        model = self._m(mj_data)
        mujoco_view.set_agent_xy(model, mj_data, agent)
        mujoco_view.set_goal_xy(model, mj_data, goal)
        mujoco_view.forward(model, mj_data)
        return self.observe(mj_data)

    def observe(self, mj_data: mujoco.MjData) -> np.ndarray:
        model = self._m(mj_data)
        a = mujoco_view.agent_xy(model, mj_data)
        g = mujoco_view.goal_xy(model, mj_data)
        v = g - a
        d = np.linalg.norm(v)
        lidar = mujoco_view.lidar(model, mj_data, self.cfg.n_lidar, self.cfg.lidar_max)
        return np.concatenate([a, g, v, [d], lidar]).astype(np.float32)

    def reward(self, mj_data: mujoco.MjData, action: np.ndarray) -> float:
        model = self._m(mj_data)
        dist = mujoco_view.dist_to_goal(model, mj_data)
        r = -self.cfg.step_penalty - dist * self.cfg.reward_scale
        if mujoco_view.scene_collision(model, mj_data):
            r -= self.cfg.collision_penalty
        if dist < self.cfg.success_radius:
            r += self.cfg.success_bonus
        return float(r)

    def terminated(self, mj_data: mujoco.MjData) -> bool:
        return mujoco_view.dist_to_goal(self._m(mj_data), mj_data) < self.cfg.success_radius

    def truncated(self, mj_data: mujoco.MjData, step: int) -> bool:
        return step >= self.cfg.max_steps
