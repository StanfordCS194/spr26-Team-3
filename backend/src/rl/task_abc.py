"""Abstract base for user-authored RL tasks (natural language → Python → PPO).

Generated modules must subclass `Task` and implement the hooks below. Physics
stepping and MuJoCo integration stay in `TaskEnv` (PR-4); tasks only define
reward, termination, observations, and spawn/goal sampling.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from numpy.random import Generator


@dataclass(frozen=True)
class TaskContext:
    """Scene metadata injected into codegen prompts and task construction."""

    mjcf_path: str
    bounds: dict[str, list[float]] | None
    spawn_region: dict[str, float] | None
    goal_3d: dict[str, float] | None


class Task(ABC):
    """Contract for a runnable task bound to one WorldScan build."""

    obs_space: gym.Space
    action_space: gym.Space
    horizon: int

    def __init__(self, ctx: TaskContext) -> None:
        self.ctx = ctx

    @abstractmethod
    def reset(self, mj_data: mujoco.MjData, rng: Generator) -> np.ndarray | dict[str, Any]:
        """Sample spawn/goal (if needed) and return the initial observation."""

    @abstractmethod
    def observe(self, mj_data: mujoco.MjData) -> np.ndarray | dict[str, Any]:
        """Build observation from current simulation state."""

    @abstractmethod
    def reward(self, mj_data: mujoco.MjData, action: np.ndarray) -> float:
        ...

    @abstractmethod
    def terminated(self, mj_data: mujoco.MjData) -> bool:
        """Episode success or other terminal condition (not timeout)."""

    @abstractmethod
    def truncated(self, mj_data: mujoco.MjData, step: int) -> bool:
        """Horizon / timeout."""

    def validate(self, mjcf_path: str) -> list[str]:
        """Optional pre-flight warnings before training. Empty list = OK."""
        return []


def nav_obs_space(n_lidar: int = 16) -> gym.Space:
    """Observation layout matching the legacy NavEnv (agent, goal, vec, dist, lidar)."""
    dim = 2 + 2 + 2 + 1 + n_lidar
    return spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def nav_action_space() -> gym.Space:
    return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
