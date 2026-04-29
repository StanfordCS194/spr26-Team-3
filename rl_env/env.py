"""Gymnasium-compatible navigation env over a MuJoCo scene built by `rl_env.build`.

Task (PRD Feature 2.4): spawn at a random free-space point, reach a random goal,
penalize collisions, terminate on success or timeout.

Action space: continuous (vx, vy) in [-1, 1] -> scaled to ctrlrange.
Observation: agent_xy, goal_xy, goal_vec, dist_to_goal, lidar (16 rays).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


@dataclass
class TaskConfig:
    success_radius: float = 0.4
    max_steps: int = 500
    collision_penalty: float = 0.05
    success_bonus: float = 10.0
    step_penalty: float = 0.001
    reward_scale: float = 0.1
    n_lidar: int = 16
    lidar_max: float = 6.0
    spawn_region: tuple[float, float, float, float] | None = None
    """xmin, xmax, ymin, ymax. If None, read from MJCF agent body / model bounds."""
    seed: int | None = None


class NavEnv(gym.Env):
    """Top-down navigation in a WorldScan-built MuJoCo scene.

    Construct with the path to an MJCF produced by `rl_env.build.build_environment`.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(
        self,
        mjcf_path: str | Path,
        task: TaskConfig | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.mjcf_path = str(mjcf_path)
        self.task = task or TaskConfig()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = mujoco.MjData(self.model)

        self._agent_x = self.model.joint("agent_x").id
        self._agent_y = self.model.joint("agent_y").id
        self._agent_x_qpos = self.model.jnt_qposadr[self._agent_x]
        self._agent_y_qpos = self.model.jnt_qposadr[self._agent_y]
        self._agent_x_qvel = self.model.jnt_dofadr[self._agent_x]
        self._agent_y_qvel = self.model.jnt_dofadr[self._agent_y]
        self._goal_site = self.model.site("goal").id
        self._agent_site = self.model.site("agent_site").id
        self._agent_geom = self.model.geom("agent_geom").id

        if self.task.spawn_region is None:
            self.task.spawn_region = self._load_spawn_region_from_sidecar() or self._infer_spawn_region()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        obs_dim = 2 + 2 + 2 + 1 + self.task.n_lidar
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._step_count = 0
        self._renderer: mujoco.Renderer | None = None

        self._lidar_dirs = self._build_lidar_dirs(self.task.n_lidar)

        self._scene_geom_ids = self._collect_scene_geoms()

        if self.task.seed is not None:
            self.reset(seed=self.task.seed)

    # ------------------------------------------------------------------ helpers

    def _load_spawn_region_from_sidecar(self) -> tuple[float, float, float, float] | None:
        meta_path = Path(self.mjcf_path).with_name("metadata.json")
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        sr = meta.get("spawn_region")
        return tuple(sr) if sr else None

    def _infer_spawn_region(self) -> tuple[float, float, float, float]:
        floor_id = self.model.geom("floor").id
        size = self.model.geom_size[floor_id]
        pos = self.model.geom_pos[floor_id]
        margin = 0.5
        return (
            float(pos[0] - size[0] + margin),
            float(pos[0] + size[0] - margin),
            float(pos[1] - size[1] + margin),
            float(pos[1] + size[1] - margin),
        )

    def _collect_scene_geoms(self) -> set[int]:
        ids: set[int] = set()
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and name.startswith("hull_"):
                ids.add(i)
        return ids

    def _build_lidar_dirs(self, n: int) -> np.ndarray:
        thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([np.cos(thetas), np.sin(thetas), np.zeros_like(thetas)], axis=1)

    # ---------------------------------------------------------- gym API

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        xmin, xmax, ymin, ymax = self.task.spawn_region

        for _ in range(64):
            agent = self.np_random.uniform([xmin, ymin], [xmax, ymax])
            goal = self.np_random.uniform([xmin, ymin], [xmax, ymax])
            if np.linalg.norm(agent - goal) > 1.5:
                break

        self.data.qpos[self._agent_x_qpos] = agent[0]
        self.data.qpos[self._agent_y_qpos] = agent[1]
        self.data.qvel[self._agent_x_qvel] = 0.0
        self.data.qvel[self._agent_y_qvel] = 0.0

        self.model.site_pos[self._goal_site, 0] = goal[0]
        self.model.site_pos[self._goal_site, 1] = goal[1]

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0

        return self._observation(), self._info()

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        ctrl_range = self.model.actuator_ctrlrange
        scaled = action * ctrl_range[:, 1]
        self.data.ctrl[:] = scaled

        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._observation()
        dist = float(np.linalg.norm(self._agent_xy() - self._goal_xy()))
        terminated = dist < self.task.success_radius
        truncated = self._step_count >= self.task.max_steps

        reward = -self.task.step_penalty - dist * self.task.reward_scale
        if self._has_scene_collision():
            reward -= self.task.collision_penalty
        if terminated:
            reward += self.task.success_bonus

        info = self._info()
        info["distance"] = dist
        info["success"] = bool(terminated)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=-1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------- observations

    def _agent_xy(self) -> np.ndarray:
        return np.array(
            [
                self.data.qpos[self._agent_x_qpos],
                self.data.qpos[self._agent_y_qpos],
            ],
            dtype=np.float32,
        )

    def _goal_xy(self) -> np.ndarray:
        return self.model.site_pos[self._goal_site, :2].astype(np.float32)

    def _lidar(self) -> np.ndarray:
        origin = self.data.site_xpos[self._agent_site].copy()
        ranges = np.full(self.task.n_lidar, self.task.lidar_max, dtype=np.float32)
        geomid_out = np.zeros(1, dtype=np.int32)
        for i, d in enumerate(self._lidar_dirs):
            dist = mujoco.mj_ray(
                self.model,
                self.data,
                origin,
                d.astype(np.float64),
                None,
                1,
                self._agent_geom,
                geomid_out,
            )
            if dist >= 0:
                ranges[i] = min(float(dist), self.task.lidar_max)
        return ranges

    def _observation(self) -> np.ndarray:
        a = self._agent_xy()
        g = self._goal_xy()
        v = g - a
        d = np.linalg.norm(v)
        lidar = self._lidar()
        return np.concatenate([a, g, v, [d], lidar]).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        return {"step": self._step_count}

    def _has_scene_collision(self) -> bool:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == self._agent_geom and g2 in self._scene_geom_ids:
                return True
            if g2 == self._agent_geom and g1 in self._scene_geom_ids:
                return True
        return False
