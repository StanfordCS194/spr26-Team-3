"""Sandbox loader + dry-run (PR-2). No API key required."""
from __future__ import annotations

from pathlib import Path

import pytest

from rl_env.build import BuildConfig, build_environment
from rl_env.sample_room import make_sample_room
from src.rl.legacy_tasks import NavToGoalTask
from src.rl.task_abc import TaskContext
from src.rl.task_runtime import TaskRuntimeError, dry_run, extract_python_module, load

MINIMAL_GENERATED = """
class GeneratedTask(Task):
    obs_space = nav_obs_space()
    action_space = nav_action_space()
    horizon = 100

    def __init__(self, ctx):
        Task.__init__(self, ctx)
        self._model = None
        self._step = 0

    def reset(self, mj_data, rng):
        self._step = 0
        return self.observe(mj_data)

    def observe(self, mj_data):
        return np.zeros(int(self.obs_space.shape[0]), dtype=np.float32)

    def reward(self, mj_data, action):
        return -0.001

    def terminated(self, mj_data):
        return False

    def truncated(self, mj_data, step):
        return step >= 100
"""

NAV_GOAL_SOURCE = """
class GeneratedTask(Task):
    obs_space = nav_obs_space()
    action_space = nav_action_space()
    horizon = 300

    def __init__(self, ctx):
        Task.__init__(self, ctx)
        self._model = None
        self._step = 0
        self._success_r = 0.4

    def reset(self, mj_data, rng):
        self._step = 0
        sr = self.ctx.spawn_region or {"xmin": 0.5, "xmax": 3.5, "ymin": 0.5, "ymax": 3.5}
        agent = rng.uniform([sr["xmin"], sr["ymin"]], [sr["xmax"], sr["ymax"]])
        goal = rng.uniform([sr["xmin"], sr["ymin"]], [sr["xmax"], sr["ymax"]])
        if self.ctx.goal_3d:
            goal = np.array([self.ctx.goal_3d["x"], self.ctx.goal_3d["y"]], dtype=np.float32)
        mujoco_view.set_agent_xy(self._model, mj_data, agent)
        mujoco_view.set_goal_xy(self._model, mj_data, goal)
        mujoco_view.forward(self._model, mj_data)
        return self.observe(mj_data)

    def observe(self, mj_data):
        a = mujoco_view.agent_xy(self._model, mj_data)
        g = mujoco_view.goal_xy(self._model, mj_data)
        v = g - a
        return np.concatenate([a, g, v, [np.linalg.norm(v)], mujoco_view.lidar(self._model, mj_data)]).astype(np.float32)

    def reward(self, mj_data, action):
        d = mujoco_view.dist_to_goal(self._model, mj_data)
        r = -0.001 - 0.1 * d
        if mujoco_view.scene_collision(self._model, mj_data):
            r -= 0.05
        return float(r)

    def terminated(self, mj_data):
        return mujoco_view.dist_to_goal(self._model, mj_data) < self._success_r

    def truncated(self, mj_data, step):
        return step >= self.horizon
"""


def _fixture_mjcf(tmp_path: Path) -> str:
    mesh = make_sample_room(size=(4.0, 3.0, 3.0), seed=7)
    mesh_path = tmp_path / "mesh.ply"
    mesh.export(str(mesh_path))
    artifacts = build_environment(
        BuildConfig(mesh_path=str(mesh_path), out_dir=str(tmp_path / "build"), up_axis="y")
    )
    return str(artifacts.mjcf_path)


def test_extract_python_module_strips_fence() -> None:
    raw = "Here:\n```python\nx = 1\n```\n"
    assert extract_python_module(raw) == "x = 1"


def test_load_rejects_import() -> None:
    with pytest.raises(TaskRuntimeError, match="import"):
        load("import os\nclass GeneratedTask(Task):\n  pass")


def test_load_requires_generated_task() -> None:
    with pytest.raises(TaskRuntimeError, match="GeneratedTask"):
        load("x = 1")


def test_dry_run_minimal(tmp_path: Path) -> None:
    mjcf = _fixture_mjcf(tmp_path)
    ctx = TaskContext(mjcf_path=mjcf, bounds=None, spawn_region=None, goal_3d=None)
    dry_run(MINIMAL_GENERATED, mjcf, ctx, steps=5)


def test_dry_run_nav_style(tmp_path: Path) -> None:
    mjcf = _fixture_mjcf(tmp_path)
    ctx = TaskContext(
        mjcf_path=mjcf,
        bounds={"min": [0, 0, 0], "max": [4, 4, 2]},
        spawn_region={"xmin": 0.5, "xmax": 3.5, "ymin": 0.5, "ymax": 3.5},
        goal_3d=None,
    )
    dry_run(NAV_GOAL_SOURCE, mjcf, ctx, steps=10)


def test_legacy_nav_task_instantiates(tmp_path: Path) -> None:
    import mujoco
    import numpy as np

    mjcf = _fixture_mjcf(tmp_path)
    ctx = TaskContext(mjcf_path=mjcf, bounds=None, spawn_region=None, goal_3d=None)
    model = mujoco.MjModel.from_xml_path(mjcf)
    mj_data = mujoco.MjData(model)
    task = NavToGoalTask(ctx)
    task._model = model
    obs = task.reset(mj_data, np.random.default_rng(0))
    assert obs.shape == task.obs_space.shape
