"""Read-only MuJoCo helpers for generated Task code (no direct model access required).

Injected into the codegen sandbox as the `mujoco_view` module.
"""
from __future__ import annotations

import mujoco
import numpy as np


def _ids(model: mujoco.MjModel) -> dict[str, int]:
    return {
        "agent_x": model.joint("agent_x").id,
        "agent_y": model.joint("agent_y").id,
        "goal_site": model.site("goal").id,
        "agent_site": model.site("agent_site").id,
        "agent_geom": model.geom("agent_geom").id,
    }


def agent_xy(model: mujoco.MjModel, mj_data: mujoco.MjData) -> np.ndarray:
    ids = _ids(model)
    return np.array(
        [
            mj_data.qpos[model.jnt_qposadr[ids["agent_x"]]],
            mj_data.qpos[model.jnt_qposadr[ids["agent_y"]]],
        ],
        dtype=np.float32,
    )


def goal_xy(model: mujoco.MjModel, mj_data: mujoco.MjData) -> np.ndarray:
    ids = _ids(model)
    return model.site_pos[ids["goal_site"], :2].astype(np.float32)


def set_agent_xy(model: mujoco.MjModel, mj_data: mujoco.MjData, xy: np.ndarray) -> None:
    ids = _ids(model)
    mj_data.qpos[model.jnt_qposadr[ids["agent_x"]]] = float(xy[0])
    mj_data.qpos[model.jnt_qposadr[ids["agent_y"]]] = float(xy[1])
    mj_data.qvel[model.jnt_dofadr[ids["agent_x"]]] = 0.0
    mj_data.qvel[model.jnt_dofadr[ids["agent_y"]]] = 0.0


def set_goal_xy(model: mujoco.MjModel, mj_data: mujoco.MjData, xy: np.ndarray) -> None:
    ids = _ids(model)
    model.site_pos[ids["goal_site"], 0] = float(xy[0])
    model.site_pos[ids["goal_site"], 1] = float(xy[1])


def dist_to_goal(model: mujoco.MjModel, mj_data: mujoco.MjData) -> float:
    return float(np.linalg.norm(agent_xy(model, mj_data) - goal_xy(model, mj_data)))


def scene_collision(model: mujoco.MjModel, mj_data: mujoco.MjData) -> bool:
    ids = _ids(model)
    agent_geom = ids["agent_geom"]
    scene_geoms = {
        i
        for i in range(model.ngeom)
        if (n := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)) and n.startswith("hull_")
    }
    for i in range(mj_data.ncon):
        c = mj_data.contact[i]
        if c.geom1 == agent_geom and c.geom2 in scene_geoms:
            return True
        if c.geom2 == agent_geom and c.geom1 in scene_geoms:
            return True
    return False


def lidar(
    model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    n_rays: int = 16,
    max_range: float = 6.0,
) -> np.ndarray:
    ids = _ids(model)
    origin = mj_data.site_xpos[ids["agent_site"]].copy()
    thetas = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    dirs = np.stack([np.cos(thetas), np.sin(thetas), np.zeros_like(thetas)], axis=1)
    ranges = np.full(n_rays, max_range, dtype=np.float32)
    geomid_out = np.zeros(1, dtype=np.int32)
    for i, d in enumerate(dirs):
        dist = mujoco.mj_ray(
            model,
            mj_data,
            origin,
            d.astype(np.float64),
            None,
            1,
            ids["agent_geom"],
            geomid_out,
        )
        if dist >= 0:
            ranges[i] = min(float(dist), max_range)
    return ranges


def forward(model: mujoco.MjModel, mj_data: mujoco.MjData) -> None:
    mujoco.mj_forward(model, mj_data)


def step_zero_ctrl(model: mujoco.MjModel, mj_data: mujoco.MjData) -> None:
    mj_data.ctrl[:] = 0.0
    mujoco.mj_step(model, mj_data)
