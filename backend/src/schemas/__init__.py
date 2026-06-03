"""Pydantic request/response schemas. One module per feature."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ProjectOut(ORMModel):
    id: str
    name: str
    video_path: str | None
    thumbnail_path: str | None
    created_at: datetime


class ProjectCreate(BaseModel):
    name: str


class ReconstructionOut(ORMModel):
    id: str
    project_id: str
    backend: str
    params: dict
    mesh_path: str | None
    status: str
    error: str | None
    elapsed_s: float | None
    inngest_run_id: str | None
    created_at: datetime


class ReconstructionRequest(BaseModel):
    backend: str = "vggt"
    params: dict = {}


class BackendInfo(BaseModel):
    name: str
    implemented: bool
    requires_gpu: bool


class CheckResult(BaseModel):
    name: str
    status: str  # 'pass' | 'warn' | 'fail'
    message: str
    fix: str


class ValidationOut(ORMModel):
    id: str
    reconstruction_id: str
    report: dict | None
    user_override: bool
    status: str
    error: str | None
    created_at: datetime


class BuildOut(ORMModel):
    id: str
    project_id: str
    reconstruction_id: str | None
    mjcf_path: str | None
    n_hulls: int | None
    bounds: dict | None
    spawn_region: dict | None
    status: str
    error: str | None
    created_at: datetime


class BuildRequest(BaseModel):
    reconstruction_id: str | None = None
    target_diagonal_m: float | None = None
    max_hulls: int | None = None
    up_axis: str = "auto"


class Goal3D(BaseModel):
    x: float
    y: float
    z: float
    radius: float = 0.3


class TaskCreate(BaseModel):
    name: str | None = None
    objective_nl: str = ""
    env_nl: str = ""
    agent_nl: str = ""
    goal_3d: dict | None = None


class TaskPatch(BaseModel):
    name: str | None = None
    objective_nl: str | None = None
    env_nl: str | None = None
    agent_nl: str | None = None
    goal_3d: dict | None = None


class TaskOut(ORMModel):
    id: str
    project_id: str
    build_id: str
    name: str
    objective_nl: str
    env_nl: str
    agent_nl: str
    goal_3d: dict | None
    status: str
    error: str | None
    codegen_model: str | None
    current_version_id: str | None
    current_code: str | None
    created_at: datetime


class PolicyOut(ORMModel):
    id: str
    build_id: str
    task_version_id: str | None
    algo: str
    ckpt_path: str
    total_steps: int
    metrics: dict
    created_at: datetime


class TrainRequest(BaseModel):
    total_steps: int = 100_000
    n_envs: int = 4
    seed: int = 0


class RunOut(ORMModel):
    id: str
    policy_id: str | None
    baseline: str | None
    status: str
    error: str | None
    episodes: int | None
    successes: int | None
    avg_reward: float | None
    created_at: datetime
