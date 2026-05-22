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
    report: dict
    user_override: bool
    created_at: datetime


class BuildOut(ORMModel):
    id: str
    project_id: str
    reconstruction_id: str | None
    mjcf_path: str
    n_hulls: int
    bounds: dict
    spawn_region: dict
    created_at: datetime


class BuildRequest(BaseModel):
    reconstruction_id: str | None = None
    target_diagonal_m: float | None = None
    max_hulls: int | None = None
    up_axis: str = "auto"


class PolicyOut(ORMModel):
    id: str
    build_id: str
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
    episodes: int
    successes: int
    avg_reward: float
    created_at: datetime
