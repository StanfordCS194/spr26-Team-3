"""All SQLAlchemy declarative models, imported here so Alembic autogenerate
can discover them via `from src.models import *`.
"""
from __future__ import annotations

from datetime import datetime

from nanoid import generate as nanoid
from sqlalchemy import Boolean, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from src.db import Base


def _id() -> str:
    return nanoid(size=12)


class Project(Base):
    __tablename__ = "project"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_id)
    name: Mapped[str] = mapped_column(String, nullable=False)
    video_path: Mapped[str | None] = mapped_column(String)
    thumbnail_path: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    reconstructions: Mapped[list[Reconstruction]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    builds: Mapped[list[Build]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class Reconstruction(Base):
    __tablename__ = "reconstruction"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("project.id", ondelete="CASCADE"))
    backend: Mapped[str] = mapped_column(String, nullable=False)
    params: Mapped[dict] = mapped_column(JSONB, default=dict)
    mesh_path: Mapped[str | None] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="pending")
    error: Mapped[str | None] = mapped_column(String)
    elapsed_s: Mapped[float | None] = mapped_column(Float)
    inngest_run_id: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    project: Mapped[Project] = relationship(back_populates="reconstructions")
    validations: Mapped[list[Validation]] = relationship(
        back_populates="reconstruction", cascade="all, delete-orphan"
    )


class Validation(Base):
    __tablename__ = "validation"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_id)
    reconstruction_id: Mapped[str] = mapped_column(
        ForeignKey("reconstruction.id", ondelete="CASCADE")
    )
    report: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    user_override: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String, default="pending")
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    inngest_run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    reconstruction: Mapped[Reconstruction] = relationship(back_populates="validations")


class Build(Base):
    __tablename__ = "build"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("project.id", ondelete="CASCADE"))
    reconstruction_id: Mapped[str | None] = mapped_column(ForeignKey("reconstruction.id"))
    mjcf_path: Mapped[str | None] = mapped_column(String, nullable=True)
    n_hulls: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bounds: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    spawn_region: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending")
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    inngest_run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    project: Mapped[Project] = relationship(back_populates="builds")
    policies: Mapped[list[Policy]] = relationship(
        back_populates="build", cascade="all, delete-orphan"
    )


class Policy(Base):
    __tablename__ = "policy"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_id)
    build_id: Mapped[str] = mapped_column(ForeignKey("build.id", ondelete="CASCADE"))
    algo: Mapped[str] = mapped_column(String, default="ppo")
    ckpt_path: Mapped[str] = mapped_column(String)
    total_steps: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)
    inngest_run_id: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    build: Mapped[Build] = relationship(back_populates="policies")
    runs: Mapped[list[Run]] = relationship(back_populates="policy", cascade="all, delete-orphan")


class Run(Base):
    __tablename__ = "run"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_id)
    policy_id: Mapped[str | None] = mapped_column(ForeignKey("policy.id", ondelete="CASCADE"))
    baseline: Mapped[str | None] = mapped_column(String)
    episodes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    successes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    avg_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    trajectories_path: Mapped[str | None] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="pending")
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    inngest_run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    policy: Mapped[Policy | None] = relationship(back_populates="runs")
