"""init — projects, reconstructions, validations, builds, policies, runs.

Revision ID: 0001_init
Revises:
Create Date: 2026-05-22

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "project",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("video_path", sa.String()),
        sa.Column("thumbnail_path", sa.String()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "reconstruction",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("project_id", sa.String(), sa.ForeignKey("project.id", ondelete="CASCADE")),
        sa.Column("backend", sa.String(), nullable=False),
        sa.Column("params", postgresql.JSONB(), server_default="{}"),
        sa.Column("mesh_path", sa.String()),
        sa.Column("status", sa.String(), server_default="pending"),
        sa.Column("error", sa.String()),
        sa.Column("elapsed_s", sa.Float()),
        sa.Column("inngest_run_id", sa.String()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "ix_reconstruction_project_created",
        "reconstruction",
        ["project_id", sa.text("created_at DESC")],
    )
    op.create_table(
        "validation",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "reconstruction_id",
            sa.String(),
            sa.ForeignKey("reconstruction.id", ondelete="CASCADE"),
        ),
        sa.Column("report", postgresql.JSONB()),
        sa.Column("user_override", sa.Boolean(), server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "build",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("project_id", sa.String(), sa.ForeignKey("project.id", ondelete="CASCADE")),
        sa.Column("reconstruction_id", sa.String(), sa.ForeignKey("reconstruction.id")),
        sa.Column("mjcf_path", sa.String(), nullable=False),
        sa.Column("n_hulls", sa.Integer()),
        sa.Column("bounds", postgresql.JSONB()),
        sa.Column("spawn_region", postgresql.JSONB()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "policy",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("build_id", sa.String(), sa.ForeignKey("build.id", ondelete="CASCADE")),
        sa.Column("algo", sa.String(), server_default="ppo"),
        sa.Column("ckpt_path", sa.String()),
        sa.Column("total_steps", sa.Integer()),
        sa.Column("metrics", postgresql.JSONB(), server_default="{}"),
        sa.Column("inngest_run_id", sa.String()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "run",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("policy_id", sa.String(), sa.ForeignKey("policy.id", ondelete="CASCADE")),
        sa.Column("baseline", sa.String()),
        sa.Column("episodes", sa.Integer()),
        sa.Column("successes", sa.Integer()),
        sa.Column("avg_reward", sa.Float()),
        sa.Column("trajectories_path", sa.String()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("run")
    op.drop_table("policy")
    op.drop_table("build")
    op.drop_table("validation")
    op.drop_index("ix_reconstruction_project_created")
    op.drop_table("reconstruction")
    op.drop_table("project")
