"""task + task_version tables; policy.task_version_id FK.

Revision ID: 0004_task
Revises: 0003_build_nullable
Create Date: 2026-06-02
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004_task"
down_revision = "0003_build_nullable"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "task",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("project_id", sa.String(), sa.ForeignKey("project.id", ondelete="CASCADE")),
        sa.Column("build_id", sa.String(), sa.ForeignKey("build.id", ondelete="CASCADE")),
        sa.Column("name", sa.String(), server_default="Task"),
        sa.Column("objective_nl", sa.Text(), server_default=""),
        sa.Column("env_nl", sa.Text(), server_default=""),
        sa.Column("agent_nl", sa.Text(), server_default=""),
        sa.Column("goal_3d", postgresql.JSONB()),
        sa.Column("status", sa.String(), server_default="drafting"),
        sa.Column("error", sa.Text()),
        sa.Column("codegen_model", sa.String()),
        sa.Column("codegen_prompt", sa.Text()),
        sa.Column("current_version_id", sa.String()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_task_project_created", "task", ["project_id", sa.text("created_at DESC")])

    op.create_table(
        "task_version",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("task_id", sa.String(), sa.ForeignKey("task.id", ondelete="CASCADE")),
        sa.Column("code", sa.Text(), server_default=""),
        sa.Column("created_by", sa.String(), server_default="user"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_task_version_task_created", "task_version", ["task_id", sa.text("created_at DESC")])

    op.create_foreign_key(
        "fk_task_current_version",
        "task",
        "task_version",
        ["current_version_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.add_column("policy", sa.Column("task_version_id", sa.String()))
    op.create_foreign_key(
        "fk_policy_task_version",
        "policy",
        "task_version",
        ["task_version_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_policy_task_version", "policy", type_="foreignkey")
    op.drop_column("policy", "task_version_id")
    op.drop_constraint("fk_task_current_version", "task", type_="foreignkey")
    op.drop_index("ix_task_version_task_created")
    op.drop_table("task_version")
    op.drop_index("ix_task_project_created")
    op.drop_table("task")
