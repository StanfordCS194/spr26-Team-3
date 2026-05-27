"""Add status + error columns to validation, build, run.

So the Inngest functions for these stages can mark pending → ok/failed
the same way reconstruction already does.

Revision ID: 0002_status
Revises: 0001_init
Create Date: 2026-05-22
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_status"
down_revision = "0001_init"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table in ("validation", "build", "run"):
        op.add_column(
            table,
            sa.Column("status", sa.String(), nullable=False, server_default="ok"),
        )
        op.add_column(table, sa.Column("error", sa.String(), nullable=True))
        op.add_column(table, sa.Column("inngest_run_id", sa.String(), nullable=True))


def downgrade() -> None:
    for table in ("validation", "build", "run"):
        op.drop_column(table, "inngest_run_id")
        op.drop_column(table, "error")
        op.drop_column(table, "status")
