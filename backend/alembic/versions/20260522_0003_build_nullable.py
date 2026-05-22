"""Relax build.mjcf_path to nullable.

The Build row is now written as `pending` from the FastAPI route and the
mjcf_path is filled in later by the Inngest function. The initial migration
left mjcf_path as NOT NULL, so every Build env click returned a 500.

Revision ID: 0003_build_nullable
Revises: 0002_status
Create Date: 2026-05-22
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_build_nullable"
down_revision = "0002_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("build", "mjcf_path", existing_type=sa.String(), nullable=True)


def downgrade() -> None:
    op.alter_column("build", "mjcf_path", existing_type=sa.String(), nullable=False)
