"""Task CRUD helpers."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models import Build, Task, TaskVersion


def latest_ok_build(db: Session, project_id: str) -> Build | None:
    return db.scalars(
        select(Build)
        .where(Build.project_id == project_id, Build.status == "ok")
        .order_by(Build.created_at.desc())
    ).first()


def get_task_for_project(db: Session, project_id: str, task_id: str) -> Task | None:
    return db.scalars(
        select(Task).where(Task.project_id == project_id, Task.id == task_id)
    ).first()


def list_tasks(db: Session, project_id: str) -> list[Task]:
    return list(
        db.scalars(
            select(Task).where(Task.project_id == project_id).order_by(Task.created_at.desc())
        )
    )


def latest_task_for_build(db: Session, build_id: str) -> Task | None:
    return db.scalars(
        select(Task).where(Task.build_id == build_id).order_by(Task.created_at.desc())
    ).first()
