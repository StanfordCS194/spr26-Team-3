"""Task CRUD helpers."""
from __future__ import annotations

import traceback

from nanoid import generate as nanoid
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.features.tasks.codegen import generate_module_code
from src.models import Build, Task, TaskVersion
from src.rl.task_abc import TaskContext
from src.rl.task_runtime import TaskRuntimeError, validate_and_dry_run


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


def run_task_codegen(db: Session, task_id: str) -> str:
    """Codegen + validate + persist task_version. Returns new version id."""
    task = db.get(Task, task_id)
    if task is None:
        raise RuntimeError(f"task {task_id} missing")
    task.status = "generating"
    task.error = None
    db.commit()

    build = db.get(Build, task.build_id)
    if build is None or not build.mjcf_path:
        raise RuntimeError("task build has no MJCF")

    ctx = TaskContext(
        mjcf_path=build.mjcf_path,
        bounds=build.bounds,
        spawn_region=build.spawn_region,
        goal_3d=task.goal_3d,
    )
    try:
        code, _raw, model_id = generate_module_code(
            ctx, task.objective_nl, task.env_nl, task.agent_nl
        )
        validate_and_dry_run(code, build.mjcf_path, ctx)
    except TaskRuntimeError as e:
        task = db.get(Task, task_id)
        assert task is not None
        task.status = "failed"
        task.error = str(e)[:1000]
        db.commit()
        raise
    except Exception:
        task = db.get(Task, task_id)
        assert task is not None
        task.status = "failed"
        task.error = traceback.format_exc()[-1000:]
        db.commit()
        raise

    version_id = nanoid(size=12)
    task = db.get(Task, task_id)
    assert task is not None
    db.add(TaskVersion(id=version_id, task_id=task_id, code=code, created_by="ai"))
    db.flush()
    task.current_version_id = version_id
    task.codegen_model = model_id
    task.status = "ready"
    task.error = None
    db.commit()
    return version_id
