"""Task authoring CRUD — natural-language fields persisted per build.

Codegen (PR-2) and the Task UI (PR-3) build on these routes. Training still
uses the legacy NavEnv until PR-4.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from nanoid import generate as nanoid

from src.deps import DbSession, ProjectDep
from src.features.tasks import service
from src.models import Task
from src.schemas import TaskCreate, TaskOut, TaskPatch

router = APIRouter()


def _to_out(task: Task) -> TaskOut:
    code = task.current_version.code if task.current_version else None
    return TaskOut(
        id=task.id,
        project_id=task.project_id,
        build_id=task.build_id,
        name=task.name,
        objective_nl=task.objective_nl,
        env_nl=task.env_nl,
        agent_nl=task.agent_nl,
        goal_3d=task.goal_3d,
        status=task.status,
        error=task.error,
        codegen_model=task.codegen_model,
        current_version_id=task.current_version_id,
        current_code=code,
        created_at=task.created_at,
    )


@router.get("/{project_id}/tasks", response_model=list[TaskOut])
def list_tasks(project: ProjectDep, db: DbSession) -> list[TaskOut]:
    rows = service.list_tasks(db, project.id)
    return [_to_out(t) for t in rows]


@router.post("/{project_id}/tasks", response_model=TaskOut, status_code=201)
def create_task(project: ProjectDep, body: TaskCreate, db: DbSession) -> TaskOut:
    build = service.latest_ok_build(db, project.id)
    if not build or not build.mjcf_path:
        raise HTTPException(400, "no successful build — complete Build before authoring a task")

    existing = service.latest_task_for_build(db, build.id)
    if existing and existing.status in ("drafting", "generating", "ready"):
        raise HTTPException(
            409,
            f"task already exists for this build ({existing.id}); PATCH it instead",
        )

    task = Task(
        id=nanoid(size=12),
        project_id=project.id,
        build_id=build.id,
        name=body.name or "Task",
        objective_nl=body.objective_nl,
        env_nl=body.env_nl,
        agent_nl=body.agent_nl,
        goal_3d=body.goal_3d,
        status="drafting",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return _to_out(task)


@router.get("/{project_id}/tasks/{task_id}", response_model=TaskOut)
def get_task(project: ProjectDep, task_id: str, db: DbSession) -> TaskOut:
    task = service.get_task_for_project(db, project.id, task_id)
    if not task:
        raise HTTPException(404, f"unknown task {task_id}")
    return _to_out(task)


@router.patch("/{project_id}/tasks/{task_id}", response_model=TaskOut)
def patch_task(
    project: ProjectDep,
    task_id: str,
    body: TaskPatch,
    db: DbSession,
) -> TaskOut:
    task = service.get_task_for_project(db, project.id, task_id)
    if not task:
        raise HTTPException(404, f"unknown task {task_id}")
    if task.status == "generating":
        raise HTTPException(409, "task is generating — wait for completion")

    if body.name is not None:
        task.name = body.name
    if body.objective_nl is not None:
        task.objective_nl = body.objective_nl
    if body.env_nl is not None:
        task.env_nl = body.env_nl
    if body.agent_nl is not None:
        task.agent_nl = body.agent_nl
    if body.goal_3d is not None:
        task.goal_3d = body.goal_3d

    db.commit()
    db.refresh(task)
    return _to_out(task)
