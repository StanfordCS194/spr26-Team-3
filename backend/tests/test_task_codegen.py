"""Codegen pipeline with mocked Claude (PR-2)."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from nanoid import generate as nanoid

from rl_env.build import BuildConfig, build_environment
from rl_env.sample_room import make_sample_room
from src.models import Build, Project, Task
from tests.test_task_runtime import NAV_GOAL_SOURCE


def _mjcf(tmp_path) -> str:
    mesh = make_sample_room(size=(4.0, 3.0, 3.0), seed=3)
    p = tmp_path / "mesh.ply"
    mesh.export(str(p))
    art = build_environment(BuildConfig(mesh_path=str(p), out_dir=str(tmp_path / "build")))
    return str(art.mjcf_path)


def test_generate_endpoint_queues_and_completes(client, db, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from src.config import get_settings

    get_settings.cache_clear()

    mjcf = _mjcf(tmp_path)
    p = Project(name="codegen-test")
    db.add(p)
    db.flush()
    build = Build(id=nanoid(size=12), project_id=p.id, status="ok", mjcf_path=mjcf, n_hulls=3)
    db.add(build)
    db.flush()
    task = Task(
        id=nanoid(size=12),
        project_id=p.id,
        build_id=build.id,
        objective_nl="reach the goal",
        status="drafting",
    )
    db.add(task)
    db.commit()

    with patch(
        "src.features.tasks.service.generate_module_code",
        return_value=(NAV_GOAL_SOURCE, "raw", "claude-test"),
    ):
        from tests.job_helpers import sync_task_codegen

        r = client.post(f"/api/projects/{p.id}/tasks/{task.id}/generate")
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "generating"

        sync_task_codegen(db, task.id)

    r = client.get(f"/api/projects/{p.id}/tasks/{task.id}")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["current_code"]
    assert "GeneratedTask" in body["current_code"]
    assert body["codegen_model"] == "claude-test"


def test_generate_503_without_api_key(client, db, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    from src.config import get_settings

    get_settings.cache_clear()

    p = Project(name="no-key")
    db.add(p)
    db.flush()
    build = Build(
        id=nanoid(size=12),
        project_id=p.id,
        status="ok",
        mjcf_path="/tmp/scene.xml",
    )
    db.add(build)
    db.flush()
    task = Task(id=nanoid(size=12), project_id=p.id, build_id=build.id, status="drafting")
    db.add(task)
    db.commit()

    r = client.post(f"/api/projects/{p.id}/tasks/{task.id}/generate")
    assert r.status_code == 503
