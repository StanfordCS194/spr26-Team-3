"""Task authoring API (PR-1): schema + CRUD, no codegen yet."""
from __future__ import annotations

from src.models import Build, Project, Task


def test_task_crud_requires_build(client, db) -> None:
    r = client.post("/api/projects", json={"name": "task-test"})
    assert r.status_code == 201
    pid = r.json()["id"]

    r = client.post(
        f"/api/projects/{pid}/tasks",
        json={
            "objective_nl": "reach the chair",
            "env_nl": "floor only",
            "agent_nl": "circle agent, lidar",
        },
    )
    assert r.status_code == 400

    p = db.get(Project, pid)
    assert p is not None
    build = Build(
        id="buildtask01",
        project_id=pid,
        status="ok",
        mjcf_path="/tmp/scene.xml",
        n_hulls=3,
        bounds={"min": [0, 0, 0], "max": [4, 4, 2]},
        spawn_region={"xmin": 0.5, "xmax": 3.5, "ymin": 0.5, "ymax": 3.5},
    )
    db.add(build)
    db.commit()

    r = client.post(
        f"/api/projects/{pid}/tasks",
        json={
            "name": "Reach chair",
            "objective_nl": "reach the chair",
            "env_nl": "floor only",
            "agent_nl": "circle agent, lidar",
            "goal_3d": {"x": 2.0, "y": 3.0, "z": 0.0, "radius": 0.3},
        },
    )
    assert r.status_code == 201, r.text
    tid = r.json()["id"]
    assert r.json()["status"] == "drafting"
    assert r.json()["build_id"] == build.id
    assert r.json()["goal_3d"]["x"] == 2.0

    r = client.get(f"/api/projects/{pid}/tasks")
    assert r.status_code == 200
    assert len(r.json()) == 1

    r = client.get(f"/api/projects/{pid}/tasks/{tid}")
    assert r.status_code == 200

    r = client.patch(
        f"/api/projects/{pid}/tasks/{tid}",
        json={"objective_nl": "reach the chair and stop"},
    )
    assert r.status_code == 200
    assert r.json()["objective_nl"] == "reach the chair and stop"

    r = client.post(f"/api/projects/{pid}/tasks", json={"objective_nl": "duplicate"})
    assert r.status_code == 409

    r = client.get(f"/api/projects/{pid}/state")
    assert r.status_code == 200
    assert r.json()["task"]["complete"] is False
    assert "PR-3" in (r.json()["task"]["reason"] or "")


def test_project_state_task_ready(db, client) -> None:
    p = Project(name="ready-task")
    db.add(p)
    db.flush()
    build = Build(
        id="buildtask02",
        project_id=p.id,
        status="ok",
        mjcf_path="/tmp/scene.xml",
    )
    db.add(build)
    db.flush()
    task = Task(
        id="taskready01",
        project_id=p.id,
        build_id=build.id,
        status="ready",
        objective_nl="x",
    )
    db.add(task)
    db.commit()

    r = client.get(f"/api/projects/{p.id}/state")
    assert r.json()["task"]["complete"] is True
