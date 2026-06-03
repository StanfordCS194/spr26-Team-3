from tests.job_helpers import sync_build, sync_reconstruct, sync_replay


def _fake_jpg_bytes() -> bytes:
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n"
        b"\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d"
        b"\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00"
        b"\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14\x00\x01"
        b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00\x00"
        b"?\x00\x37\xff\xd9"
    )


def test_build_and_replay_sample_room(client, db, tmp_path, monkeypatch) -> None:
    """Create project → reconstruct (fixture) → build → replay greedy."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from src.config import get_settings

    get_settings.cache_clear()

    r = client.post("/api/projects", json={"name": "pr-a-smoke"})
    assert r.status_code == 201
    pid = r.json()["id"]

    with (tmp_path / "in.jpg").open("wb") as fh:
        fh.write(_fake_jpg_bytes())
    with (tmp_path / "in.jpg").open("rb") as fh:
        r = client.post(
            f"/api/projects/{pid}/upload-video",
            files={"file": ("in.jpg", fh, "image/jpeg")},
        )
    assert r.status_code == 200, r.text

    r = client.post(
        f"/api/projects/{pid}/reconstruct",
        json={"backend": "demo_fixture"},
    )
    assert r.status_code == 200, r.text
    sync_reconstruct(db, r.json()["id"], "demo_fixture")

    r = client.post(f"/api/projects/{pid}/build", json={"up_axis": "y"})
    assert r.status_code == 200, r.text
    sync_build(db, r.json()["id"], up_axis="y")

    r = client.get(f"/api/projects/{pid}/build/latest")
    assert r.json()["n_hulls"] >= 2

    r = client.post(
        f"/api/projects/{pid}/replay",
        json={"policy": "greedy", "episodes": 3, "max_steps": 200, "seed": 0},
    )
    assert r.status_code == 200, r.text
    run_id = r.json()["id"]
    sync_replay(
        db,
        run_id,
        project_id=pid,
        policy="greedy",
        episodes=3,
        max_steps=200,
        seed=0,
    )

    r = client.get(f"/api/projects/{pid}/runs/{run_id}")
    assert r.json()["successes"] >= 2
