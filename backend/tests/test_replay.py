def test_build_and_replay_sample_room(client, tmp_path, monkeypatch) -> None:
    """End-to-end PR-A pause-state demo: create project → build with fixture
    sample room → replay greedy policy → expect successes."""
    # Force data_dir to a temp path so the test doesn't touch /data.
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from src.config import get_settings

    get_settings.cache_clear()

    r = client.post("/api/projects", json={"name": "pr-a-smoke"})
    assert r.status_code == 201
    pid = r.json()["id"]

    r = client.post(f"/api/projects/{pid}/build", json={"up_axis": "y"})
    assert r.status_code == 200, r.text
    build = r.json()
    assert build["n_hulls"] >= 2

    r = client.post(
        f"/api/projects/{pid}/replay",
        json={"policy": "greedy", "episodes": 3, "max_steps": 200, "seed": 0},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    # Greedy on the procedural sample room is robust — we verified 28/30 earlier.
    assert data["successes"] >= 2
