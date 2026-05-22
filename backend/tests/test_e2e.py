"""End-to-end smoke test exercising the full PR-A through PR-D pipeline
against the procedural sample room (no GPU, no model weights).

Uses the testcontainers Postgres + FastAPI TestClient fixtures from
conftest.py. Validates that every documented "live demo" loop still works
after the chain is squashed onto main.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path


def test_full_pipeline(client, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from src.config import get_settings
    get_settings.cache_clear()

    # 1. create project
    r = client.post("/api/projects", json={"name": "e2e demo"})
    assert r.status_code == 201
    pid = r.json()["id"]

    # 2. upload (use the sample_room as a stand-in for video — endpoint
    #    accepts jpg/png too, but the demo_fixture backend doesn't read it)
    fake_jpg = tmp_path / "in.jpg"
    fake_jpg.write_bytes(
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
    with fake_jpg.open("rb") as fh:
        r = client.post(
            f"/api/projects/{pid}/upload-video",
            files={"file": ("in.jpg", fh, "image/jpeg")},
        )
    assert r.status_code == 200, r.text

    # 3. reconstruct via demo_fixture (works offline, no GPU)
    r = client.post(
        f"/api/projects/{pid}/reconstruct",
        json={"backend": "demo_fixture"},
    )
    assert r.status_code == 200, r.text
    rid = r.json()["id"]

    # poll until done (in-process worker — should complete in <2s)
    import time
    for _ in range(20):
        time.sleep(0.2)
        r = client.get(f"/api/projects/{pid}/reconstruction")
        if r.json() and r.json().get("status") in ("ok", "failed"):
            break
    assert r.json()["status"] == "ok", r.json()

    # 4. validate
    r = client.post(f"/api/projects/{pid}/validate")
    assert r.status_code == 200, r.text
    assert r.json()["report"]["overall"] in ("pass", "warn")

    # 5. build
    r = client.post(f"/api/projects/{pid}/build", json={"up_axis": "y"})
    assert r.status_code == 200, r.text
    assert r.json()["n_hulls"] >= 2

    # 6. replay greedy — should succeed on demo_fixture room
    r = client.post(
        f"/api/projects/{pid}/replay",
        json={"policy": "greedy", "episodes": 3, "max_steps": 200, "seed": 0},
    )
    assert r.status_code == 200
    assert r.json()["successes"] >= 2

    # 7. summary reflects the chain
    r = client.get("/api/projects/summary")
    assert r.status_code == 200
    summary = next(p for p in r.json() if p["id"] == pid)
    assert summary["status_pill"] == "Built"

    # 8. export bundles the project folder as a zip
    r = client.post(f"/api/projects/{pid}/export")
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    assert any("mesh.ply" in n for n in names), names
    assert any("scene.xml" in n for n in names), names

    # 9. delete cascades
    r = client.delete(f"/api/projects/{pid}")
    assert r.status_code == 204
