def test_create_list_get_delete_project(client) -> None:
    # create
    r = client.post("/api/projects", json={"name": "smoke"})
    assert r.status_code == 201
    pid = r.json()["id"]

    # list
    r = client.get("/api/projects")
    assert r.status_code == 200
    assert any(p["id"] == pid for p in r.json())

    # get
    r = client.get(f"/api/projects/{pid}")
    assert r.status_code == 200
    assert r.json()["name"] == "smoke"

    # rename
    r = client.patch(f"/api/projects/{pid}", json={"name": "renamed"})
    assert r.status_code == 200
    assert r.json()["name"] == "renamed"

    # delete
    r = client.delete(f"/api/projects/{pid}")
    assert r.status_code == 204
