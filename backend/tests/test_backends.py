def test_reconstruction_backends_list(client) -> None:
    r = client.get("/api/reconstruction/backends")
    assert r.status_code == 200
    names = {b["name"] for b in r.json()}
    assert names == {"vggt", "splat", "colmap", "depth_fusion"}
    # all four are stubs in PR-A
    assert all(not b["implemented"] for b in r.json())
