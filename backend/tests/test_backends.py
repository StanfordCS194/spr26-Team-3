def test_reconstruction_backends_list(client) -> None:
    r = client.get("/api/reconstruction/backends")
    assert r.status_code == 200
    names = {b["name"] for b in r.json()}
    assert names == {"vggt", "splat", "depth_fusion", "demo_fixture"}
