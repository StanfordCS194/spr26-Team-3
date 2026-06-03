def test_reconstruction_backends_list(client) -> None:
    r = client.get("/api/reconstruction/backends")
    assert r.status_code == 200
    by_name = {b["name"]: b for b in r.json()}
    assert set(by_name) == {"vggt", "splat", "colmap", "depth_fusion", "demo_fixture"}
    assert by_name["demo_fixture"]["implemented"] is True
    assert by_name["demo_fixture"]["requires_gpu"] is False
    assert by_name["colmap"]["implemented"] is False
    assert by_name["splat"]["implemented"] is False
