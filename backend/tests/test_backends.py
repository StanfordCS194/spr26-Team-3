def test_reconstruction_backends_list(client) -> None:
    r = client.get("/api/reconstruction/backends")
    assert r.status_code == 200
    names = {b["name"] for b in r.json()}
    assert names == {
        "vggt", "pi3", "mapanything", "depth_fusion", "depth_fusion_da3",
        "demo_fixture",
    }
