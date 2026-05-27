# worldscan-v2.3-depth-fusion — Proposal

## What

Implement the `DepthFusionBackend` slot — one of the four reconstruction
backends that the worldscan-v2 architecture left as a stub in PR-A
(`backend/src/features/reconstruction/backends/depth_fusion.py`).

The implementation is a server-side port of matthew's browser-side pipeline
in `prototype/v4.html` (currently runnable via `python -m rl_env serve`):

1. Per frame: metric depth from Depth-Anything-V2-Metric-Indoor-Small
   (already wired through `rl_env.server._ensure_depth_model`).
2. Per frame: SuperPoint feature extraction (ONNX, fabio-sim/LightGlue-ONNX).
3. Per pair: LightGlue matching (ONNX) against the best previous frame.
4. Per pair: median-robust linear depth calibration `z_ref ≈ k·z_new + c`
   to correct monocular drift before pose alignment.
5. Per pair: rigid Umeyama + RANSAC (200 iters, 0.15 m threshold, ≥10
   inliers, scale fixed at 1 since depth is metric).
6. Per frame: back-projected colored triangle mesh in world frame, with
   depth-discontinuity culling (>0.30 m span per quad) to avoid stretched
   ghost geometry around object edges.

Outputs `mesh.ply` + `points.ply` under `out_dir`. Reuses matthew's depth
model loaders directly via the `rl_env` workspace member, so the legacy
Flask `/api/depth` endpoint and the new backend share identical weights.

## Why now

- **The architecture explicitly reserved this slot.** PR-A registered
  `depth_fusion` with `implemented = False` and a `NotImplementedError`
  pointing at this change. The registry pattern was set up to make this a
  drop-in.
- **Matthew's PR #22 demonstrated the algorithm works.** The browser
  pipeline already reconstructs from multi-photo input on consumer
  hardware. Porting it server-side puts depth-fusion on equal footing with
  the `vggt`, `colmap`, and `splat` backends inside the FastAPI flow
  (selectable from the frontend, persisted in the DB, validated by the
  validation gate).
- **No GPU-feature-matching alternative exists in the new flow.** `vggt`
  needs a serious GPU; `colmap` is classical SfM (slow); `splat` is
  gaussian splatting (different output shape). Depth-fusion is the only
  backend that produces a triangle mesh from sparse keypoint-matched depth
  on commodity hardware (CPU works, GPU optional for speed).

## What it changes

- **MODIFIED** capability `video-reconstruction`: `depth_fusion` flips
  `implemented = False → True`. No interface change — the existing
  `ReconstructionBackend` ABC and `ReconstructionInput/Output` carry it.
- **NEW internal modules** (private to `backends/`):
  - `_geometry.py` — pure-NumPy math (`robust_linear_fit`, `umeyama_rigid`,
    `ransac_umeyama`, `back_project`, `assume_intrinsics`).
  - `_models.py` — lazy-loaded depth pipelines + ONNX SuperPoint/LightGlue
    sessions, wrapping `rl_env.server` helpers.
- **Backend deps**: `transformers`, `torchvision`, `onnxruntime` added to
  `backend/pyproject.toml` (already present in `rl_env`).
- **Unit tests**: `backend/tests/test_depth_fusion_geometry.py` covers the
  math (depth calibration, Umeyama, RANSAC inlier consensus, back-projection
  with discontinuities).

## Non-goals

- **TSDF / volumetric fusion.** Matthew's pipeline doesn't do TSDF — it
  produces per-frame back-projected meshes in shared world space. We
  preserve that simplicity. A future change can add open3d-based TSDF if
  mesh quality demands it.
- **Bundle adjustment.** Per-pair pose only. Global optimization is left
  for a follow-up.
- **Depth-Pro (`apple/DepthPro-hf`)** is supported via the same code path
  but not the default — `indoor` (Depth-Anything-V2-Metric-Indoor-Small) is
  smaller, faster, and matches v4.html's default.
- **Backend selection UI changes.** The frontend already lists implemented
  backends; nothing to change there.

## Acceptance

1. `pytest backend/tests/test_depth_fusion_geometry.py` passes locally
   (math correctness — runs without GPU or model weights).
2. `GET /api/reconstruction/backends` returns `depth_fusion` with
   `implemented: true, requires_gpu: true`.
3. End-to-end on a short clip (5–10 frames, walking around a room):
   - `POST /api/projects/{id}/reconstruct` with body `{"backend": "depth_fusion"}`
     triggers an Inngest run.
   - On completion: `mesh.ply` and `points.ply` exist under
     `data/projects/{id}/reconstruction/`.
   - `Reconstruction.backend_meta` includes `n_fused`, `n_failed_fusion`,
     `avg_inliers`, vertex/face counts.
4. The legacy Flask flow still works (`python -m rl_env serve` →
   `open prototype/v4.html`) — confirming we didn't break matthew's
   reference implementation.

## Open questions

- **Intrinsics.** No real camera intrinsics yet; we default to 60° hFOV
  (matches v4.html). Frontend could let users pick "phone" / "DSLR"
  presets or read EXIF. Out of scope for this change.
- **Depth-Pro FOV.** Depth-Pro emits a per-image FOV estimate; we surface
  it in `backend_meta["fov_deg"]` but don't yet use it to override the
  default intrinsics. Worth wiring up in a follow-up — should improve
  reconstruction quality on wide-angle phone footage.
- **GPU detection.** `requires_gpu = True` advertises the requirement but
  the implementation runs (slowly) on CPU. Should the runtime check
  `torch.cuda.is_available() or torch.backends.mps.is_available()` and
  refuse on CPU, like `vggt`? Probably yes — but matthew's flow worked on
  CPU, so I left it permissive.
