# Legacy prototype HTMLs

These are the single-file browser-side photo-to-3D prototypes from before
the worldscan-v2 restructure.

| File | What it was |
|---|---|
| `index.html` | original WorldScan interactive prototype |
| `v0.html` | duplicate of `index.html` checkpoint |
| `v1.html` | single-photo depth-anything reconstruction |
| `v2.html` | multi-photo fusion experiment |
| `v3.html` | merged v1+v2 + RL env integration (the last live demo) |

They are kept as a frozen reference. Do not modify. The new product lives
under `frontend/`, `backend/`, `shared/`.

To run any of them standalone, open the file directly in a browser:
`open archive/legacy/v3.html` — the RL-env integration buttons will be
inert because no Flask server is running.

## v4 / v4.2 — still active

Matthew's later browser prototypes — `prototype/v4.html` and
`prototype/v4.2.html` — are *not* archived. They live in `prototype/` and
are served by the legacy Flask server:

```bash
python -m rl_env serve   # → http://127.0.0.1:5174/
```

The server exposes `/api/depth` (metric depth via Depth-Anything-V2 or
Apple Depth-Pro) and `/api/models/<name>` (ONNX model cache for
SuperPoint + LightGlue feature matching). The browser does all the
feature matching, depth calibration, RANSAC + rigid Umeyama pose, and
mesh fusion client-side.

This is matthew's reference implementation of the depth-fusion pipeline.
The server-side port — exposed via the new FastAPI backend as a selectable
backend — lives at
`backend/src/features/reconstruction/backends/depth_fusion.py` (change
`worldscan-v2.3-depth-fusion`).

See `openspec/changes/archive/worldscan-v2-video-product/` (once landed)
for why this slice of the project was archived.
