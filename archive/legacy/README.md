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
inert because the old Flask server (`rl_env/server.py`) has been removed.

See `openspec/changes/archive/worldscan-v2-video-product/` (once landed)
for why this slice of the project was archived.
