# `rl_env` — WorldScan Tier 1 RL pipeline (MVP)

Mesh → MuJoCo MJCF navigation environment. Implements the 10-week-MVP slice of the
PRD's Tier 1 pipeline: preprocess → whole-scene convex decomposition → material
estimation (stub) → MJCF export → Gymnasium-compatible navigation env.

3D reconstruction lives in `prototype/` (browser-side, photo-to-mesh). This package
consumes the `.obj`/`.ply` it exports.

## Install

```bash
pip install -r requirements.txt
```

`mujoco`, `trimesh`, `gymnasium`, `numpy`. Optional: `vhacdx` or `coacd` for higher-quality convex decomposition (the pipeline auto-detects and falls back to per-component convex hulls if neither is installed).

## End-to-end demo (no scan needed)

```bash
python -m rl_env demo
```

Generates a synthetic room, builds the MJCF, runs a greedy goal-seeking policy. Should report ~3-5/5 successes — proves env, task, and physics are wired correctly.

## Browser workflow (recommended)

```bash
python -m rl_env serve
```

Then open <http://127.0.0.1:5174/>. The page is `prototype/v1.html` with two extra sections:

1. Drop a photo → click Reconstruct (browser-side photo-to-mesh, unchanged from the standalone v1).
2. **Build RL env** — uploads the in-memory mesh to the server, runs the Tier 1 pipeline, shows hulls / bounds / spawn region.
3. **Run rollout** — runs the greedy or random policy in the Gym env, shows per-episode reward / distance / success.

If the server is offline, the RL section greys out and the standalone Photo→3D + PLY/OBJ export still work.

## CLI workflow (no browser)

```bash
# 1. From prototype/v1.html, click OBJ to export.
# 2. Build the env:
python -m rl_env build path/to/worldscan.obj --out build/ --up y --diag 6

# 3. Run a rollout:
python -m rl_env run build/scene.xml --episodes 5 --policy greedy
```

## Training a PPO policy

```bash
# Train (100k steps ≈ 17s on CPU; 300k ≈ 50s)
python -m rl_env train build/scene.xml --steps 100000 --ckpt build/policy.zip

# Evaluate the trained policy
python -m rl_env play build/scene.xml --ckpt build/policy.zip --episodes 10
```

On the procedural sample room, PPO at 100k steps reaches ~9/10 success vs the
greedy baseline's 9/10 — both solve the obvious cases; the 1 failure is the
spawn/goal config where an obstacle blocks the straight-line path AND the
detour can't fit in the 300-step budget. PPO is library defaults
(`stable_baselines3.PPO("MlpPolicy", ...)`) over the
`(agent_xy, goal_xy, goal_vec, dist, 16-ray lidar)` observation; it slots
into any sb3-compatible algorithm (SAC, TD3, etc.) without code changes.

`--up y` matches the prototype's coordinate convention (Y-up from depth-anything output). `--diag 6` rescales the unitless mesh to a 6m xy-diagonal — adjust to whatever the photographed room actually is. Pass `--diag 0` to skip rescaling for already-metric scans (Polycam, iPhone LiDAR, etc.).

## Outputs

```
build/
├── scene.xml         # MJCF — load with mujoco.MjModel.from_xml_path
├── metadata.json     # bounds, spawn region, floor_z, hull count
└── meshes/
    ├── hull_0000.stl
    └── ...           # one STL per convex hull
```

## Programmatic use

```python
from rl_env import build_environment, BuildConfig, NavEnv

artifacts = build_environment(BuildConfig(
    mesh_path="scan.obj", out_dir="build/", up_axis="y", target_diagonal_m=6.0
))

env = NavEnv(artifacts.mjcf_path)
obs, _ = env.reset(seed=0)
for _ in range(500):
    action = my_policy(obs)
    obs, reward, term, trunc, info = env.step(action)
    if term or trunc:
        break
```

The env conforms to the Gymnasium API, so it slots directly into `stable-baselines3` PPO/SAC for training.

## What's in scope vs. follow-up

In this MVP (Tier 1, single static collision body):
- Mesh ingestion (`.obj`, `.ply`, anything trimesh handles)
- Y-up→Z-up auto-rotation, scale normalization, ground-plane alignment
- Convex decomposition (VHACD/CoACD if available, else per-component hulls)
- Material lookup table — geometric heuristic for floor/wall/object classes
- MJCF export with collision meshes, friction, lighting, agent body, goal site
- Gymnasium env: 2D continuous control, position + lidar observation, distance + collision reward

Out of scope — Tier 2 / post-10-week (per PRD):
- Per-object segmentation (SAM 2 + SAM 3D Objects + ICP)
- CLIP-based per-region material classification (current code has the hook in `materials.classify_hull`; replace with CLIP outputs)
- Manipulation tasks; only navigation here
- Isaac Sim USD export (P0.5 in PRD; not implemented)

## Known sharp edges

- Convex decomposition uses VHACD/CoACD if installed (`pip install vhacdx`); otherwise we fall back to one convex hull per connected component, which is coarse for irregular scans. Real scans may need VHACD for tight collision geometry.
- Single-photo reconstructions from `prototype/v1.html` are 2.5D — open at the back. The agent can wander past the captured surface. This is fine for visualizing the navigation task but not for closed-room training; multi-photo (`v2.html`) or LiDAR scans are needed for production.
- The "floor" classification heuristic looks for the lowest 8% of mesh height. On a tilted scan this may misclassify; pass `--up` explicitly to ensure correct gravity alignment.
