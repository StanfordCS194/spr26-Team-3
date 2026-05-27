# WorldScan

Turn a recording of a real room into a physically simulated environment that
an RL agent can be trained inside. CS194W (Stanford, Spring 2026).

Matthew Kim · Adarsh Ambati · Andrew Sung · Kevin Wang · Aditya Iyengar

[Wiki](https://github.com/StanfordCS194/spr26-Team-3/wiki)

---

## What's here

```
frontend/        React 19 + Vite + TS + TanStack + Radix + Tailwind v4
backend/         FastAPI + SQLAlchemy 2.0 + Alembic + Inngest (Python SDK)
shared/          OpenAPI → openapi-typescript output, consumed by frontend
rl_env/          Python ML package: mesh → MJCF → Gymnasium env → PPO
archive/legacy/  the v0-v3 prototype HTMLs (read-only)
openspec/        change proposals + capability specs
data/            gitignored runtime state (per-project artifacts)
```

The active restructure is documented under
`openspec/changes/worldscan-v2-video-product/` — a five-PR refactor with
explicit pause-state acceptance criteria. See `AGENTS.md` for the workflow.

## Quick start

```bash
# bring up the whole stack (postgres, inngest-dev, api, worker, frontend)
just dev

# open the SPA
open http://localhost:5173

# alternative: headless smoke test (no docker, no UI)
.venv/bin/python -m rl_env demo
```

Frontend: `http://localhost:5173` · Backend OpenAPI: `http://localhost:8000/docs`
· Inngest dashboard: `http://localhost:8289` · Postgres: `localhost:5433`

(Ports 5433 and 8289 instead of the conventional 5432/8288 to avoid clashing
with the sibling `Haleum/` stack if it's running on the same host.)

## First-time setup

```bash
# 1. Python deps (rl_env + backend share a workspace)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt              # rl_env runtime
cd backend && uv sync && cd ..                          # backend + dev tools

# 2. Node deps
yarn install

# 3. Database
just dev      # postgres comes up under docker-compose
just db-migrate

# 4. Generate the typed API client from the running backend
just typegen
```

If you don't have `uv` or `just` installed:

```bash
brew install uv just     # macOS
# or: pip install uv && cargo install just
```

## Where to look

- **The plan.** `openspec/changes/worldscan-v2-video-product/proposal.md`
  (why), `design.md` (how), `tasks.md` (broken into five PRs with
  acceptance criteria).
- **The contract.** `openspec/changes/worldscan-v2-video-product/specs/`
  contains the spec deltas — five capabilities total.
- **The workflow.** `AGENTS.md` at the repo root.

## What's currently live

This branch (`change/worldscan-v2-video-product/foundation`) is **PR-A**: the
foundation. The product runs end-to-end on the procedural sample room
through the new stack; Capture / Reconstruct / Validate / Train screens are
visible-but-disabled placeholders.

Live demo loop:

1. `just dev`
2. Open `localhost:5173`, click `+` in the sidebar, name a project.
3. Click into the project → **Build** screen → "Build env".
4. **Replay** screen → "Run greedy". Expect ~4-5/5 successes (same numbers
   as the headless `python -m rl_env demo`).

PR-B adds the real Capture / Reconstruct / Validate flow.

## Headless CLI (unchanged)

The `rl_env` package keeps its CLI:

```bash
.venv/bin/python -m rl_env demo
.venv/bin/python -m rl_env build path/to/mesh.obj --out build/
.venv/bin/python -m rl_env run   build/scene.xml --episodes 5 --policy greedy
.venv/bin/python -m rl_env train build/scene.xml --steps 100000 --ckpt build/policy.zip
.venv/bin/python -m rl_env play  build/scene.xml --ckpt build/policy.zip --episodes 10
```

This is the headless smoke test we promise to keep working at every PR
boundary of the restructure.
