# Tasks — worldscan-v2-video-product

Five PRs, each independently mergeable and demoable. PR boundaries are drawn
so every PR is reversible without breaking earlier work. Acceptance criteria
are explicit so a reviewer (or `openspec validate`) can mechanically check
"is this PR done."

## Conventions

- Every PR opens against `main` and lands behind a single commit (squash on
  merge).
- Every PR ships green CI: `just test` (pytest + vitest) and `just lint`.
- Every PR updates `openspec/changes/worldscan-v2-video-product/tasks.md`
  to check off its scope (`[ ]` → `[x]`).
- The change is archived when PR-E lands. At that point `openspec` archives
  this directory and merges the spec deltas into `openspec/specs/<cap>/`.

## Pause-stage guarantee

Every PR leaves the product in a **runnable, demonstrable state**. Not
complete, but usable: a teammate cloning at any merged commit can run
`just dev` and follow the "Live state after this PR ships" section below to
exercise what's available. This means:

- No half-wired routes (every endpoint in the API returns a sane response).
- No dead-end UI (every screen that exists either works or is gated behind
  a feature that the user can't reach yet).
- No required env vars without working defaults.
- Failed jobs surface a visible error, never a blank screen.
- The legacy `python -m rl_env demo` CLI keeps working as a headless smoke
  test at every PR boundary.

If a PR cannot satisfy this, it gets split until it can.

---

## PR-A — Cleanup + scaffold (foundation)

Goal: legacy prototype archived, new monorepo tree in place, the procedural
sample-room flow works end-to-end through FastAPI + the new React UI, no new
product features.

### Tasks

- [ ] A.1. Move `prototype/v0.html`, `v1.html`, `v2.html`, `v3.html`,
       `index.html` into `archive/legacy/`. Add `archive/README.md` linking
       back to the v3 demo URL and noting why it's frozen.
- [ ] A.2. Delete `rl_env/server.py`. Routes move to `backend/`. Leave
       `rl_env/__main__.py` (CLI) untouched.
- [ ] A.3. Initialize yarn workspaces at the repo root. Root `package.json`
       lists `frontend/`, `shared/` as workspaces. Add `Justfile` with
       `dev`, `test`, `lint`, `typegen`, `db:migrate`, `db:reset`.
- [ ] A.4. Replace `requirements.txt` with `backend/pyproject.toml`
       (Python 3.13, uv-managed). Move existing deps under `[project]
       dependencies`; add fastapi, uvicorn, pydantic, pydantic-settings,
       sqlalchemy, alembic, psycopg, inngest, sentry-sdk, ruff, black,
       pytest, pytest-asyncio.
- [ ] A.5. Create `backend/src/` with `app.py` (FastAPI factory), `config.py`
       (pydantic-settings), `db.py`, `deps.py`, `inngest_client.py`,
       `worker.py`, `instrument.py`, `middleware/`. App boots, `/health`
       returns 200.
- [ ] A.6. Postgres schema + initial Alembic migration. Tables:
       `project`, `reconstruction`, `validation`, `build`, `policy`, `run`
       (see `design.md` data model). `just db:migrate` applies cleanly.
- [ ] A.7. `backend/src/features/projects/` + `backend/src/features/builds/`
       + `backend/src/features/replay/` — minimum routers/services to drive
       the sample room. Endpoints: `POST /api/projects` (no video yet, name
       only), `POST /api/projects/{id}/build` (calls
       `rl_env.build.build_environment` with a fixture mesh), `POST
       /api/projects/{id}/replay` (calls `rl_env.env.NavEnv`, returns
       trajectories).
- [ ] A.8. `backend/src/features/reconstruction/backends/` —
       `base.py` (ABC), `__init__.py` (registry), and four stub backends
       (`vggt.py`, `splat.py`, `colmap.py`, `depth_fusion.py`) — all with
       `implemented=False` and `reconstruct` raising. `GET
       /api/reconstruction/backends` returns the list with availability
       flags.
- [ ] A.9. Create `frontend/` with Vite + React 19 + TS, TanStack Router,
       TanStack Query, Tailwind v4 (`@tailwindcss/vite`), Radix primitives,
       shadcn `components.json`, Sentry, Posthog (deps only — no init keys
       yet). Routes: `/` (project list), `/p/$projectId/build`,
       `/p/$projectId/replay`. Sidebar component renders project list from
       `GET /api/projects`.
- [ ] A.10. Create `shared/` workspace. `just typegen` writes
        `shared/openapi.json` from the running backend and runs
        `openapi-typescript` into `shared/ts/api.d.ts`. Husky pre-commit
        hook re-runs typegen.
- [ ] A.11. `docker-compose.yaml` at the root: services `postgres`,
        `inngest-dev`, `api`, `worker`, `frontend`. `just dev` starts all
        five with `docker compose up --watch`.
- [ ] A.12. `tests/conftest.py` with a Postgres test container fixture and
        an autouse rollback per test. Add `backend/tests/test_projects.py`,
        `backend/tests/test_builds.py`, `backend/tests/test_replay.py` —
        each creates a project, hits the endpoint, asserts schema.
- [ ] A.13. Update root `README.md` to point at the new dev story
        (`just dev`, `http://localhost:5173`). Link to `archive/legacy/`.

### Acceptance

- Repo tree matches `design.md`'s layout exactly (no stragglers in
  `prototype/`).
- `just dev` brings up five containers. `http://localhost:5173` renders the
  project list. Creating a project → clicking Build → seeing a trajectory in
  Replay reproduces the headline number we already verified: greedy ≈ 28/30
  on the procedural sample room.
- `GET /openapi.json` is non-empty; `shared/ts/api.d.ts` is generated and
  consumed by the frontend client (typecheck passes).
- `just test` green. `just lint` green.

### Live state after this PR ships

End-to-end demo available on day 1 of the new stack:

1. Clone the repo, `just dev`.
2. Open `http://localhost:5173`. Sidebar is empty.
3. Click "+ New project" → name it.
4. Click Build (no Capture / Reconstruct / Validate screens yet, but a fixture
   mesh is pre-seeded).
5. Click Replay → see greedy & random trajectories on the procedural sample
   room.

What does **not** work yet, gated visibly: Capture, Reconstruct, Validate,
Train. Those routes either don't exist in the router or render a "Coming in
PR-B" placeholder. No broken buttons, no 500s.

---

## PR-B — Video capture + VGGT reconstruction + validation gate

Goal: user can drop a video, see it become a mesh, see a pass/warn/fail
report on that mesh.

### Tasks

- [ ] B.1. Extend `POST /api/projects` to accept a multipart `video` field.
       Server saves under `data/projects/<id>/input.<ext>` and uses
       `ffmpeg-python` to sample N=24 frames evenly under
       `data/projects/<id>/frames/`. Persist `video_path` and
       `thumbnail_path` on the row.
- [ ] B.2. Implement `backend/src/features/reconstruction/backends/vggt.py`.
       Wraps the VGGT reference implementation; caches weights under
       `~/.cache/worldscan/vggt/`. Flip `implemented=True`. License check
       documented in `docs/reconstruction-vggt.md`.
- [ ] B.3. Inngest function `reconstruct_video` in
       `backend/src/features/reconstruction/inngest_functions.py`. Steps:
       `extract_frames` → `run_backend` → `persist`. Emits progress events
       via `step.send_event` so the frontend can subscribe.
- [ ] B.4. `POST /api/projects/{id}/reconstruct` endpoint queues the
       Inngest event with `{project_id, backend, params}`. Returns
       `reconstruction_id` and `inngest_run_id` immediately.
- [ ] B.5. `backend/src/features/validation/checks.py` — implement all six
       checks from `design.md`'s validation catalog. Each is a pure
       function `(mesh: trimesh.Trimesh) -> CheckResult`.
- [ ] B.6. `POST /api/projects/{id}/validate` runs all checks synchronously
       on the project's latest reconstruction. Persists a `validation` row.
- [ ] B.7. `frontend/src/routes/p.$projectId.capture.tsx` — drag-drop
       upload, ffprobe-style summary (duration, fps, resolution), kicks off
       project creation. `Reconstruct.tsx` — backend dropdown (only `vggt`
       enabled), progress bar driven by `@inngest/react` events, mesh
       preview via `MeshViewer.tsx` once done.
- [ ] B.8. `frontend/src/routes/p.$projectId.validate.tsx` —
       `ValidationReport.tsx` component renders the pass/warn/fail
       checklist with `fix` text inline. "Build env" button disabled until
       no `fail` results OR user-override checkbox is ticked.
- [ ] B.9. Tests: `backend/tests/test_validation.py` runs each check on
       hand-crafted good/bad meshes; `backend/tests/test_reconstruct.py`
       uses a precomputed-output fixture so CI doesn't need a GPU.

### Acceptance

- Drop in any `.mp4` of a real room → `reconstruct` job runs → mesh
  appears in the frontend's `MeshViewer`.
- Validation runs automatically after reconstruction; report shows the six
  named checks; bad scans surface their failure modes plainly.
- The chair-photo failure case from our verification session now produces a
  validation `fail` on `convex_decomp_quality` and `floor_detected` instead
  of silently building a broken env.

### Live state after this PR ships

Full pipeline up to a built env, no training yet:

1. `just dev`, open the app.
2. Capture screen accepts an mp4 (or still image for back-compat).
3. Reconstruct screen runs VGGT, shows live progress via Inngest events,
   renders the mesh in `MeshViewer`.
4. Validate screen shows the six-check report; bad scans block Build
   unless overridden.
5. Build + Replay still work end-to-end with the new reconstruction
   feeding into them — the procedural-sample-room flow from PR-A
   continues to work as the headless smoke test.

What does **not** work yet, gated visibly: Train screen renders a
"Coming in PR-C" panel with a disabled "Train PPO" button. PPO can still
be trained via `python -m rl_env train` CLI for anyone who needs it
mid-PR.

---

## PR-C — Training UX + trajectory visualization

Goal: replace today's "fail/success table" with something a non-author
understands in 10 seconds.

### Tasks

- [ ] C.1. Inngest function `train_policy` in
       `backend/src/features/training/inngest_functions.py`. PPO loop
       emits a progress event every N=1000 steps with `{progress,
       current_reward, current_success_rate, fps}`.
- [ ] C.2. `POST /api/projects/{id}/train` queues the event. Returns
       `policy_id` and `inngest_run_id`.
- [ ] C.3. `frontend/src/components/MetricsChart.tsx` — recharts line
       chart, three series (reward, success rate, episode length).
       Subscribes to progress events; updates ≥ 1 Hz.
- [ ] C.4. `frontend/src/components/TrajectoryViewer.tsx` — SVG / canvas
       hybrid: floor polygon (from build metadata) + agent trace + start
       (green) + goal (red) + collision dots (magenta). Same visual we
       prototyped in `demo_out/ppo_rollout.png`.
- [ ] C.5. `backend/src/features/replay/service.py` — classify each
       trajectory as `success` / `timeout` / `stuck` / `collided` /
       `near-miss`. Rules documented in the service.
- [ ] C.6. `POST /api/projects/{id}/replay` returns trajectories for the
       trained policy plus the random + greedy baselines (computed inline,
       cached on the `run` row).
- [ ] C.7. `frontend/src/routes/p.$projectId.replay.tsx` — three panels:
       (a) per-policy benchmark numbers (current table view), (b)
       trajectories overlaid on the floor view, (c) failure-class summary
       with click-to-zoom on representative episodes.

### Acceptance

- "Why did the agent fail?" answerable from the Replay screen in ≤ 10
  seconds for any reasonably-cooperative scene.
- Training metrics update live; the user never wonders "is it still
  running?"
- Procedural sample room shows the same 27/30 PPO vs 28/30 greedy gap we
  verified, now visually instead of as terminal output.

### Live state after this PR ships

End-to-end demo of a single project:

1. `just dev`.
2. Capture → Reconstruct → Validate → Build → Train → Replay all work
   for one project at a time.
3. Replay shows random, greedy, and trained PPO trajectories overlaid,
   plus the failure-class breakdown.

What does **not** work yet, gated visibly: the sidebar shows projects but
can't rename / delete / export them; closing the app and reopening lands
on the project list, not on the last-visited step (resume-where-you-left
comes in PR-D). Compare-two-runs UI not yet present.

---

## PR-D — Project sidebar + run history

Goal: persistent left sidebar. Resume any prior project from where you left
off. Compare runs.

### Tasks

- [ ] D.1. `frontend/src/components/Sidebar.tsx` — list of projects with
       thumbnail, inline-editable name (`PATCH /api/projects/{id}`),
       status pill, created-at. Context menu: rename, delete, duplicate.
- [ ] D.2. `GET /api/projects` joins across child tables so the status
       pill is derivable from the response without N+1 queries.
- [ ] D.3. Resume-where-you-left-off: when a project is clicked, the
       backend computes the furthest stage with a non-failed artifact and
       the frontend navigates to it.
- [ ] D.4. Run-history panel inside Replay: list of past runs with
       `(algo, total_steps, success_rate, date)`. Multi-select two runs →
       side-by-side `TrajectoryViewer`.
- [ ] D.5. `POST /api/projects/{id}/export` zips `data/projects/<id>/`,
       streams the bundle. Sidebar context menu exposes it.
- [ ] D.6. `DELETE /api/projects/{id}` — cascade in Postgres, then
       `shutil.rmtree(data/projects/<id>)`. Confirm dialog: type project
       name to confirm.

### Acceptance

- Close laptop, open next day, find every project where you left it.
- Two PPO runs at different `total_steps` compare side-by-side; user can
  see which one generalizes better.
- Deleting a project leaves no orphan rows or files.

### Live state after this PR ships

Multi-project workflow fully usable:

1. Multiple projects sit in the sidebar.
2. Each project resumes where you left off.
3. Rename / delete / export / duplicate work from the sidebar context
   menu.
4. Compare-two-runs visualizes both trajectories on the same scene.

What does **not** work yet, gated visibly: rough edges around error UX
(some errors still surface as red toasts without recovery actions),
no keyboard shortcuts, no advanced settings panel, README still points
at the legacy demo. PR-E polishes all of this.

---

## PR-E — Polish + DX + docs

Goal: a teammate clones the repo, runs `just dev`, and finishes the chair →
policy demo in 5 minutes.

### Tasks

- [ ] E.1. `backend/tests/test_e2e.py` — bundled tiny fixture video →
       reconstruct (uses precomputed-output fixture in CI) → validate →
       build → train (1k steps) → replay. Asserts shape of every artifact.
- [ ] E.2. `README.md` rewrite. New `docs/architecture.md` distilled from
       `design.md`. New `docs/adding-a-reconstruction-backend.md` walk-through.
- [ ] E.3. Error UX: every Inngest failure surfaces a useful error in the
       UI (current backend traces are eaten by 500s). Sentry-tagged with
       project/run IDs.
- [ ] E.4. Loading skeletons everywhere; no blank screens during
       reconstruction or training waits.
- [ ] E.5. Keyboard shortcuts: ⌘K command palette (cmdk) to switch
       projects, ⌘\\ toggle sidebar, ⌘↵ advance to next step.
- [ ] E.6. Configuration: move all magic numbers (`success_radius`,
       `max_steps`, `lidar_max`, `target_diagonal_m`) into
       `backend/src/config.py`; expose in `Build.tsx` as Advanced
       Settings (collapsed by default).
- [ ] E.7. Migration script: any existing `server_builds/<bid>/` on disk
       is moved to `data/projects/<bid>/build/` and a `project` row is
       created with `name="Migrated <bid>"`. Run as one-shot in PR-E.

### Acceptance

- Fresh `git clone` + `just dev` to demo working: under 5 minutes,
  including Docker pulls.
- Every error has a recovery path in the UI (Retry button on jobs, clear
  red banner with detail link).
- All five PRs' spec deltas validate (`openspec validate`).

### Live state after this PR ships

The whole change is now in **archive-ready** state. After PR-E merges,
the change is archived:

```
openspec archive worldscan-v2-video-product
```

This moves `openspec/changes/worldscan-v2-video-product/` to
`openspec/changes/archive/`, and merges the spec deltas into
`openspec/specs/{video-reconstruction,physics-validation,project-history,nav-env,rl-training}/spec.md`.

The product at this point: ship a teammate a 5-minute demo by handing
them the repo URL. Fresh clone, `just dev`, drop a phone video of a room,
watch it become a navigable RL environment, train PPO, watch the policy
solve it.

---

## Deferred (waiting on input)

- **Visual direction.** The user will share UI examples to emulate; the
  shadcn primitives chosen in PR-A and PR-C are framework-neutral, so the
  visual layer can be tuned without restructuring components.

---

## Out of this change (future)

Two orthogonal axes for reconstruction extensions: which **backend** runs
(splat / colmap / depth-fusion) versus **where** it runs (local / cloud)
versus **what** it ingests (batch video / live stream). Each axis is its
own future change so they can compose.

| Future change | Axis | Builds on |
|---|---|---|
| `worldscan-v2.1-splat` | new backend | reconstruction plugin interface |
| `worldscan-v2.2-colmap` | new backend (CPU-friendly) | reconstruction plugin interface |
| `worldscan-v2.3-depth-fusion` | new backend | reconstruction plugin interface |
| `worldscan-v2.4-cloud-reconstruction` | new execution location | adds a `local` ↔ `cloud` selector orthogonal to backend choice; offloads to Modal / Runpod / AWS for splat-class work |
| `worldscan-v2.5-stream-reconstruction` | new input modality + dynamic scene support | WebRTC / HLS ingest, incremental per-frame reconstruction, ego-motion vs object-motion separation for moving objects |
| `worldscan-v3-object-segmentation` | Tier 2 — semantics | SAM 2 + CLIP, per-object hulls, manipulation tasks |
| `worldscan-v3.1-deploy` | hosting | Inngest Cloud + managed Postgres + auth + multi-user |
