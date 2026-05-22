# AGENTS.md — WorldScan

Guidance for Claude Code, the team's AI assistants, and humans dropping in
cold.

## What this repo is

WorldScan turns a recording of a real room into a physically simulated
environment that an RL agent can be trained inside. CS194W (Stanford,
Spring 2026).

Two halves:

- **`prototype/*.html`** (legacy, being archived) — single-photo browser-side
  reconstruction.
- **`rl_env/`** — Python package: mesh → MJCF → Gymnasium env → PPO.

The active restructure lives under
`openspec/changes/worldscan-v2-video-product/` — a five-PR refactor into a
real monorepo product with a video pipeline, validation gate, and run
history.

## Repo layout (post-restructure)

```
spr26-Team-3/
├── frontend/        React 19 + Vite + TS + TanStack + Radix + Tailwind v4
├── backend/         FastAPI + SQLAlchemy 2.0 + Alembic + Inngest (Python SDK)
├── shared/          OpenAPI → openapi-typescript output, consumed by frontend
├── rl_env/          Python ML package, unchanged (used as a library)
├── archive/legacy/  the v0-v3 prototype HTMLs, frozen for reference
├── data/            gitignored; per-project artifacts
├── docker-compose.yaml
├── Justfile         dev, test, lint, typegen, db:migrate
└── openspec/        change proposals + capability specs
```

## Development commands

```bash
just dev          # docker compose up --watch — postgres, inngest-dev, api, worker, frontend
just test         # pytest + vitest
just lint         # ruff + black + eslint + prettier
just typegen      # backend openapi.json → shared/ts/api.d.ts
just db:migrate   # alembic upgrade head
just db:reset     # drop, recreate, migrate, seed
```

Frontend at `http://localhost:5173`. Backend OpenAPI at
`http://localhost:8000/docs`. Inngest dashboard at `http://localhost:8288`.

## Stack alignment

Everything above the ML/sim line mirrors the team's sibling repo
`Haleum/` so engineers don't context-switch between two stacks:

| Layer | Choice | Same as Haleum? |
|---|---|---|
| Frontend | Vite + React 19 + TS + TanStack Router/Query/Form + Radix + shadcn + Tailwind v4 | yes |
| Jobs | Inngest + separate worker process | yes |
| DB | Postgres + SQLAlchemy 2.0 + Alembic | yes (dialect; ORM differs by language) |
| Backend | FastAPI | no — Haleum is Express+TS; we need Python for MuJoCo/sb3/VGGT |
| Repo shape | yarn workspaces monorepo + Docker Compose dev | yes |

## OpenSpec workflow

Non-trivial changes go through `openspec/changes/<change-id>/` before code
is written.

```
openspec/
├── config.yaml                   project metadata + conventions
├── specs/                        the CURRENT state of capabilities (truth)
│   └── <capability>/spec.md
└── changes/                      proposals
    ├── archive/                  landed changes
    └── <change-id>/              in-flight
        ├── proposal.md           why + what
        ├── design.md             how (optional)
        ├── tasks.md              broken into PRs with discrete checklists
        └── specs/<capability>/spec.md   delta to apply on archive
```

### Spec delta format

```markdown
## ADDED Requirements
### Requirement: <Title>
<one paragraph with SHALL>

#### Scenario: <name>
- **WHEN** ...
- **THEN** ...
- **AND** ...
```

`## MODIFIED Requirements` and `## REMOVED Requirements` follow the same
shape. Every Requirement has at least one Scenario.

### Multi-PR rules

1. **PR-A** is always pure restructure / scaffolding. It introduces no new
   product features. Its job: prove the foundation works before features
   land on top of it.
2. **Every PR is a pause stage.** The product must run after each PR — not
   complete, but usable end-to-end at some level. `tasks.md` makes the
   "Live state after this PR" explicit for each one.
3. **Acceptance criteria are mechanical.** A reviewer checks them
   one-by-one; "feels good" is not acceptance.
4. **The final PR triggers archival.** Run `openspec archive <change-id>`
   — it moves the directory and applies spec deltas to `openspec/specs/`.

### CLI

```bash
npm install -g @fission-ai/openspec@latest
openspec list                       # in-flight changes
openspec show <change-id>           # render proposal + design + tasks
openspec validate <change-id>       # structural checks
openspec diff <change-id>           # what archival would change in specs/
openspec archive <change-id>        # land the change
```

If the CLI isn't installed, validation is by-eye against the rules above.

## Current in-flight change

| Change | Status | PRs landed | Owner |
|---|---|---|---|
| `worldscan-v2-video-product` | proposed | 0 / 5 | Adarsh |

## Queued (not yet proposed)

Reconstruction extensions split into three orthogonal axes — different
backend, different execution location, different input modality.

| Change | Axis |
|---|---|
| `worldscan-v2.1-splat` | new backend (Gaussian Splatting) |
| `worldscan-v2.2-colmap` | new backend (classical COLMAP+MVS, CPU-friendly) |
| `worldscan-v2.3-depth-fusion` | new backend (monocular depth + TSDF) |
| `worldscan-v2.4-cloud-reconstruction` | new execution location (offload heavy work to Modal / Runpod / AWS GPUs) |
| `worldscan-v2.5-stream-reconstruction` | new input modality + dynamic scene support (WebRTC / HLS ingest, incremental + ego-vs-object-motion separation) |
| `worldscan-v3-object-segmentation` | Tier 2 semantics (SAM 2 + CLIP, per-object hulls) |
| `worldscan-v3.1-deploy` | hosting (Inngest Cloud, managed Postgres, auth, multi-user) |

## House style

- SHALL in Requirements; lowercase narrative in Scenarios.
- Implementation specifics (class names, SQL) live in `design.md`, not in
  Requirements.
- Scenarios are testable. If you can't picture verifying it, rewrite.
- No emojis in spec files. ASCII checkmarks (`[ ]` / `[x]`) only.
