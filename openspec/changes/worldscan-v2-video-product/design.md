# Design — worldscan-v2-video-product

## Stack at a glance

```
┌──────────────────────────────────────────────────────────────────┐
│ frontend/  React 19 + Vite + TS                                 │
│   TanStack Router · TanStack Query · TanStack Form               │
│   Radix UI primitives + shadcn pattern · Tailwind v4             │
│   OpenAPI-generated typed client (consumes shared/)              │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP (typed via OpenAPI)
┌────────────────────────────┴─────────────────────────────────────┐
│ backend/  FastAPI + Pydantic v2                                  │
│   SQLAlchemy 2.0 + Alembic           → Postgres                  │
│   Inngest Python SDK (functions)     → Inngest Dev Server / Cloud│
│   calls rl_env.* directly (in-process Python)                    │
└────────────────────────────┬─────────────────────────────────────┘
                             │
            ┌────────────────┴──────────────────┐
            │ worker.py — separate process      │
            │   Inngest function handlers       │
            │   reconstruction, training, eval  │
            └───────────────────────────────────┘
```

Everything in one `docker compose up`: `postgres`, `inngest-dev`, `api`,
`worker`, `frontend`. Five containers, one command.

## Repository layout (post-change)

```
spr26-Team-3/
├── frontend/                              React + Vite + TS
│   ├── src/
│   │   ├── routes/                       (TanStack Router file-based)
│   │   │   ├── __root.tsx
│   │   │   ├── index.tsx                 ProjectList
│   │   │   ├── p.$projectId.capture.tsx
│   │   │   ├── p.$projectId.reconstruct.tsx
│   │   │   ├── p.$projectId.validate.tsx
│   │   │   ├── p.$projectId.build.tsx
│   │   │   ├── p.$projectId.train.tsx
│   │   │   └── p.$projectId.replay.tsx
│   │   ├── components/
│   │   │   ├── Sidebar.tsx
│   │   │   ├── StepNav.tsx
│   │   │   ├── MeshViewer.tsx            (react-three-fiber + drei)
│   │   │   ├── TrajectoryViewer.tsx
│   │   │   ├── ValidationReport.tsx
│   │   │   ├── MetricsChart.tsx          (recharts)
│   │   │   └── ui/                        (shadcn-generated)
│   │   ├── lib/
│   │   │   ├── api.ts                    (TanStack Query hooks)
│   │   │   ├── client.ts                 (openapi-fetch instance)
│   │   │   └── inngest.ts                (subscribe to job events)
│   │   ├── instrument.ts                 (Sentry + Posthog init)
│   │   └── main.tsx
│   ├── components.json                   (shadcn config)
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   └── package.json
│
├── backend/                               FastAPI app
│   ├── src/
│   │   ├── app.py                        (FastAPI factory)
│   │   ├── worker.py                     (Inngest worker entrypoint)
│   │   ├── config.py                     (pydantic-settings)
│   │   ├── instrument.py                 (Sentry init)
│   │   ├── db.py                         (SQLAlchemy engine, session)
│   │   ├── deps.py                       (FastAPI dependencies)
│   │   ├── inngest_client.py             (Inngest client + function registry)
│   │   ├── models/                       (SQLAlchemy declarative models)
│   │   ├── schemas/                      (Pydantic request/response models)
│   │   ├── features/
│   │   │   ├── projects/
│   │   │   │   ├── router.py
│   │   │   │   └── service.py
│   │   │   ├── reconstruction/
│   │   │   │   ├── router.py
│   │   │   │   ├── service.py
│   │   │   │   ├── inngest_functions.py  (reconstruct_video)
│   │   │   │   └── backends/             (PLUGGABLE; see below)
│   │   │   │       ├── base.py
│   │   │   │       ├── __init__.py       (registry)
│   │   │   │       ├── vggt.py           (v1 implementation)
│   │   │   │       ├── splat.py          (stub)
│   │   │   │       ├── colmap.py         (stub)
│   │   │   │       └── depth_fusion.py   (stub)
│   │   │   ├── validation/
│   │   │   │   ├── router.py
│   │   │   │   ├── service.py
│   │   │   │   └── checks.py
│   │   │   ├── builds/
│   │   │   │   ├── router.py
│   │   │   │   └── service.py            (thin wrapper over rl_env.build)
│   │   │   ├── training/
│   │   │   │   ├── router.py
│   │   │   │   ├── service.py
│   │   │   │   └── inngest_functions.py  (train_policy)
│   │   │   └── replay/
│   │   │       ├── router.py
│   │   │       └── service.py
│   │   └── middleware/
│   │       ├── error_handler.py
│   │       └── request_id.py
│   ├── alembic/                          (migrations)
│   │   ├── env.py
│   │   └── versions/
│   ├── tests/                            (pytest)
│   │   ├── conftest.py
│   │   └── ...
│   ├── alembic.ini
│   ├── pyproject.toml
│   └── Dockerfile
│
├── shared/                                cross-cutting contracts
│   ├── openapi.json                      (regenerated from backend on every commit hook)
│   ├── ts/                               (openapi-typescript output, consumed by frontend)
│   └── package.json
│
├── rl_env/                                unchanged Python package
│
├── archive/legacy/                        prototype/v0–v3.html, original index.html
│   └── README.md
│
├── data/                                  gitignored runtime state
│   └── projects/<project-id>/
│       ├── manifest.json
│       ├── input.mp4
│       ├── frames/
│       ├── reconstruction/
│       │   ├── mesh.ply
│       │   ├── point_cloud.ply
│       │   └── meta.json
│       ├── validation.json
│       ├── build/
│       ├── policies/<policy-id>/
│       └── runs/<run-id>/
│
├── docker-compose.yaml
├── pyproject.toml                        (root, workspace-level Python tooling)
├── package.json                          (yarn workspaces root)
├── Justfile                              (`just dev`, `just test`, `just typegen`)
├── README.md
└── openspec/
```

## Data model (Postgres, SQLAlchemy 2.0)

Tables match Haleum's discipline: TEXT primary keys (`nanoid()`),
`created_at TIMESTAMPTZ NOT NULL DEFAULT now()`, JSONB for irregular columns,
foreign keys with `ON DELETE CASCADE` where the child has no meaning without
its parent.

```python
# backend/src/models/__init__.py (excerpt)

class Project(Base):
    id: Mapped[str] = mapped_column(primary_key=True)            # nanoid
    name: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    video_path: Mapped[str | None]
    thumbnail_path: Mapped[str | None]

class Reconstruction(Base):
    id: Mapped[str] = mapped_column(primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("project.id", ondelete="CASCADE"))
    backend: Mapped[str]                                          # 'vggt' | 'splat' | ...
    params: Mapped[dict] = mapped_column(JSONB)
    mesh_path: Mapped[str | None]
    status: Mapped[str]                                           # 'pending'|'running'|'ok'|'failed'
    error: Mapped[str | None]
    elapsed_s: Mapped[float | None]
    inngest_run_id: Mapped[str | None]                            # back-link to Inngest dashboard
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

class Validation(Base):
    id: Mapped[str] = mapped_column(primary_key=True)
    reconstruction_id: Mapped[str] = mapped_column(ForeignKey("reconstruction.id", ondelete="CASCADE"))
    report: Mapped[dict] = mapped_column(JSONB)                   # {checks: [...], overall}
    user_override: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

class Build(Base):
    id: Mapped[str] = mapped_column(primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("project.id", ondelete="CASCADE"))
    reconstruction_id: Mapped[str] = mapped_column(ForeignKey("reconstruction.id"))
    mjcf_path: Mapped[str]
    n_hulls: Mapped[int]
    bounds: Mapped[dict] = mapped_column(JSONB)
    spawn_region: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

class Policy(Base):
    id: Mapped[str] = mapped_column(primary_key=True)
    build_id: Mapped[str] = mapped_column(ForeignKey("build.id", ondelete="CASCADE"))
    algo: Mapped[str]                                             # 'ppo' | 'sac' | ...
    ckpt_path: Mapped[str]
    total_steps: Mapped[int]
    metrics: Mapped[dict] = mapped_column(JSONB)
    inngest_run_id: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

class Run(Base):
    id: Mapped[str] = mapped_column(primary_key=True)
    policy_id: Mapped[str | None] = mapped_column(ForeignKey("policy.id", ondelete="CASCADE"))
    baseline: Mapped[str | None]                                  # 'random'|'greedy'; null if PPO
    episodes: Mapped[int]
    successes: Mapped[int]
    avg_reward: Mapped[float]
    trajectories_path: Mapped[str]                                # JSON file on disk
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
```

No `jobs` table is needed — Inngest owns job state and exposes it via its API
and dashboard. We just store the `inngest_run_id` on rows where the user
might want to deep-link into the dashboard.

## Reconstruction plugins

The reconstruction stage is the only part of the backend where multiple
implementations are explicitly designed for. The interface is a Python ABC
with a registry.

```python
# backend/src/features/reconstruction/backends/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

@dataclass
class ReconstructionInput:
    frames_dir: Path
    fps_sampled: float
    intrinsics_hint: dict | None

@dataclass
class ReconstructionOutput:
    mesh_path: Path
    point_cloud_path: Path | None
    camera_poses: dict | None
    backend_meta: dict

class ReconstructionBackend(ABC):
    name: str
    requires_gpu: bool
    implemented: bool = True

    @abstractmethod
    def reconstruct(
        self,
        inp: ReconstructionInput,
        out_dir: Path,
        progress_cb: Callable[[float, str], None],
    ) -> ReconstructionOutput: ...
```

```python
# backend/src/features/reconstruction/backends/__init__.py
_BACKENDS: dict[str, type[ReconstructionBackend]] = {}

def register(cls): _BACKENDS[cls.name] = cls; return cls
def list_backends() -> list[dict]: ...                # name, implemented, requires_gpu
def get_backend(name) -> ReconstructionBackend: ...
```

Backends in this change:

| Backend | `implemented` |
|---|---|
| `vggt` | **True** |
| `splat` (gaussian splatting) | `False` — `reconstruct` raises `NotImplementedError("planned: worldscan-v2.1-splat")` |
| `colmap` | `False` |
| `depth_fusion` | `False` |

The UI calls `GET /api/reconstruction/backends`, gets the list, renders
unimplemented ones as disabled with a tooltip pointing at the future change
name. Adding a new backend later = one new file + decorator; no UI changes.

## Inngest functions

```python
# backend/src/features/reconstruction/inngest_functions.py
from backend.src.inngest_client import inngest

@inngest.create_function(
    fn_id="reconstruct-video",
    trigger=inngest.TriggerEvent(event="reconstruction/requested"),
    retries=2,
)
async def reconstruct_video(ctx, step):
    payload = ctx.event.data                          # {project_id, backend, params}
    frames = await step.run("extract-frames", extract_frames, payload)
    meta = await step.run("run-backend",      run_backend,     payload, frames)
    await step.run("persist", persist_reconstruction, meta)
    return meta
```

The frontend subscribes to `reconstruction/*` events via Inngest's React
hooks (`@inngest/react`) and updates progress in real time. No manual job
polling endpoints to build.

Three Inngest functions in v1:

| Function | Trigger event | What it does |
|---|---|---|
| `reconstruct-video` | `reconstruction/requested` | sample frames → run backend → persist mesh |
| `validate-mesh` | `validation/requested` | runs all checks, writes report |
| `train-policy` | `training/requested` | PPO loop, streams metrics via `step.send_event` |

`validate-mesh` could run synchronously since it's <1s, but routing it
through Inngest keeps every long-or-medium operation consistent and gives us
retries for free.

## Frontend ↔ backend contract

FastAPI emits an OpenAPI 3 spec at `/openapi.json`. A `just typegen` task:

```
just typegen:
    curl -s http://localhost:8000/openapi.json > shared/openapi.json
    pnpm dlx openapi-typescript shared/openapi.json -o shared/ts/api.d.ts
```

Frontend uses `openapi-fetch` for a runtime client that's automatically typed
against `shared/ts/api.d.ts`. TanStack Query hooks wrap it:

```ts
// frontend/src/lib/api.ts
import createClient from "openapi-fetch";
import type { paths } from "@worldscan/shared/ts/api";
export const api = createClient<paths>({ baseUrl: "/api" });

export const useProject = (id: string) =>
  useQuery({
    queryKey: ["project", id],
    queryFn: async () => {
      const { data, error } = await api.GET("/projects/{id}", { params: { path: { id } } });
      if (error) throw error;
      return data;
    },
  });
```

Husky pre-commit hook runs `just typegen` so the typed client never drifts.

## Validation checks (v1 catalog)

Each check returns `{status: 'pass'|'warn'|'fail', message: str, fix: str}`.

| Check | pass | warn | fail |
|---|---|---|---|
| watertight | `trimesh.is_watertight` true | < 5% boundary edges | ≥ 5% boundary edges |
| connected_components | == 1 | 2–3 (small islands) | ≥ 4 |
| bbox_plausibility | longest dim 1–30 m | 0.5–1 m or 30–100 m | < 0.5 m or > 100 m |
| floor_detected | largest horizontal patch ≥ 1 m² at bottom | < 1 m² | none |
| convex_decomp_quality | hulls in [3, 64] **and** volume preservation ≥ 0.7 | hulls < 3 or vol [0.5, 0.7) | vol < 0.5 |
| scale_calibration | metric source flag set | unitless, will rescale | aspect ratio absurd |

"Build env" is disabled if any check is `fail` and the override checkbox is
unticked. Warnings don't block.

## Sidebar / project model

A project is the unit of work. Sidebar lists all projects newest-first with
thumbnail (first frame of the video), inline-editable name, status pill, and
last-modified time. Status is derived: furthest stage with a non-failed
artifact.

Clicking a project navigates to whichever step the user was last on,
inferred from artifact existence: `train` if a policy exists, `validate` if a
reconstruction exists, otherwise `capture`.

## Local dev story

```yaml
# docker-compose.yaml (excerpt)
services:
  postgres:
    image: postgres:17
    ports: ["5432:5432"]
    volumes: ["pgdata:/var/lib/postgresql/data"]
  inngest-dev:
    image: inngest/inngest:latest
    command: ["inngest", "dev", "-u", "http://api:8000/api/inngest"]
    ports: ["8288:8288"]
  api:
    build: ./backend
    command: ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    ports: ["8000:8000"]
    depends_on: [postgres]
    volumes: ["./backend:/app", "./rl_env:/app/rl_env", "./data:/data"]
  worker:
    build: ./backend
    command: ["python", "-m", "src.worker"]
    depends_on: [postgres, inngest-dev]
    volumes: ["./backend:/app", "./rl_env:/app/rl_env", "./data:/data"]
  frontend:
    build: ./frontend
    command: ["pnpm", "dev"]
    ports: ["5173:5173"]
    volumes: ["./frontend:/app", "./shared:/shared"]
```

`just dev` runs `docker compose up --watch`. Vite proxies `/api` to the api
container, and `@inngest/react` connects to the inngest-dev container.

## Visual direction

Reference: `prototype/v3.html`'s 3D viewer (dark canvas, faint grid floor,
top-right monospace control hints, bottom status bar with colored status
dots). Aim for a Linear / Blender / Three.js-editor feel — a tool, not a
consumer product.

Tokens locked in PR-A's `frontend/src/index.css` (shadcn theme overrides):

| Token | Value | Notes |
|---|---|---|
| `--background` | `oklch(0.08 0 0)` | near-black canvas |
| `--foreground` | `oklch(0.95 0 0)` | near-white text |
| `--card` | `oklch(0.11 0 0)` | side panels, slightly lifted |
| `--border` | `oklch(0.18 0 0)` | hairline 1 px |
| `--primary` | `oklch(0.72 0.18 230)` | electric blue accent |
| `--muted-foreground` | `oklch(0.55 0 0)` | secondary text |
| status: ok | `oklch(0.72 0.18 145)` | green dot |
| status: warn | `oklch(0.78 0.16 80)` | amber dot |
| status: fail | `oklch(0.65 0.22 25)` | red dot |
| `--radius` | `2px` | sharp; aligns with tooling feel |

Fonts:

- **Sans (UI):** Inter Variable (via `@fontsource-variable/inter`).
- **Mono (technical surfaces — IDs, control hints, code, numeric tables):**
  JetBrains Mono Variable (via `@fontsource-variable/jetbrains-mono`).

The 3D viewers (`MeshViewer.tsx`, `TrajectoryViewer.tsx`) keep their own
grid + dark canvas independent of the shadcn theme.

If the user provides additional example UIs later, PR-A's theme tokens are
the only place that needs to change — components consume tokens, not raw
colors.

## Decisions table

| Decision | Choice | Why |
|---|---|---|
| Frontend framework | React 19 + Vite + TS | Matches Haleum |
| Frontend routing | TanStack Router (file-based) | Matches Haleum; typed routes |
| Frontend server state | TanStack Query | Matches Haleum |
| Frontend forms | TanStack Form | Matches Haleum |
| Component primitives | Radix + shadcn | Matches Haleum (`components.json`) |
| Styling | Tailwind v4 via `@tailwindcss/vite` | Matches Haleum |
| Backend framework | FastAPI | Pydantic ≈ Zod; auto-OpenAPI ≈ Express+Zod codegen; matches Haleum's typed-bodies discipline |
| ORM | SQLAlchemy 2.0 + Alembic | Closest Python analog to Drizzle; mature, typed, migrations baked in |
| Database | Postgres | Matches Haleum; JSONB; one Docker line |
| Jobs | Inngest (Python SDK) | Durable, retries, dashboards; matches Haleum exactly |
| Shared types | OpenAPI → openapi-typescript | Single source of truth, prevents drift |
| Monorepo tool | yarn workspaces | Matches Haleum |
| Lint / format (TS) | ESLint + Prettier + Husky | Matches Haleum |
| Lint / format (Py) | Ruff + Black | FastAPI / Pydantic community default |
| Old prototype HTMLs | Move to `archive/legacy/`, don't delete | Honor prior work, keep demo available |
| `rl_env/` | Used as-is, library role | It works; UI churn shouldn't touch the engine |
| Reconstruction backend (v1) | VGGT | Best quality-per-effort for short videos; designed for swapping |
| Job runner location | Separate `worker.py` process | Matches Haleum's `worker.ts` split; prevents API requests blocking on training |

## Risks and mitigations

- **VGGT availability.** Model weights are on HuggingFace; license check
  happens in PR-B before any code is written against them. Fallback: MASt3R
  (same interface, same input/output shape).
- **GPU requirement.** VGGT needs CUDA or recent Apple-Metal. CI cannot run
  reconstruction. Mitigation: ship a `tests/fixtures/precomputed_mesh.ply`
  so the validate/build/train path is testable without GPU.
- **Inngest learning curve.** Two engineers on the team don't know Inngest.
  Mitigation: Inngest's Python tutorial takes <1 hour; we use exactly three
  functions; the Dev Server's dashboard makes debugging visual rather than
  log-diving.
- **Mesh-quality unknowns.** Real videos may still produce ugly meshes. The
  validation gate is the safety net — we surface "convex decomp failed,
  here's why" instead of training on garbage.
- **Scope creep.** Tempting to do segmentation in this change. Don't.

## Sequencing — cleanup vs feature

**Cleanup first.** PR-A is pure restructure: move legacy HTML to
`archive/legacy/`, scaffold `frontend/`, `backend/`, `shared/`,
`docker-compose.yaml`, Postgres + Inngest Dev Server running, the
procedural-sample-room flow working through FastAPI + the new React UI. No
new product features. PR-A is the foundation every other PR builds on.
