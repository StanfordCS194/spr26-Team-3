# worldscan-v3-rl-authoring — Tasks

Sequenced as small PRs, each one leaving the product in a runnable state.

## PR-1 — Task data model + ABC

- [ ] Alembic migration: `task`, `task_version` tables; `policy.task_version_id` FK.
- [ ] `backend/src/rl/task_abc.py`: the `Task` abstract base, obs/action space helpers, validate signature.
- [ ] `backend/src/features/tasks/router.py`: CRUD endpoints `POST /api/projects/{id}/tasks`, `GET .../tasks`, `GET .../tasks/{tid}`, `PATCH` for NL field edits.
- [ ] Pause-stage: Train still uses the hardcoded task (existing behavior) — no UI change yet.

## PR-2 — Codegen via Inngest

- [ ] `task/codegen_requested` Inngest event + function (`backend/src/features/tasks/inngest_functions.py`).
- [ ] Claude API client (`anthropic` Python SDK already pinned in Haleum — copy the pattern).
- [ ] Prompt template that injects mesh bounds, spawn region, NL fields, optional goal_3d.
- [ ] `task.status` flips drafting → generating → ready/failed via the same generic `mark_row_failed` handler.
- [ ] `backend/src/rl/task_runtime.py`: restricted `exec()` loader, timeout wrapper, validator that the loaded module defines `GeneratedTask(Task)`.

## PR-3 — Task screen (frontend)

- [ ] New route `/p/$projectId/task` between Build and Train in `StepNav`.
- [ ] Three textareas (Objective / Environment / Agent) with placeholder examples.
- [ ] "Generate" button → fires `POST /tasks`, then polls `task.status`.
- [ ] Generated code shown in a read-only monaco editor underneath; "Edit" toggles writable.
- [ ] Goal placement: click in the persistent right-panel MeshViewer → ray-cast to floor plane → write `goal_3d` to the task row.
- [ ] Stage gating in `useProjectState`: Train unlocks only when `task.status === ready` AND task has been dry-run validated.

## PR-4 — Train consumes the generated task

- [ ] `backend/src/features/training/service.py` loads `task_version.code` via `task_runtime`, instantiates `GeneratedTask`, hands it to PPO.
- [ ] Drop the hardcoded `NavToGoal` task (the previous one moves to `backend/src/rl/legacy_tasks.py`, kept as a reference example for the LLM prompt).
- [ ] Pre-flight dry run: instantiate task, call `reset()` + 10 `step()`s before the PPO worker spins up. Failures surface in the UI; Train aborts.
- [ ] `policy.task_version_id` written when training starts.

## PR-5 — Replay in the 3D mesh

- [ ] `MeshViewer.tsx`: expose an `overlay` slot via a ref; add `goal`, `trajectory`, `currentStep` props.
- [ ] `MeshPlayback.tsx` (new): timeline slider, play/pause, episode picker. Lives in `ProjectRightPanel`.
- [ ] Remove `TrajectoryViewer.tsx` (the 2D top-down chart) once parity is reached. Keep its episode-failure-class chips — they move into `MeshPlayback`.
- [ ] Replay screen left column shrinks to: run history + per-episode stats. The mesh is the protagonist.

## PR-6 — Polish + safeguards

- [ ] Cost telemetry per codegen request (log to Inngest event for now).
- [ ] "Re-generate" button preserves the NL fields, creates a new `task_version`, leaves prior versions intact.
- [ ] Diff view between two task versions (monaco diff editor).
- [ ] Empty-state copy on the Task screen: an example NL block users can paste-and-tweak.
- [ ] E2E test: scan → build → author task ("reach the chair") → train (small step budget) → replay → assert at least one episode renders in the 3D viewer.

## Out of scope (do not start in this change)

- Multi-agent tasks.
- Reward shaping via reference demonstrations.
- Cross-project task templates.
- Allowing user-installed Python packages inside the sandbox.
