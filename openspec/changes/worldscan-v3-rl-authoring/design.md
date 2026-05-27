# worldscan-v3-rl-authoring — Design

## Overview

A new stage **Task** sits between Build and Train. Users author a task in natural language; an LLM emits a Python `Task` module; PPO consumes it; Replay paints trajectories into the 3D mesh.

```
Capture → Reconstruct → Validate → Build → ┌── Task ──┐ → Train → Replay
                                            │ author + │
                                            │ codegen  │
                                            │  + edit  │
                                            └──────────┘
```

The Task stage is the first stage that is *content-bearing on its own* — every previous stage was deterministic processing. Task is the first place a user expresses what they actually want.

## Data model

New table `task`:
```
id              str  pk
project_id      str  fk → project
build_id        str  fk → build           (binds task to a specific scene)
name            str                       short user label
objective_nl    text                      natural-language objective
env_nl          text                      natural-language env constraints
agent_nl        text                      natural-language agent constraints
goal_3d         jsonb                     { x, y, z, radius } | null
generated_code  text                      LLM output (canonical Task module)
edited_code     text | null               user edits override generated_code
status          enum: drafting | generating | ready | failed
error           text | null
codegen_model   str                       "claude-opus-4-7" etc — provenance
codegen_prompt  text                      stored prompt, for re-runs
created_at      timestamptz
```

New table `task_version` (each Generate / Edit creates a row; `task.current_version_id` points at latest):
```
id              str  pk
task_id         str  fk
code            text
created_by      enum: ai | user
created_at      timestamptz
```

`policy` gains `task_version_id` → which task version this policy was trained against. A policy is meaningless without its task; comparisons across versions must be explicit.

## Backend codegen

`backend/src/features/tasks/codegen.py`:

- One Inngest function `generate_task` (`trigger: task/codegen_requested`). Reads task row, calls Claude API with a system prompt that contains:
  - The `Task` ABC (Python interface the user code must implement).
  - The mesh metadata (bounds, spawn region, build's MJCF path — for ray-casts at validate time).
  - The three NL fields.
  - The optional 3D goal point.
- Output: a single Python module string. Must define `class GeneratedTask(Task)` with the required methods.
- Stored to `task.generated_code`, status → ready (or failed with stderr).

`backend/src/rl/task_runtime.py`:

- Loads generated code into a restricted execution context (`exec()` with a curated `__builtins__` — no `open`, `os`, `subprocess`, `socket`, `eval`, `exec`, `import`).
- Required imports (`numpy`, `mujoco` typing, our `Task` ABC) are pre-bound. Anything else fails.
- Wraps reward / termination / obs calls in 10 ms timeouts (signal-based on Linux, threaded fallback on macOS dev).

`Task` ABC (rough):
```python
class Task(ABC):
    obs_space: gym.Space          # static
    action_space: gym.Space       # static
    horizon: int                  # static

    @abstractmethod
    def reset(self, mj_data, rng) -> dict: ...        # returns obs
    @abstractmethod
    def reward(self, mj_data, action) -> float: ...
    @abstractmethod
    def terminated(self, mj_data) -> bool: ...
    @abstractmethod
    def truncated(self, mj_data, step) -> bool: ...
    @abstractmethod
    def observe(self, mj_data) -> dict: ...
    def validate(self, mjcf_path) -> list[str]: return []   # warnings
```

The LLM is told: implement exactly this; no extra public methods; use only `numpy` and our injected `mujoco_view` helpers.

## Why an AI-codegen path instead of a structured form

Considered three approaches:

1. **Structured form** (dropdowns for reward shape, sliders for weights). Pros: deterministic, no sandboxing problem. Cons: limited to tasks we anticipated. Doesn't capture "the agent must approach the chair from the front" without us pre-building that as a checkbox.
2. **DSL** (small reward expression language). Pros: safe by construction. Cons: every user has to learn it. We'd be building a new programming language and tutorials.
3. **NL → Python via LLM** (chosen). Pros: arbitrary tasks; users write English; output is reviewable code so power users can iterate. Cons: codegen quality varies; need sandboxing; LLM cost per task (~$0.05 with Sonnet at expected lengths).

We pick (3) because it's the only one that delivers on "super flexible" — and the codegen output is *Python we wrote the ABC for*, not a chat session. Inspectable, diffable, version-controlled per project.

## 3D mesh playback

The right-panel `MeshViewer` (already persistent across stages) gains an overlay layer:

- Goal marker (sphere, clickable, draggable on the floor plane during authoring).
- Trajectory line (a `THREE.Line` built from `episode.trajectory` x/y points lifted to the floor plane Z).
- An "agent dot" that animates along the line, controlled by a timeline slider in the bottom of the right panel.
- During multi-episode runs: an episode picker chip strip on top; selecting an episode swaps the line + dot.

We deliberately **do not** rebuild the viewer per stage. The same `MeshViewer` instance gets new props for `goal`, `trajectory`, `playback`. The PLY mesh itself doesn't reload.

Implementation:
- `MeshViewer.tsx` already exposes the THREE scene via a ref. We add an overlay group, cleared on prop change.
- A separate `MeshPlayback` component owns the timeline state and emits the `currentStep` index that `MeshViewer` reads.

## Why this subsumes "agent goal manipulation in the mesh"

Task #22 (open) was: drag a goal marker in the 3D scene during training setup. That was a narrow feature. Here, goal placement is one of three NL fields' worth of authoring — the click-in-mesh interaction just becomes the input mechanism for the `goal_3d` column. Solving the bigger problem solves the smaller one for free.

## Risks

- **LLM produces broken code.** Mitigation: pre-flight `Task.validate()` + a dry-run reset/step before Train unlocks. If the dry run throws, we show the traceback in the UI and ask the user to either re-generate or edit.
- **Sandbox escape.** Mitigation: restricted `exec` namespace, no `import`, no I/O. Worst case the user can write an infinite loop in `reward()` — that hits our per-call timeout.
- **Codegen cost.** Expected $0.02–$0.05 per generate. Acceptable; users won't regenerate constantly.
- **PPO with arbitrary rewards may not converge.** That's fine — it's the user's task. We surface the reward curve in Train and let them adjust the NL and re-generate.
