# rl-authoring — capability delta (NEW)

## ADDED Requirements

### Requirement: Users author RL tasks in natural language

The Task screen SHALL accept three independent natural-language inputs — Objective, Environment constraints, Agent constraints — and persist them on a per-project `task` row keyed to a specific `build_id`.

#### Scenario: Drafting a task

- **GIVEN** a project with a successful build (`build.status === "ok"`)
- **WHEN** the user opens the Task screen and types
  - Objective: "the agent must reach the chair and stop within 30 cm of it"
  - Environment: "the floor is the only walkable surface; doors are closed"
  - Agent: "30 cm radius circle, can move forward/back and turn, observes a 64×64 depth image and its own velocity"
- **AND** clicks Generate
- **THEN** a `POST /api/projects/{id}/tasks` request is sent with those three fields plus any `goal_3d` already placed in the mesh
- **AND** a new `task` row is created with `status = "drafting" → "generating"` and a `task/codegen_requested` Inngest event is emitted

### Requirement: The backend generates a runnable `Task` module via an LLM

The system SHALL translate the three NL fields into a Python module that conforms to the `Task` ABC, store the code in `task_version.code`, and flip the task to `status = "ready"` only after the generated code imports cleanly under the restricted runtime AND a smoke `reset() + 10×step()` succeeds.

#### Scenario: Successful codegen

- **GIVEN** a `task` row in `status = "generating"`
- **WHEN** the `generate_task` Inngest function runs
- **THEN** it calls the Claude API with a system prompt containing the `Task` ABC + the mesh bounds/spawn region + the three NL fields
- **AND** writes the raw response to `task.generated_code` and to a new `task_version` (`created_by = "ai"`)
- **AND** loads the code via `task_runtime.load()` inside a restricted exec namespace with no `open` / `os` / `import` / network builtins
- **AND** runs a dry-run `reset()` + 10 `step()`s with a no-op policy
- **AND** flips `task.status` to `"ready"` if all checks pass; otherwise `"failed"` with the traceback in `task.error`

#### Scenario: Generated code raises during dry-run

- **GIVEN** the LLM returns code where `reward()` references an attribute that doesn't exist on `mj_data`
- **WHEN** the dry-run executes step 1
- **THEN** the task row is marked `status = "failed"` with the traceback truncated to 1000 chars in `task.error`
- **AND** the Task screen surfaces the error inline below the code editor
- **AND** the Generate button re-enables so the user can re-prompt without losing the NL fields

### Requirement: Generated code is reviewable and editable

The Task screen SHALL display the generated Python module in a monaco editor, default read-only, with an "Edit" toggle. User edits create a new `task_version` row (`created_by = "user"`); the original AI version is preserved.

#### Scenario: User tweaks the reward

- **GIVEN** a task with `status = "ready"` and one `task_version` (AI-generated)
- **WHEN** the user clicks Edit, changes the reward weight from `1.0` to `2.0`, and clicks Save
- **THEN** a new `task_version` is written with `created_by = "user"`
- **AND** `task.current_version_id` updates to the new row
- **AND** the dry-run runs again on the edited code before `status` returns to `"ready"`

### Requirement: Goal placement happens in the 3D mesh viewer

The persistent right-panel MeshViewer SHALL accept clicks during the Task stage, ray-cast to the floor plane, and write the hit point to `task.goal_3d`.

#### Scenario: Click to place the goal

- **GIVEN** the user is on the Task screen and no `goal_3d` is set yet
- **WHEN** they click on the mesh at a point above the floor plane
- **THEN** the click is ray-cast from the camera through the cursor against the floor plane (Y = bounds.min.y)
- **AND** the hit point is sent via `PATCH /api/projects/{id}/tasks/{tid}` as `{goal_3d: {x, y, z, radius: 0.3}}`
- **AND** a draggable sphere marker is rendered at that point in the MeshViewer
- **AND** the codegen prompt on next Generate includes the goal coordinates so the LLM can reference them in the task

#### Scenario: Drag to relocate

- **GIVEN** a goal marker exists at point A
- **WHEN** the user drags the marker to point B
- **THEN** the drag continuously raycasts to the floor plane
- **AND** on mouseup, `goal_3d` is updated via PATCH
- **AND** any cached generated code is marked stale (`task.status` flips back to `"drafting"`) so the user is prompted to re-Generate

### Requirement: Tasks are versioned independently from policies

Every `policy` row SHALL store a `task_version_id` foreign key so a trained policy is always reproducible against the exact task it was trained on, even after the user edits the task.

#### Scenario: Re-training after editing the reward

- **GIVEN** a `policy_v1` trained against `task_version_a`
- **WHEN** the user edits the reward, creating `task_version_b`, and trains again
- **THEN** the new `policy_v2.task_version_id = task_version_b.id`
- **AND** the Replay screen labels each policy with its task version
- **AND** comparing v1 and v2 surfaces a "tasks differ" banner so users don't draw invalid conclusions

### Requirement: The codegen runtime is sandboxed

The `task_runtime.load(code)` function SHALL execute generated code in a namespace where `open`, `os`, `subprocess`, `socket`, `eval`, `exec`, and `__import__` are unavailable, and every call into the generated `Task` methods SHALL be wrapped in a 10 ms timeout.

#### Scenario: Generated code attempts disallowed import

- **GIVEN** an LLM that returns `import os; os.system("rm -rf /")`
- **WHEN** `task_runtime.load()` exec()s the module
- **THEN** a `NameError` (or equivalent) is raised because `__import__` is not in the namespace
- **AND** the task is marked failed with that error message
- **AND** no filesystem side effects occur

#### Scenario: Reward function hangs

- **GIVEN** generated code where `reward()` runs an infinite loop
- **WHEN** PPO calls `task.reward(...)` during training
- **THEN** the call returns within 10 ms (timeout) with a `TimeoutError`
- **AND** the training job marks itself failed with "reward timeout" via the existing `mark_row_failed` handler
