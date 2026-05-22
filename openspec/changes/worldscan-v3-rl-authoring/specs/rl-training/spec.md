# rl-training — capability delta (MODIFIED)

## MODIFIED Requirements

### Requirement: Training consumes a user-authored Task

PPO training SHALL load the active `task_version.code` for the project, instantiate `GeneratedTask` via the sandboxed runtime, and use it as the RL environment's task. The hardcoded `NavToGoal` task is removed from the training path.

#### Scenario: Train unlocks only after a ready task

- **GIVEN** a project with a successful build but no `task` row in `status = "ready"`
- **WHEN** the user navigates to the Train screen
- **THEN** Train is locked with the reason "Author a task on the Task stage first"
- **AND** the unlock CTA navigates to `/p/$projectId/task`

#### Scenario: Train uses the current task version

- **GIVEN** a task with `current_version_id = task_version_b`
- **WHEN** the user clicks Start Training
- **THEN** the `train-policy` Inngest function loads `task_version_b.code` via `task_runtime.load()`
- **AND** the resulting `policy` row has `task_version_id = task_version_b.id`
- **AND** the training progress events include `task_version_id` so the UI can label the live reward curve with its task

### Requirement: Training surfaces a pre-flight dry run

Before allocating PPO workers, the system SHALL execute one full `reset() + 10×step()` cycle against the generated task and fail fast if any call raises.

#### Scenario: Dry run catches a broken observation builder

- **GIVEN** generated code where `observe()` returns a tensor whose shape doesn't match the declared `obs_space`
- **WHEN** the user clicks Start Training
- **THEN** the dry run detects the shape mismatch
- **AND** the policy row is marked `failed` with the shape mismatch error in `policy.metrics.error`
- **AND** no PPO worker is launched (saves the 30+ seconds of warmup before discovering the bug)

## REMOVED Requirements

### Requirement: ~~Training uses the hardcoded NavToGoal task~~

Removed. Previously training spawned the agent inside `build.spawn_region` and rewarded distance reduction to a hardcoded goal cell. That task is preserved as a reference example in `backend/src/rl/legacy_tasks.py` and shown in the codegen system prompt as a worked example, but is no longer the runtime task.
