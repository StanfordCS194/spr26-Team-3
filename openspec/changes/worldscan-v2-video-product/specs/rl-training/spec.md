# rl-training — capability delta

## MODIFIED Requirements

### Requirement: Training surfaces live metrics via Inngest events

PPO training SHALL stream interim metrics to the UI via Inngest progress events so the user sees reward, success rate, and FPS update during the run rather than only at the end.

#### Scenario: Live reward curve

- **GIVEN** a training job started from the Train screen via `POST /api/projects/{id}/train`
- **WHEN** the `train_policy` Inngest function calls `step.send_event("training/progress", {progress, current_reward, current_success_rate, fps})` every N=1000 PPO steps
- **THEN** the frontend (subscribed via `@inngest/react`) updates the `MetricsChart` series within 500 ms of each event
- **AND** the chart shows reward, success rate, and episode length as three independent series

### Requirement: Per-episode failure classification

The system SHALL classify every evaluation episode into one of the categories `success`, `timeout`, `stuck`, `collided`, `near-miss`, with the classification rule documented in `backend/src/features/replay/service.py`.

#### Scenario: User sees why episodes failed

- **GIVEN** an evaluation of 30 episodes with 22 successes
- **WHEN** the user opens the Replay screen
- **THEN** the 8 failures are grouped by category
- **AND** each category shows a count and a representative trajectory preview
- **AND** clicking a category expands to show all trajectories in that category, ordered by severity

### Requirement: Baselines are first-class runs

The Replay screen SHALL always show random, greedy, and the trained PPO together unless the user opts out, and baselines MUST be stored as `run` rows with `baseline ∈ {'random','greedy'}` (and `policy_id=NULL`) so they appear in history alongside PPO runs.

#### Scenario: Trained policy is within noise of the heuristic

- **GIVEN** a benchmark on the procedural sample room where PPO scores 27/30 and greedy scores 28/30
- **WHEN** the user views Replay
- **THEN** both runs are displayed side-by-side with their numbers
- **AND** a hint banner surfaces: "PPO is statistically tied with greedy on this scene. To widen the gap, try a denser room or larger start-goal distance."

### Requirement: Training is resumable across worker restarts

PPO training MUST be resumable across worker restarts by chunking the PPO loop into `step.run`-wrapped blocks so Inngest can memoize completed chunks and re-invoke from the last successful boundary.

#### Scenario: Worker container restarts mid-training

- **GIVEN** a `train_policy` Inngest run in progress with PPO chunked into 20k-step blocks
- **WHEN** the worker container is killed and restarted
- **THEN** Inngest re-invokes the function from the last successful `step.run` boundary
- **AND** prior chunk results are memoized; only the in-flight chunk re-runs
- **AND** the final policy.zip is identical (deterministic seed) to a run that didn't restart
