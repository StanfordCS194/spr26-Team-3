# replay-mesh-playback — capability delta (NEW)

## ADDED Requirements

### Requirement: Replay paints trajectories into the persistent 3D MeshViewer

The Replay screen SHALL render each evaluation episode's trajectory as a 3D polyline inside the same `MeshViewer` instance used on all other stages, lifted onto the floor plane defined by the build's bounds. The legacy 2D top-down `TrajectoryViewer` is removed once parity is reached.

#### Scenario: A single episode renders as a 3D line

- **GIVEN** a completed run with `episodes[0].trajectory` containing N (x, y) points
- **WHEN** the user opens the Replay screen and selects episode 1
- **THEN** the MeshViewer in the right panel shows a 3D polyline whose vertices are `(x_i, floor_y, y_i)` (mapping the 2D top-down coordinates back onto the floor plane)
- **AND** the line color reflects the failure class (green=success, amber=timeout, red=collided, etc.)
- **AND** the existing PLY mesh is **not** unloaded — the overlay is additive

### Requirement: Episode playback is scrubbable

The Replay screen SHALL provide a timeline scrubber whose current step drives an animated agent marker (a small sphere) along the trajectory line.

#### Scenario: Scrub to step 47

- **GIVEN** the user is viewing an episode of 120 steps
- **WHEN** they drag the timeline slider to step 47
- **THEN** the agent sphere snaps to position `trajectory[47]` lifted onto the floor plane
- **AND** the trajectory line ahead of the marker is rendered with 40% opacity (preview)
- **AND** the path behind the marker stays at full opacity (history)

#### Scenario: Play / pause animation

- **GIVEN** a paused timeline at step 0
- **WHEN** the user clicks Play
- **THEN** the marker advances one step per ~33 ms (≈30 FPS) until the end
- **AND** the slider position updates in lockstep
- **AND** clicking Play again pauses; the user can change playback speed (0.5×, 1×, 2×, 4×)

### Requirement: Multi-episode comparison via overlay

The Replay screen SHALL allow the user to overlay multiple episodes simultaneously, each with its own color, to make pattern-recognition obvious (e.g. "all the failures cluster on the left side of the room").

#### Scenario: Overlay all 30 episodes

- **GIVEN** a 30-episode run
- **WHEN** the user clicks "Show all episodes"
- **THEN** all 30 trajectory polylines are rendered in the MeshViewer
- **AND** the color encodes failure class
- **AND** hovering a line dims the others to 20% opacity and shows that episode's stats in a tooltip

### Requirement: The goal marker stays visible during replay

The goal marker placed during Task authoring SHALL remain rendered in the MeshViewer on the Replay screen, so the user can visually verify that the agent's trajectories converge toward (or fail to reach) the intended goal.

#### Scenario: Goal renders alongside trajectories

- **GIVEN** a task with `goal_3d` set
- **WHEN** the user opens Replay
- **THEN** the goal marker (sphere with a faint radius ring) renders at `goal_3d`
- **AND** trajectories that ended within the goal's radius are tinted green; outside, red/amber per failure class

### Requirement: Comparing two policies trained on different task versions surfaces the divergence

When the user selects two policies for comparison and they were trained against different `task_version_id`s, the Replay screen SHALL display a banner indicating the task versions differ, with a "View diff" link to the monaco diff of the two task modules.

#### Scenario: Comparing v1 (AI-generated) vs v2 (user-edited)

- **GIVEN** `policy_v1.task_version_id = a`, `policy_v2.task_version_id = b`, where `b` is the user-edited variant of `a`
- **WHEN** the user picks both for side-by-side replay
- **THEN** a banner reads: "These policies were trained on different task versions — direct comparison may be misleading."
- **AND** clicking "View diff" opens a monaco diff modal of `a.code` vs `b.code`
