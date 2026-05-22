# project-history — capability delta

## ADDED Requirements

### Requirement: Project as the unit of work

The system SHALL create a Project for every video uploaded, and all downstream artifacts (reconstruction, validation, build, policy, run) MUST hang off it through Postgres foreign keys with `ON DELETE CASCADE`.

#### Scenario: Listing projects

- **WHEN** the user loads the app at `/`
- **THEN** the left sidebar renders all projects newest-first
- **AND** each row shows thumbnail, name (inline-editable), status pill, and created-at timestamp
- **AND** the data comes from a single `GET /api/projects` request that joins child tables (no N+1)

#### Scenario: Status pill reflects furthest completed stage

- **GIVEN** a project with a successful reconstruction but no build
- **THEN** the status pill reads "Reconstructed"
- **GIVEN** a project with a trained policy
- **THEN** the status pill reads "Trained — N% success" where N is the most recent run's success rate

### Requirement: Resume from last-visited step

The system SHALL navigate the user to the furthest non-failed stage of a project when they open it, so closing and reopening the app preserves work-in-progress flow.

#### Scenario: Closing and reopening a project

- **GIVEN** the user opened a project, advanced to Train, then closed the app
- **WHEN** they click that project in the sidebar
- **THEN** the backend computes the furthest stage with a non-failed artifact (`policy → train`, `build → build`, `reconstruction → validate`, `video → capture`)
- **AND** the frontend navigates to that route, not back to Capture

### Requirement: Run history per policy

The system SHALL persist every evaluation of a trained policy as a `run` row so users can review and compare runs across sessions.

#### Scenario: Comparing two runs

- **GIVEN** a policy with three saved runs at different seeds
- **WHEN** the user selects two runs in the run-history panel and clicks "Compare"
- **THEN** both runs' trajectories render in the same `TrajectoryViewer` with distinct colors
- **AND** a summary table shows the per-run success rate, avg reward, and avg steps

### Requirement: Project export and delete

The system MUST support exporting a project as a self-contained zip bundle and deleting a project (with confirmation) so users can hand work off or clean up.

#### Scenario: Exporting

- **WHEN** the user clicks "Export" on a project's context menu
- **THEN** `POST /api/projects/{id}/export` streams a zip of `data/projects/<id>/`
- **AND** the bundle is self-contained (mesh + scene.xml + policy.zip + trajectories) so a teammate can unzip and inspect without running the backend

#### Scenario: Deleting

- **WHEN** the user clicks "Delete" and types the project name to confirm
- **THEN** `DELETE /api/projects/{id}` removes the row
- **AND** Postgres cascades remove all child rows
- **AND** the backend removes `data/projects/<id>/` from disk
- **AND** the sidebar updates immediately via TanStack Query invalidation

### Requirement: Inspectable on-disk layout

The on-disk layout under `data/projects/<id>/` SHALL be human-readable and self-describing so a teammate can inspect a project without DB access.

#### Scenario: A teammate browses to `data/projects/`

- **WHEN** they navigate the directory tree
- **THEN** every project is a single folder under `data/projects/<id>/`
- **AND** files use human-readable names (`input.mp4`, `mesh.ply`, `scene.xml`, `policy.zip`)
- **AND** a `manifest.json` at the project root summarizes ids, paths, and status for offline inspection
