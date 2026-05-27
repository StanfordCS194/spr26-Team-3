# physics-validation — capability delta

## ADDED Requirements

### Requirement: Mesh sanity report

The system SHALL run a fixed catalog of checks on every reconstructed mesh before it can become an RL environment, and SHALL persist the result to the `validation` table as a structured pass/warn/fail report.

#### Scenario: All checks pass

- **GIVEN** a high-quality mesh from a multi-frame reconstruction
- **WHEN** the user calls `POST /api/projects/{id}/validate`
- **THEN** the response body is `{checks: [...6 results], overall: 'pass'}`
- **AND** a `validation` row is inserted with `report` containing the same payload
- **AND** the UI's "Build env" button is enabled

#### Scenario: A check warns

- **WHEN** a check returns `{status: 'warn', message, fix}` (e.g. 2 connected components instead of 1)
- **THEN** the report's overall status downgrades to `warn` (not `fail`)
- **AND** the warning is shown inline with its `fix` text
- **AND** the "Build env" button remains enabled
- **AND** the warning is preserved in `build.metadata` once the user proceeds

#### Scenario: A check fails

- **WHEN** any check returns `{status: 'fail', message, fix}`
- **THEN** the report's overall status is `fail`
- **AND** the "Build env" button is disabled by default
- **AND** a "Build anyway (override validation)" checkbox is rendered
- **AND** ticking it re-enables the button and stores `user_override=true` on the row that records the build's reconstruction lineage

### Requirement: Catalog of checks

The validation catalog SHALL include at minimum the six named checks `watertight`, `connected_components`, `bbox_plausibility`, `floor_detected`, `convex_decomp_quality`, and `scale_calibration`, each returning `{name, status, message, fix}`.

#### Scenario: A new check can be added

- **GIVEN** a new function in `backend/src/features/validation/checks.py` with signature `(mesh: trimesh.Trimesh) -> CheckResult`
- **WHEN** registered via the `CATALOG` dict at the bottom of the module
- **THEN** the check runs automatically on every validation request
- **AND** appears in the UI's checklist without UI code changes

### Requirement: Validation is re-runnable

The system SHALL allow validation to be re-run on the same reconstruction or on subsequent reconstructions of the same project, and MUST preserve prior reports for history.

#### Scenario: User changes reconstruction params and revalidates

- **GIVEN** a project with a failed validation
- **WHEN** the user re-runs reconstruction with new params and calls `POST /api/projects/{id}/validate` again
- **THEN** a new `validation` row is created (the prior is preserved for history)
- **AND** the UI shows the latest report
- **AND** a "previous reports" link surfaces past attempts ordered newest first

### Requirement: Validation drives diagnostic UI on failure

The validation report from the upstream reconstruction SHALL be reachable from the Replay screen with one click so the user can answer "is this a bad policy or a bad scene?" in seconds.

#### Scenario: Replay shows the scene's validation badge

- **GIVEN** a project whose validation overall status is `warn`
- **WHEN** the user opens Replay and sees 0/10 PPO success
- **THEN** a yellow badge "Scene validation: 2 warnings" is visible at the top of the Replay screen
- **AND** clicking the badge navigates to the Validate screen for the same project
