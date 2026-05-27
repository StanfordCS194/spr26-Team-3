# video-reconstruction — capability delta

## ADDED Requirements

### Requirement: Video → mesh pipeline

The system SHALL accept a video file as the source of geometry for a project, extract frames from it, and produce a triangle mesh of the captured scene.

#### Scenario: User uploads a short room video

- **WHEN** the user uploads an `.mp4` or `.mov` ≤ 200 MB via multipart POST to `/api/projects/{id}` (or via project creation)
- **THEN** the system samples frames at a configurable rate (default 4 fps, capped at 32 frames total) using `ffmpeg-python`
- **AND** persists frames under `data/projects/<id>/frames/`
- **AND** writes a thumbnail PNG from frame 0 at `data/projects/<id>/thumbnail.png`
- **AND** sets `project.video_path` and `project.thumbnail_path`

#### Scenario: User can still use a single still image (legacy)

- **GIVEN** the user has only a single photo
- **WHEN** they upload a `.jpg`/`.png`
- **THEN** the system treats it as a one-frame "video" and runs the same pipeline
- **AND** the validation gate is expected to surface a `floor_detected` warning for the resulting 2.5D shell

### Requirement: Pluggable reconstruction backends

The reconstruction stage SHALL be implemented behind a stable interface so multiple techniques can be added without changes to the FastAPI routes, the React UI, or downstream stages.

#### Scenario: Backend list is queryable

- **WHEN** the frontend calls `GET /api/reconstruction/backends`
- **THEN** the response is a list of `{name, implemented, requires_gpu}`
- **AND** only `vggt` has `implemented: true` in this release
- **AND** the UI shows unimplemented backends as disabled options with a tooltip naming the future change that will implement them

#### Scenario: A backend can be added without touching upstream/downstream code

- **GIVEN** a new file `backend/src/features/reconstruction/backends/<new>.py`
- **WHEN** the file defines a `ReconstructionBackend` subclass with a `name` attribute and is decorated with `@register`
- **THEN** the backend appears in `GET /api/reconstruction/backends` with `implemented: true` (or `false` if explicitly stubbed)
- **AND** no changes to `router.py`, `service.py`, the Inngest function, or any frontend code are required

### Requirement: VGGT reconstruction backend

The system SHALL provide a working `vggt` reconstruction backend that takes sampled frames and produces a triangle mesh and dense point cloud.

#### Scenario: Successful run on a real video

- **GIVEN** a project with uploaded video and frames extracted
- **WHEN** the user calls `POST /api/projects/{id}/reconstruct` with body `{"backend": "vggt"}`
- **THEN** an Inngest event `reconstruction/requested` is emitted
- **AND** the `reconstruction` row is created with `status='pending'`
- **AND** the response is `202 Accepted` with `reconstruction_id` and `inngest_run_id`
- **AND** once the Inngest function completes, the backend writes `mesh.ply`, `point_cloud.ply`, and `meta.json` under `data/projects/<id>/reconstruction/`
- **AND** the `reconstruction` row transitions to `status='ok'` with `elapsed_s` populated

#### Scenario: Reconstruction fails gracefully

- **GIVEN** a corrupted or unprocessable video
- **WHEN** the Inngest function raises an exception (after configured retries)
- **THEN** the `reconstruction` row is `status='failed'` with `error` set to the exception message
- **AND** the UI surfaces the error on the Reconstruct screen with a "Retry" button
- **AND** no orphan partial files remain on disk

### Requirement: Asynchronous job execution

Reconstruction MUST run as a durable Inngest function so the React app never blocks on it and survives navigation away from the Reconstruct screen.

#### Scenario: User navigates away mid-reconstruction

- **GIVEN** a reconstruction Inngest run is in progress
- **WHEN** the user clicks a different project in the sidebar
- **THEN** the job continues independently
- **AND** returning to the project shows the current job progress, sourced from Inngest run events
- **AND** when the job completes, the project's status pill updates on whichever screen the user is currently on

#### Scenario: User can re-run with different parameters

- **GIVEN** a project with a completed reconstruction
- **WHEN** the user changes params (frame count, confidence threshold) and re-runs
- **THEN** a new `reconstruction` row is created (the prior row is preserved for comparison)
- **AND** prior validation rows are unaffected and remain queryable

### Requirement: Frontend subscribes to progress events

The Reconstruct screen SHALL show live progress without manual polling, driven by Inngest events emitted from the worker.

#### Scenario: Progress events drive the progress bar

- **GIVEN** an active reconstruction Inngest run
- **WHEN** the function calls `step.send_event("reconstruction/progress", {progress: 0.4, message: "running VGGT"})`
- **THEN** the frontend (via `@inngest/react` subscription) updates the progress bar within 500 ms
- **AND** the message text appears beneath the bar
