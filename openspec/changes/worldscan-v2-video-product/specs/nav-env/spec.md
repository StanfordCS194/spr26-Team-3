# nav-env — capability delta

## MODIFIED Requirements

### Requirement: Build endpoint moved behind the project API

The HTTP entry point for building an MJCF from a mesh SHALL move from `rl_env.server`'s `/api/build` to FastAPI's `POST /api/projects/{id}/build` and MUST consume the project's latest reconstruction by default. Building remains the responsibility of the `rl_env.build` Python library.

#### Scenario: Build implicitly uses the project's latest reconstruction

- **GIVEN** a project with a successful, validated reconstruction
- **WHEN** the user calls `POST /api/projects/{id}/build` with no `reconstruction_id` in the body
- **THEN** the backend resolves the most recent `reconstruction` row with `status='ok'`
- **AND** writes the new MJCF + hulls under `data/projects/<id>/build/`
- **AND** inserts a `build` row referencing both `project_id` and `reconstruction_id`

#### Scenario: Build can be re-run with different params on the same mesh

- **GIVEN** an existing build
- **WHEN** the user opens Advanced Settings and changes `max_hulls` or `target_diagonal_m` and clicks Build again
- **THEN** a new `build` row is created (the prior is preserved for comparison)
- **AND** the new MJCF lands under `data/projects/<id>/build/<build_id>/scene.xml`

#### Scenario: Build is synchronous

- **GIVEN** build typically completes in < 1 second on the procedural sample room and < 5 seconds on real meshes
- **WHEN** the user calls `POST /api/projects/{id}/build`
- **THEN** the request handler runs synchronously (no Inngest event)
- **AND** the response includes the full `build` row plus `bounds`, `spawn_region`, `n_hulls` so the frontend can render the floor view immediately
