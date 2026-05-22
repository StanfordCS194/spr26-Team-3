"""Inngest function for MJCF build."""
from __future__ import annotations

import logging

import inngest

from rl_env.build import BuildConfig, build_environment
from src.config import get_settings
from src.db import SessionLocal
from src.inngest_client import inngest_client, register
from src.models import Build, Reconstruction

log = logging.getLogger(__name__)


@register
@inngest_client.create_function(
    fn_id="build-env",
    trigger=inngest.TriggerEvent(event="build/requested"),
    retries=0,
)
async def build_env(ctx: inngest.Context) -> dict:
    step = ctx.step
    payload = ctx.event.data or {}
    build_id: str = payload["build_id"]
    # `.get(key, default)` returns the default only when the KEY is missing —
    # the route always populates these from Pydantic, which writes `None` for
    # omitted fields. Use `or` to coerce None → default.
    target_diagonal_m: float | None = payload.get("target_diagonal_m")
    max_hulls: int = int(payload.get("max_hulls") or 64)
    up_axis: str = payload.get("up_axis") or "auto"

    log.info("build-env: %s", build_id)

    async def _run() -> dict:
        settings = get_settings()
        with SessionLocal() as db:
            b = db.get(Build, build_id)
            if b is None:
                raise RuntimeError(f"build {build_id} disappeared")
            b.status = "running"
            db.commit()
            recon = (
                db.get(Reconstruction, b.reconstruction_id)
                if b.reconstruction_id
                else None
            )
            project_id = b.project_id
            mesh_path = recon.mesh_path if recon and recon.mesh_path else None

        if not mesh_path:
            raise RuntimeError(
                "no reconstruction mesh for this project — run Reconstruct first"
            )

        out_dir = settings.data_dir / "projects" / project_id / "build"
        artifacts = build_environment(
            BuildConfig(
                mesh_path=mesh_path,
                out_dir=str(out_dir),
                target_diagonal_m=target_diagonal_m or settings.default_target_diagonal_m,
                up_axis=up_axis,
                max_hulls=max_hulls,
            )
        )

        with SessionLocal() as db:
            b = db.get(Build, build_id)
            assert b is not None
            b.mjcf_path = str(artifacts.mjcf_path)
            b.n_hulls = artifacts.n_hulls
            b.bounds = {
                "min": artifacts.bounds[0].tolist(),
                "max": artifacts.bounds[1].tolist(),
            }
            b.spawn_region = {
                "xmin": artifacts.spawn_region[0],
                "xmax": artifacts.spawn_region[1],
                "ymin": artifacts.spawn_region[2],
                "ymax": artifacts.spawn_region[3],
            }
            b.status = "ok"
            db.commit()
        return {"build_id": build_id, "ok": True}

    return await step.run("rl-env-build", _run)
