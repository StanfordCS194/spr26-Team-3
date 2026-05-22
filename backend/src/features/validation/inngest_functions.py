"""Inngest function for mesh validation. Fires on `validation/requested`,
runs the check catalog, persists the report to the validation row.
"""
from __future__ import annotations

import logging

import inngest

from src.db import SessionLocal
from src.features.validation.checks import run_all
from src.inngest_client import inngest_client, register
from src.models import Reconstruction, Validation

log = logging.getLogger(__name__)


@register
@inngest_client.create_function(
    fn_id="validate-mesh",
    trigger=inngest.TriggerEvent(event="validation/requested"),
    retries=0,
)
async def validate_mesh(ctx: inngest.Context) -> dict:
    step = ctx.step
    payload = ctx.event.data or {}
    validation_id: str = payload["validation_id"]
    log.info("validate-mesh: %s", validation_id)

    async def _run() -> dict:
        with SessionLocal() as db:
            v = db.get(Validation, validation_id)
            if v is None:
                raise RuntimeError(f"validation {validation_id} disappeared")
            v.status = "running"
            db.commit()
            recon = db.get(Reconstruction, v.reconstruction_id)
            if recon is None or not recon.mesh_path:
                raise RuntimeError("upstream reconstruction has no mesh")
            mesh_path = recon.mesh_path

        report = run_all(mesh_path)

        with SessionLocal() as db:
            v = db.get(Validation, validation_id)
            assert v is not None
            v.report = report
            v.status = "ok"
            db.commit()
        return {"validation_id": validation_id, "overall": report.get("overall")}

    return await step.run("run-checks", _run)
