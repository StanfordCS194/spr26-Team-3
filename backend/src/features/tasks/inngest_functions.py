"""Inngest function: NL task fields → Claude → validated GeneratedTask module."""
from __future__ import annotations

import logging
import traceback

import inngest

from src.db import SessionLocal
from src.features.tasks.service import run_task_codegen
from src.inngest_client import inngest_client, register

log = logging.getLogger(__name__)


@register
@inngest_client.create_function(
    fn_id="generate-task",
    trigger=inngest.TriggerEvent(event="task/codegen_requested"),
    retries=0,
)
async def generate_task(ctx: inngest.Context) -> dict:
    step = ctx.step
    payload = ctx.event.data or {}
    task_id: str = payload["task_id"]
    log.info("generate-task: %s", task_id)

    async def _run() -> dict:
        try:
            with SessionLocal() as db:
                version_id = run_task_codegen(db, task_id)
            return {"task_id": task_id, "task_version_id": version_id, "ok": True}
        except Exception as e:
            log.exception("generate-task failed: %s", task_id)
            with SessionLocal() as db:
                from src.models import Task

                t = db.get(Task, task_id)
                if t is not None and t.status == "generating":
                    t.status = "failed"
                    t.error = traceback.format_exc()[-1000:]
                    db.commit()
            raise RuntimeError(str(e)) from e

    return await step.run("codegen-and-validate", _run)
