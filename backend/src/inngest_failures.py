"""Generic Inngest failure handler.

Inngest fires `inngest/function.failed` when a function raises out of its
last step. We listen for it and flip the corresponding DB row from
`pending`/`running` to `failed`, recording the error message — otherwise
those rows sit forever in `pending` and the UI shows "Building…" with no
way to retry.

Each entry in `_MARKERS` maps a function id to (Model class, payload key
holding the row id). The handler dispatches based on the failed fn_id.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import inngest

from src.db import SessionLocal
from src.inngest_client import inngest_client, register
from src.models import Build, Reconstruction, Run, Task, Validation

if TYPE_CHECKING:
    from src.db import Base

log = logging.getLogger(__name__)

# fn_id substring → (Model, payload key for row id)
_MARKERS: list[tuple[str, type["Base"], str]] = [
    ("reconstruct-video", Reconstruction, "reconstruction_id"),
    ("build-env", Build, "build_id"),
    ("validate-mesh", Validation, "validation_id"),
    ("run-replay", Run, "run_id"),
    ("generate-task", Task, "task_id"),
]


@register
@inngest_client.create_function(
    fn_id="mark-row-failed",
    trigger=inngest.TriggerEvent(event="inngest/function.failed"),
    retries=0,
)
async def mark_row_failed(ctx: inngest.Context) -> dict:
    data = ctx.event.data or {}
    fn_id = data.get("function_id") or ""

    marker = next((m for m in _MARKERS if m[0] in fn_id), None)
    if marker is None:
        return {"skipped": True, "reason": "no marker for fn", "fn_id": fn_id}
    _, Model, id_key = marker

    event = data.get("event") or {}
    payload = (event.get("data") or {}) if isinstance(event, dict) else {}
    row_id = payload.get(id_key)
    if not row_id:
        return {"skipped": True, "reason": "no row id in payload"}

    error = data.get("error") or {}
    msg = error.get("message") if isinstance(error, dict) else str(error)

    with SessionLocal() as db:
        row = db.get(Model, row_id)
        if row is None:
            return {"skipped": True, "reason": "row disappeared"}
        row.status = "failed"
        row.error = (msg or "unknown error")[:1000]
        db.commit()
    log.warning("marked %s row %s failed: %s", Model.__name__, row_id, msg)
    return {"marked_failed": row_id, "model": Model.__name__}
