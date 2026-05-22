"""Inngest functions for reconstruction. Stubbed in PR-A.
The real `reconstruct_video` lands in PR-B.
"""
from __future__ import annotations

import inngest

from src.inngest_client import inngest_client, register


@register
@inngest_client.create_function(
    fn_id="reconstruct-video",
    trigger=inngest.TriggerEvent(event="reconstruction/requested"),
    retries=2,
)
async def reconstruct_video(ctx: inngest.Context, step: inngest.Step) -> dict:
    # PR-B will replace this with extract_frames -> run_backend -> persist.
    return {"status": "stub", "message": "reconstruct-video lands in PR-B"}
