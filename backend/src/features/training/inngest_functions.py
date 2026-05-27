"""Inngest functions for training. Stubbed in PR-A; real PPO loop in PR-C."""
from __future__ import annotations

import inngest

from src.inngest_client import inngest_client, register


@register
@inngest_client.create_function(
    fn_id="train-policy",
    trigger=inngest.TriggerEvent(event="training/requested"),
    retries=1,
)
async def train_policy(ctx: inngest.Context, step: inngest.Step) -> dict:
    # PR-C will replace this with a chunked PPO loop that emits progress.
    return {"status": "stub", "message": "train-policy lands in PR-C"}
