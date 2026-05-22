"""Inngest function for PPO training.

Long-running (10s – 10min). Runs the full PPO loop inside a single
`step.run` block — Inngest's retry will re-execute the step on failure,
which is the desired behaviour for training (a partial checkpoint is
useless; we want the run to either fully complete or be retried).
"""
from __future__ import annotations

import logging

import inngest

from src.features.training.service import run_training
from src.inngest_client import inngest_client, register

log = logging.getLogger(__name__)


@register
@inngest_client.create_function(
    fn_id="train-policy",
    trigger=inngest.TriggerEvent(event="training/requested"),
    retries=0,
)
async def train_policy(ctx: inngest.Context) -> dict:
    step = ctx.step
    payload = ctx.event.data or {}
    policy_id: str = payload["policy_id"]
    mjcf_path: str = payload["mjcf_path"]
    total_steps: int = int(payload.get("total_steps", 50_000))
    n_envs: int = int(payload.get("n_envs", 4))
    max_steps: int = int(payload.get("max_steps", 300))
    seed: int = int(payload.get("seed", 0))

    log.info("train-policy: policy_id=%s steps=%d n_envs=%d seed=%d",
             policy_id, total_steps, n_envs, seed)

    async def _train() -> dict:
        # run_training is blocking (sb3.learn). Inngest's Python SDK runs
        # the step in a worker thread under the hood so blocking calls are
        # fine here.
        run_training(
            policy_id=policy_id,
            mjcf_path=mjcf_path,
            total_steps=total_steps,
            n_envs=n_envs,
            max_steps=max_steps,
            seed=seed,
        )
        return {"policy_id": policy_id, "ok": True}

    return await step.run("ppo-train", _train)
