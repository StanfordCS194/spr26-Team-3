"""Training routes — stubbed in PR-A; real PPO loop lands in PR-C."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.deps import ProjectDep

router = APIRouter()


@router.post("/{project_id}/train")
def train(project: ProjectDep) -> dict:
    raise HTTPException(501, "training lands in PR-C of worldscan-v2-video-product")
