"""Validation routes. Stubbed in PR-A; real checks land in PR-B."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.deps import ProjectDep

router = APIRouter()


@router.post("/{project_id}/validate")
def validate_project(project: ProjectDep) -> dict:
    raise HTTPException(501, "validation lands in PR-B of worldscan-v2-video-product")
