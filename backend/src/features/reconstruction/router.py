"""Reconstruction routes. v1 only exposes the backend catalog;
`POST /api/projects/{id}/reconstruct` lands in PR-B.
"""
from __future__ import annotations

from fastapi import APIRouter

from src.features.reconstruction.backends import list_backends
from src.schemas import BackendInfo

router = APIRouter()


@router.get("/reconstruction/backends", response_model=list[BackendInfo])
def get_backends() -> list[dict]:
    return list_backends()
