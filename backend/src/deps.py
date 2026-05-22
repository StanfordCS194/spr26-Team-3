"""Shared FastAPI dependencies."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from src.config import Settings, get_settings
from src.db import get_db
from src.models import Project

DbSession = Annotated[Session, Depends(get_db)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_project(project_id: str, db: DbSession) -> Project:
    p = db.get(Project, project_id)
    if not p:
        raise HTTPException(404, f"unknown project {project_id}")
    return p


ProjectDep = Annotated[Project, Depends(get_project)]


def project_data_dir(project: ProjectDep, settings: SettingsDep) -> Path:
    p = settings.data_dir / "projects" / project.id
    p.mkdir(parents=True, exist_ok=True)
    return p
