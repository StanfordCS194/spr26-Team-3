"""Reconstruction routes:
  GET  /api/reconstruction/backends           - list available backends
  POST /api/projects/{id}/upload-video        - multipart video upload
  POST /api/projects/{id}/reconstruct         - kick off reconstruction job
  GET  /api/projects/{id}/reconstruction      - latest reconstruction row
"""
from __future__ import annotations

import shutil
import threading
import time
import traceback
from pathlib import Path

import inngest
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from nanoid import generate as nanoid
from sqlalchemy import select

from src.config import get_settings
from src.db import SessionLocal
from src.deps import DbSession, ProjectDep
from src.features.reconstruction.backends import get_backend, list_backends
from src.features.reconstruction.backends.base import ReconstructionInput
from src.features.reconstruction.service import extract_frames, write_thumbnail
from src.inngest_client import inngest_client
from src.models import Project, Reconstruction
from src.schemas import BackendInfo, ReconstructionOut, ReconstructionRequest

router = APIRouter()


@router.get("/reconstruction/backends", response_model=list[BackendInfo])
def get_backends() -> list[dict]:
    return list_backends()


@router.post("/projects/{project_id}/upload-video", response_model=dict)
async def upload_video(
    project: ProjectDep,
    db: DbSession,
    file: UploadFile = File(...),
) -> dict:
    if not file.filename:
        raise HTTPException(400, "missing filename")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".webm", ".jpg", ".jpeg", ".png"}:
        raise HTTPException(400, f"unsupported file type {suffix}")
    settings = get_settings()
    project_dir = settings.data_dir / "projects" / project.id
    project_dir.mkdir(parents=True, exist_ok=True)
    out_path = project_dir / f"input{suffix}"
    with out_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    size = out_path.stat().st_size

    project.video_path = str(out_path)
    db.commit()
    return {
        "project_id": project.id,
        "video_path": str(out_path),
        "size_bytes": size,
    }


@router.post("/projects/{project_id}/reconstruct", response_model=ReconstructionOut)
async def queue_reconstruction(
    project: ProjectDep,
    body: ReconstructionRequest,
    db: DbSession,
) -> Reconstruction:
    if not project.video_path:
        raise HTTPException(400, "no video uploaded — POST /upload-video first")

    backend_name = body.backend
    try:
        get_backend(backend_name)  # validate registry
    except ValueError as e:
        raise HTTPException(400, str(e))

    recon = Reconstruction(
        id=nanoid(size=12),
        project_id=project.id,
        backend=backend_name,
        params=body.params,
        status="pending",
    )
    db.add(recon)
    db.commit()
    db.refresh(recon)

    # Inngest is the single source of truth now. The durable function in
    # backend/src/features/reconstruction/inngest_functions.py picks up
    # the event, runs extract_frames → run_backend → persist, and writes
    # the row's status from pending → ok/failed.
    await inngest_client.send(
        events=[
            inngest.Event(
                name="reconstruction/requested",
                data={
                    "reconstruction_id": recon.id,
                    "backend": backend_name,
                    "params": body.params,
                },
            )
        ]
    )

    return recon


def _run_reconstruction_blocking(
    reconstruction_id: str, backend_name: str, params: dict
) -> None:
    """In-process worker for reconstruction. Same shape as the Inngest
    function's steps so swapping back to durable execution is a one-line
    change in `queue_reconstruction`."""
    settings = get_settings()
    try:
        # Mark running + resolve project
        with SessionLocal() as db:
            recon = db.get(Reconstruction, reconstruction_id)
            assert recon is not None
            recon.status = "running"
            db.commit()
            project = db.get(Project, recon.project_id)
            assert project is not None
            video_path = Path(project.video_path) if project.video_path else None
            project_id = recon.project_id

        if not video_path or not video_path.exists():
            raise RuntimeError("project has no video_path on disk")

        # Step 1: extract frames
        frames_dir = settings.data_dir / "projects" / project_id / "frames"
        n_frames = int(params.get("n_frames", 24))
        frames = extract_frames(video_path, frames_dir, n_frames=n_frames)
        try:
            write_thumbnail(frames[0], settings.data_dir / "projects" / project_id / "thumbnail.png")
        except Exception:
            pass

        # Step 2: run backend
        backend = get_backend(backend_name)
        out_dir = settings.data_dir / "projects" / project_id / "reconstruction"
        t0 = time.time()
        hint = {k: params[k] for k in ("depth_model", "fov_deg") if k in params}
        result = backend.reconstruct(
            ReconstructionInput(
                frames_dir=frames_dir,
                fps_sampled=float(params.get("fps", 4.0)),
                intrinsics_hint=hint or None,
            ),
            out_dir,
            progress_cb=lambda p, m: None,
        )
        elapsed = time.time() - t0

        # Step 3: persist
        with SessionLocal() as db:
            r = db.get(Reconstruction, reconstruction_id)
            assert r is not None
            r.mesh_path = str(result.mesh_path)
            r.status = "ok"
            r.elapsed_s = elapsed
            r.params = {**(r.params or {}), **result.backend_meta}
            db.commit()
    except Exception as exc:
        with SessionLocal() as db:
            r = db.get(Reconstruction, reconstruction_id)
            if r is not None:
                r.status = "failed"
                r.error = f"{exc.__class__.__name__}: {exc}"[:1000]
                db.commit()
        print(f"reconstruction {reconstruction_id} failed:")
        traceback.print_exc()


@router.get("/projects/{project_id}/reconstruction", response_model=ReconstructionOut | None)
def latest_reconstruction(project: ProjectDep, db: DbSession) -> Reconstruction | None:
    return db.scalars(
        select(Reconstruction)
        .where(Reconstruction.project_id == project.id)
        .order_by(Reconstruction.created_at.desc())
    ).first()
