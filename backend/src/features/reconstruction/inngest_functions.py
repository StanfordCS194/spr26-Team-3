"""Inngest functions for reconstruction.

`reconstruct-video` runs three steps:
  1. extract_frames — ffmpeg samples N frames from the uploaded video
  2. run_backend    — invokes the chosen ReconstructionBackend
  3. persist        — updates the reconstruction row with mesh path + meta

Progress events are emitted via step.send_event so the frontend's progress
bar can move. Each step is independently retried by Inngest if it fails.
"""
from __future__ import annotations

import time
from pathlib import Path

import inngest

from src.config import get_settings
from src.db import SessionLocal
from src.features.reconstruction.backends import get_backend
from src.features.reconstruction.backends.base import ReconstructionInput
from src.features.reconstruction.service import extract_frames, write_thumbnail
from src.inngest_client import inngest_client, register
from src.models import Project, Reconstruction


@register
@inngest_client.create_function(
    fn_id="reconstruct-video",
    trigger=inngest.TriggerEvent(event="reconstruction/requested"),
    retries=1,
)
async def reconstruct_video(ctx: inngest.Context, step: inngest.Step) -> dict:
    payload = ctx.event.data
    reconstruction_id: str = payload["reconstruction_id"]
    backend_name: str = payload.get("backend", "demo_fixture")
    params: dict = payload.get("params", {})

    settings = get_settings()

    # ---- step 1: extract frames ----------------------------------------------

    async def _extract() -> dict:
        with SessionLocal() as db:
            recon = db.get(Reconstruction, reconstruction_id)
            if recon is None:
                raise RuntimeError(f"reconstruction {reconstruction_id} disappeared")
            recon.status = "running"
            db.commit()
            project = db.get(Project, recon.project_id)
            assert project is not None
            video_path = Path(project.video_path) if project.video_path else None
            project_id = recon.project_id

        if not video_path or not video_path.exists():
            raise RuntimeError("project has no video_path on disk")

        frames_dir = settings.data_dir / "projects" / project_id / "frames"
        n_frames = int(params.get("n_frames", 24))
        frames = extract_frames(video_path, frames_dir, n_frames=n_frames)

        thumb = settings.data_dir / "projects" / project_id / "thumbnail.png"
        try:
            write_thumbnail(frames[0], thumb)
        except Exception:
            pass

        return {
            "n_frames": len(frames),
            "frames_dir": str(frames_dir),
            "project_id": project_id,
        }

    extracted = await step.run("extract-frames", _extract)

    await step.send_event(
        "progress-extracted",
        inngest.Event(
            name="reconstruction/progress",
            data={
                "reconstruction_id": reconstruction_id,
                "progress": 0.20,
                "message": f"extracted {extracted['n_frames']} frames",
            },
        ),
    )

    # ---- step 2: run backend -------------------------------------------------

    async def _run_backend() -> dict:
        backend = get_backend(backend_name)
        out_dir = settings.data_dir / "projects" / extracted["project_id"] / "reconstruction"

        t0 = time.time()

        def cb(p: float, msg: str) -> None:
            pass  # in-process callback; cross-step progress emits below

        inp = ReconstructionInput(
            frames_dir=Path(extracted["frames_dir"]),
            fps_sampled=float(params.get("fps", 4.0)),
            intrinsics_hint=None,
        )
        result = backend.reconstruct(inp, out_dir, cb)
        elapsed = time.time() - t0

        return {
            "mesh_path": str(result.mesh_path),
            "point_cloud_path": str(result.point_cloud_path) if result.point_cloud_path else None,
            "backend_meta": result.backend_meta,
            "elapsed_s": elapsed,
        }

    backend_out = await step.run("run-backend", _run_backend)

    await step.send_event(
        "progress-meshed",
        inngest.Event(
            name="reconstruction/progress",
            data={
                "reconstruction_id": reconstruction_id,
                "progress": 0.85,
                "message": "backend complete; persisting",
            },
        ),
    )

    # ---- step 3: persist -----------------------------------------------------

    async def _persist() -> dict:
        with SessionLocal() as db:
            recon = db.get(Reconstruction, reconstruction_id)
            assert recon is not None
            recon.mesh_path = backend_out["mesh_path"]
            recon.status = "ok"
            recon.elapsed_s = backend_out["elapsed_s"]
            recon.params = {**(recon.params or {}), **backend_out.get("backend_meta", {})}
            db.commit()
        return {"reconstruction_id": reconstruction_id, "ok": True}

    persisted = await step.run("persist", _persist)

    await step.send_event(
        "progress-done",
        inngest.Event(
            name="reconstruction/progress",
            data={
                "reconstruction_id": reconstruction_id,
                "progress": 1.0,
                "message": "done",
            },
        ),
    )
    return persisted


# When ANY step in `reconstruct-video` raises (after retries), Inngest emits
# `inngest/function.failed`. Catch it and persist the failure to the
# reconstruction row so the UI can surface it without polling Inngest.
@register
@inngest_client.create_function(
    fn_id="mark-reconstruction-failed",
    trigger=inngest.TriggerEvent(event="inngest/function.failed"),
    retries=0,
)
async def mark_reconstruction_failed(ctx: inngest.Context, step: inngest.Step) -> dict:
    data = ctx.event.data or {}
    fn_id = (data.get("function_id") or "")
    if "reconstruct-video" not in fn_id:
        return {"skipped": True}
    event = data.get("event") or {}
    payload = (event.get("data") or {}) if isinstance(event, dict) else {}
    reconstruction_id = payload.get("reconstruction_id")
    error = data.get("error") or {}
    msg = error.get("message") if isinstance(error, dict) else str(error)
    if not reconstruction_id:
        return {"skipped": True}
    with SessionLocal() as db:
        recon = db.get(Reconstruction, reconstruction_id)
        if recon is None:
            return {"skipped": True}
        recon.status = "failed"
        recon.error = (msg or "unknown error")[:1000]
        db.commit()
    return {"marked_failed": reconstruction_id}
