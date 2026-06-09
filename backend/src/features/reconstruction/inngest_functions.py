"""Inngest functions for reconstruction.

Runs the full reconstruction pipeline as a durable, step-wrapped workflow.
SDK convention: handlers take a single `ctx` arg; `ctx.step` exposes the
step API.
"""
from __future__ import annotations

import logging
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

log = logging.getLogger(__name__)


class _Cancelled(Exception):
    """Raised from the backend progress callback to abort a cancelled run."""


def _is_cancelled(reconstruction_id: str) -> bool:
    with SessionLocal() as db:
        r = db.get(Reconstruction, reconstruction_id)
        return r is not None and r.status == "cancelled"


@register
@inngest_client.create_function(
    fn_id="reconstruct-video",
    trigger=inngest.TriggerEvent(event="reconstruction/requested"),
    retries=1,
)
async def reconstruct_video(ctx: inngest.Context) -> dict:
    step = ctx.step
    payload = ctx.event.data or {}
    reconstruction_id: str = payload["reconstruction_id"]
    backend_name: str = payload.get("backend", "demo_fixture")
    params: dict = payload.get("params", {})

    settings = get_settings()

    async def _extract() -> dict:
        with SessionLocal() as db:
            recon = db.get(Reconstruction, reconstruction_id)
            if recon is None:
                raise RuntimeError(f"reconstruction {reconstruction_id} disappeared")
            # Don't un-cancel a run the user cancelled before we picked it up.
            if recon.status != "cancelled":
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
        try:
            write_thumbnail(frames[0], settings.data_dir / "projects" / project_id / "thumbnail.png")
        except Exception:
            pass
        return {"n_frames": len(frames), "project_id": project_id, "frames_dir": str(frames_dir)}

    extracted = await step.run("extract-frames", _extract)

    async def _run_backend() -> dict:
        # Bail before loading models if already cancelled.
        if _is_cancelled(reconstruction_id):
            return {"cancelled": True}
        backend = get_backend(backend_name)
        out_dir = settings.data_dir / "projects" / extracted["project_id"] / "reconstruction"
        t0 = time.time()
        # Forward reconstruction tunables (depth model, FOV override) to the
        # backend; depth_fusion reads them from intrinsics_hint.
        hint = {k: params[k] for k in ("depth_model", "fov_deg") if k in params}

        def _progress(_p: float, _m: str) -> None:
            # Cooperative cancellation: stop between frames if the user hit
            # Cancel, so we don't keep burning compute (and memory) on it.
            if _is_cancelled(reconstruction_id):
                raise _Cancelled()

        try:
            result = backend.reconstruct(
                ReconstructionInput(
                    frames_dir=Path(extracted["frames_dir"]),
                    fps_sampled=float(params.get("fps", 4.0)),
                    intrinsics_hint=hint or None,
                ),
                out_dir,
                progress_cb=_progress,
            )
        except _Cancelled:
            return {"cancelled": True}
        elapsed = time.time() - t0
        return {
            "mesh_path": str(result.mesh_path),
            "elapsed_s": elapsed,
            "backend_meta": result.backend_meta,
        }

    backend_out = await step.run("run-backend", _run_backend)

    async def _persist() -> dict:
        with SessionLocal() as db:
            r = db.get(Reconstruction, reconstruction_id)
            if r is None:
                raise RuntimeError(f"reconstruction {reconstruction_id} disappeared")
            # Leave a cancelled run cancelled — don't flip it to ok.
            if backend_out.get("cancelled") or r.status == "cancelled":
                return {"reconstruction_id": reconstruction_id, "cancelled": True}
            r.mesh_path = backend_out["mesh_path"]
            r.status = "ok"
            r.elapsed_s = backend_out["elapsed_s"]
            r.params = {**(r.params or {}), **backend_out.get("backend_meta", {})}
            db.commit()
        return {"reconstruction_id": reconstruction_id, "ok": True}

    return await step.run("persist", _persist)


# (failure handler lives in src.inngest_failures — one generic dispatcher
# marks reconstruction/build/validation/replay rows failed.)
