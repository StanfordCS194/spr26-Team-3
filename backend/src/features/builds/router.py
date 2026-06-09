"""Build route: queues a Build row + emits Inngest event."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException
from nanoid import generate as nanoid
from sqlalchemy import select

from src.deps import DbSession, ProjectDep
from src.models import Build, Reconstruction
from src.schemas import BuildOut, BuildRequest

router = APIRouter()


@router.post("/{project_id}/build", response_model=BuildOut)
async def build_project(
    project: ProjectDep,
    body: BuildRequest,
    db: DbSession,
    background_tasks: BackgroundTasks,
) -> Build:
    recon: Reconstruction | None = None
    if body.reconstruction_id:
        recon = db.get(Reconstruction, body.reconstruction_id)
    else:
        recon = db.scalars(
            select(Reconstruction)
            .where(Reconstruction.project_id == project.id, Reconstruction.status == "ok")
            .order_by(Reconstruction.created_at.desc())
        ).first()

    if not recon or not recon.mesh_path:
        raise HTTPException(
            400, "no successful reconstruction — run Reconstruct first"
        )

    build = Build(
        id=nanoid(size=12),
        project_id=project.id,
        reconstruction_id=recon.id,
        status="pending",
    )
    db.add(build)
    db.commit()
    db.refresh(build)

    background_tasks.add_task(
        _run_build_blocking, build.id, body.target_diagonal_m, body.max_hulls,
        body.up_axis, body.enclose,
    )

    return build


def _run_build_blocking(
    build_id: str, target_diagonal_m: float | None, max_hulls: int | None,
    up_axis: str | None, enclose: bool,
) -> None:
    """In-process build worker (host mode, no Inngest). mesh -> MJCF."""
    import traceback

    from rl_env.build import BuildConfig, build_environment

    from src.config import get_settings
    from src.db import SessionLocal

    settings = get_settings()
    try:
        with SessionLocal() as db:
            b = db.get(Build, build_id)
            if b is None:
                return
            b.status = "running"
            db.commit()
            recon = db.get(Reconstruction, b.reconstruction_id) if b.reconstruction_id else None
            project_id = b.project_id
            mesh_path = recon.mesh_path if recon and recon.mesh_path else None

        if not mesh_path:
            raise RuntimeError("no reconstruction mesh — run Reconstruct first")

        out_dir = settings.data_dir / "projects" / project_id / "build"
        artifacts = build_environment(
            BuildConfig(
                mesh_path=mesh_path,
                out_dir=str(out_dir),
                target_diagonal_m=target_diagonal_m or settings.default_target_diagonal_m,
                up_axis=up_axis or "auto",
                max_hulls=max_hulls or 64,
                enclose=enclose,
            )
        )

        with SessionLocal() as db:
            b = db.get(Build, build_id)
            assert b is not None
            b.mjcf_path = str(artifacts.mjcf_path)
            b.n_hulls = artifacts.n_hulls
            b.bounds = {
                "min": artifacts.bounds[0].tolist(),
                "max": artifacts.bounds[1].tolist(),
                # 4x4 raw-mesh -> sim transform so the viewer can place the
                # robot's sim-frame trajectory onto the textured mesh.
                "raw_to_sim": artifacts.raw_to_sim.tolist() if artifacts.raw_to_sim is not None else None,
            }
            b.spawn_region = {
                "xmin": artifacts.spawn_region[0],
                "xmax": artifacts.spawn_region[1],
                "ymin": artifacts.spawn_region[2],
                "ymax": artifacts.spawn_region[3],
            }
            b.status = "ok"
            db.commit()
    except Exception as exc:
        with SessionLocal() as db:
            b = db.get(Build, build_id)
            if b is not None:
                b.status = "failed"
                b.error = f"{exc.__class__.__name__}: {exc}"[:1000]
                db.commit()
        traceback.print_exc()


@router.get("/{project_id}/build/latest", response_model=BuildOut | None)
def latest_build(project: ProjectDep, db: DbSession) -> Build | None:
    return db.scalars(
        select(Build)
        .where(Build.project_id == project.id)
        .order_by(Build.created_at.desc())
    ).first()


@router.post("/{project_id}/build/cancel", response_model=BuildOut | None)
def cancel_build(project: ProjectDep, db: DbSession) -> Build | None:
    build = db.scalars(
        select(Build)
        .where(
            Build.project_id == project.id,
            Build.status.in_(["pending", "running"]),
        )
        .order_by(Build.created_at.desc())
    ).first()
    if build is None:
        return None
    build.status = "cancelled"
    db.commit()
    db.refresh(build)
    return build
