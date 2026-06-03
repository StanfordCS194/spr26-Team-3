"""Run Inngest-backed jobs synchronously in pytest (no worker process).

Always pass the test ``db`` session from conftest so work shares the same
transaction as the FastAPI TestClient.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from rl_env.build import BuildConfig, build_environment
from rl_env.env import NavEnv, TaskConfig
from src.config import get_settings
from src.features.reconstruction.backends import get_backend
from src.features.reconstruction.backends.base import ReconstructionInput
from src.features.reconstruction.service import extract_frames, write_thumbnail
from src.features.replay.inngest_functions import _classify, _policy_greedy, _policy_random
from src.features.validation.checks import run_all
from src.models import Build, Policy, Project, Reconstruction, Run, Validation


def _seed_frames_from_image(video_path: Path, frames_dir: Path, n_frames: int = 8) -> list[Path]:
    """CI-friendly frame dir for still uploads (no ffmpeg/ffprobe required)."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    for p in frames_dir.iterdir():
        if p.is_file():
            p.unlink()
    suffix = video_path.suffix.lower() if video_path.suffix else ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png"}:
        suffix = ".jpg"
    frames: list[Path] = []
    for i in range(n_frames):
        target = frames_dir / f"frame_{i + 1:04d}{suffix}"
        shutil.copy(video_path, target)
        frames.append(target)
    return frames


def sync_reconstruct(
    db: Session,
    reconstruction_id: str,
    backend_name: str,
    params: dict | None = None,
) -> None:
    params = params or {}
    settings = get_settings()
    recon = db.get(Reconstruction, reconstruction_id)
    if recon is None:
        raise RuntimeError(f"reconstruction {reconstruction_id} missing")
    recon.status = "running"
    db.commit()

    project = db.get(Project, recon.project_id)
    if project is None:
        raise RuntimeError("project missing")
    video_path = Path(project.video_path) if project.video_path else None
    if not video_path or not video_path.exists():
        raise RuntimeError("project has no video_path on disk")

    project_id = recon.project_id
    frames_dir = settings.data_dir / "projects" / project_id / "frames"
    n_frames = int(params.get("n_frames", 24))
    if video_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        frames = _seed_frames_from_image(video_path, frames_dir, n_frames=n_frames)
    else:
        frames = extract_frames(video_path, frames_dir, n_frames=n_frames)
    try:
        write_thumbnail(
            frames[0],
            settings.data_dir / "projects" / project_id / "thumbnail.png",
        )
    except Exception:
        pass

    backend = get_backend(backend_name)
    out_dir = settings.data_dir / "projects" / project_id / "reconstruction"
    t0 = time.time()
    result = backend.reconstruct(
        ReconstructionInput(
            frames_dir=frames_dir,
            fps_sampled=float(params.get("fps", 4.0)),
            intrinsics_hint=None,
        ),
        out_dir,
        progress_cb=lambda _p, _m: None,
    )
    elapsed = time.time() - t0

    recon = db.get(Reconstruction, reconstruction_id)
    assert recon is not None
    recon.mesh_path = str(result.mesh_path)
    recon.status = "ok"
    recon.elapsed_s = elapsed
    recon.params = {**(recon.params or {}), **result.backend_meta}
    db.commit()


def sync_validate(db: Session, validation_id: str) -> None:
    v = db.get(Validation, validation_id)
    if v is None:
        raise RuntimeError(f"validation {validation_id} missing")
    v.status = "running"
    db.commit()

    recon = db.get(Reconstruction, v.reconstruction_id)
    if recon is None or not recon.mesh_path:
        raise RuntimeError("reconstruction has no mesh")

    report = run_all(recon.mesh_path)
    v = db.get(Validation, validation_id)
    assert v is not None
    v.report = report
    v.status = "ok"
    db.commit()


def sync_build(
    db: Session,
    build_id: str,
    *,
    up_axis: str = "y",
    target_diagonal_m: float | None = None,
    max_hulls: int = 64,
) -> None:
    settings = get_settings()
    b = db.get(Build, build_id)
    if b is None:
        raise RuntimeError(f"build {build_id} missing")
    b.status = "running"
    db.commit()

    recon = db.get(Reconstruction, b.reconstruction_id) if b.reconstruction_id else None
    mesh_path = recon.mesh_path if recon and recon.mesh_path else None
    if not mesh_path:
        raise RuntimeError("no reconstruction mesh — run Reconstruct first")

    out_dir = settings.data_dir / "projects" / b.project_id / "build"
    artifacts = build_environment(
        BuildConfig(
            mesh_path=mesh_path,
            out_dir=str(out_dir),
            target_diagonal_m=target_diagonal_m or settings.default_target_diagonal_m,
            up_axis=up_axis,
            max_hulls=max_hulls,
        )
    )

    b = db.get(Build, build_id)
    assert b is not None
    b.mjcf_path = str(artifacts.mjcf_path)
    b.n_hulls = artifacts.n_hulls
    b.bounds = {
        "min": artifacts.bounds[0].tolist(),
        "max": artifacts.bounds[1].tolist(),
    }
    b.spawn_region = {
        "xmin": artifacts.spawn_region[0],
        "xmax": artifacts.spawn_region[1],
        "ymin": artifacts.spawn_region[2],
        "ymax": artifacts.spawn_region[3],
    }
    b.status = "ok"
    db.commit()


def sync_replay(
    db: Session,
    run_id: str,
    *,
    project_id: str,
    policy: str = "greedy",
    policy_id: str | None = None,
    episodes: int = 5,
    max_steps: int = 300,
    seed: int = 0,
) -> None:
    settings = get_settings()
    r = db.get(Run, run_id)
    if r is None:
        raise RuntimeError(f"run {run_id} missing")
    r.status = "running"
    db.commit()

    pol = db.get(Policy, policy_id) if policy_id else None
    if pol:
        build = db.get(Build, pol.build_id)
    else:
        build = (
            db.query(Build)
            .filter(Build.project_id == project_id, Build.status == "ok")
            .order_by(Build.created_at.desc())
            .first()
        )
    if build is None or not build.mjcf_path:
        raise RuntimeError("no built scene")
    mjcf_path = build.mjcf_path
    bounds = build.bounds
    spawn_region = build.spawn_region
    pid = build.project_id

    rng = np.random.default_rng(seed)
    if policy == "ppo":
        if not pol or not pol.ckpt_path or not Path(pol.ckpt_path).exists():
            raise RuntimeError("no PPO checkpoint")
        from stable_baselines3 import PPO

        sb3 = PPO.load(pol.ckpt_path, device="cpu")

        def policy_fn(obs, _rng):
            action, _ = sb3.predict(obs, deterministic=True)
            return action

    else:
        policy_fn = _policy_greedy if policy == "greedy" else _policy_random

    env = NavEnv(mjcf_path, task=TaskConfig(max_steps=max_steps))
    successes = 0
    rewards = []
    episodes_out = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_steps = 0
        collisions = 0
        traj = []
        speeds = []
        info: dict = {}
        prev_xy = env._agent_xy().astype(float)
        for _ in range(max_steps):
            a = policy_fn(obs, rng)
            obs, rew, term, trunc, info = env.step(a)
            ep_reward += rew
            ep_steps += 1
            curr = env._agent_xy().astype(float)
            speeds.append(float(np.linalg.norm(curr - prev_xy)))
            prev_xy = curr
            if env._has_scene_collision():
                collisions += 1
            traj.append(
                {
                    "step": ep_steps,
                    "x": float(curr[0]),
                    "y": float(curr[1]),
                    "collision": bool(env._has_scene_collision()),
                }
            )
            if term or trunc:
                break
        success = bool(info.get("success", False))
        distance = float(info.get("distance", -1.0))
        failure_class = _classify(ep_steps, max_steps, distance, success, collisions, speeds[-50:])
        if success:
            successes += 1
        rewards.append(ep_reward)
        episodes_out.append(
            {
                "steps": ep_steps,
                "reward": round(float(ep_reward), 3),
                "distance": round(distance, 3),
                "success": success,
                "failure_class": failure_class,
                "spawn": env._agent_xy().astype(float).tolist(),
                "goal": env._goal_xy().astype(float).tolist(),
                "trajectory": traj,
            }
        )
    env.close()

    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    traj_dir = settings.data_dir / "projects" / pid / "runs" / run_id
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_path = traj_dir / "trajectories.json"
    traj_path.write_text(
        json.dumps(
            {
                "bounds": bounds,
                "spawn_region": spawn_region,
                "episodes": episodes_out,
            }
        )
    )

    r = db.get(Run, run_id)
    assert r is not None
    r.episodes = episodes
    r.successes = successes
    r.avg_reward = round(avg_reward, 3)
    r.trajectories_path = str(traj_path)
    r.status = "ok"
    db.commit()
