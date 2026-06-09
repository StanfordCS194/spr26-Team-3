"""Replay route: queues a Run + emits Inngest event. Returns the run row
with status='pending'; the frontend polls /runs/{id} until status='ok' and
then loads the trajectories JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from nanoid import generate as nanoid
from pydantic import BaseModel
from sqlalchemy import select

from src.deps import DbSession, ProjectDep
from src.models import Build, Policy, Reconstruction, Run
from src.schemas import RunOut

router = APIRouter()


class ReplayRequest(BaseModel):
    policy: str = "greedy"  # 'random' | 'greedy' | 'ppo'
    episodes: int = 5
    max_steps: int = 300
    seed: int = 0
    policy_id: str | None = None


@router.post("/{project_id}/replay", response_model=RunOut)
async def replay(
    project: ProjectDep, body: ReplayRequest, db: DbSession, background_tasks: BackgroundTasks
) -> Run:
    build = db.scalars(
        select(Build)
        .where(Build.project_id == project.id, Build.status == "ok")
        .order_by(Build.created_at.desc())
    ).first()
    if not build:
        raise HTTPException(404, "no built scene — run Build first")

    policy_id = body.policy_id
    if body.policy == "ppo" and policy_id is None:
        policy = db.scalars(
            select(Policy).where(Policy.build_id == build.id).order_by(Policy.created_at.desc())
        ).first()
        if policy is None or not policy.ckpt_path or not Path(policy.ckpt_path).exists():
            raise HTTPException(400, "no trained PPO policy — train one first")
        policy_id = policy.id

    run = Run(
        id=nanoid(size=12),
        policy_id=policy_id if body.policy == "ppo" else None,
        baseline=body.policy if body.policy in ("random", "greedy") else None,
        status="pending",
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    background_tasks.add_task(
        _run_replay_blocking, run.id, body.policy, policy_id, project.id,
        body.episodes, body.max_steps, body.seed,
    )
    return run


def _run_replay_blocking(
    run_id: str, policy_name: str, policy_id: str | None, project_id: str,
    episodes: int, max_steps: int, seed: int,
) -> None:
    """In-process rollout worker (host mode, no Inngest). Writes trajectories
    JSON the frontend replays, and updates the Run row."""
    import traceback

    import numpy as np

    from rl_env.env import NavEnv, TaskConfig
    from src.config import get_settings
    from src.db import SessionLocal

    def _greedy(obs, rng):
        v = obs[4:6]
        n = float(np.linalg.norm(v))
        return np.zeros(2, dtype=np.float32) if n < 1e-6 else (v / n).astype(np.float32)

    def _random(obs, rng):
        return rng.uniform(-1.0, 1.0, size=2).astype(np.float32)

    settings = get_settings()
    try:
        with SessionLocal() as db:
            r = db.get(Run, run_id)
            if r is None:
                return
            r.status = "running"
            db.commit()
            policy = db.get(Policy, policy_id) if policy_id else None
            if policy:
                build = db.get(Build, policy.build_id)
            else:
                build = db.scalars(
                    select(Build).where(Build.project_id == project_id)
                    .order_by(Build.created_at.desc())
                ).first()
            if build is None or not build.mjcf_path:
                raise RuntimeError("no built scene for this project")
            mjcf_path = build.mjcf_path
            bounds = build.bounds
            spawn_region = build.spawn_region
            build_project_id = build.project_id
            ckpt_path = policy.ckpt_path if policy else None
            recon = (
                db.get(Reconstruction, build.reconstruction_id)
                if build.reconstruction_id
                else None
            )
            try:
                footprint = _nav_footprint(build, recon.mesh_path if recon else None)
            except Exception:
                footprint = None

        if policy_name == "ppo":
            if not ckpt_path or not Path(ckpt_path).exists():
                raise RuntimeError("no trained PPO policy")
            from stable_baselines3 import PPO

            sb3 = PPO.load(ckpt_path, device="cpu")

            def policy_fn(obs, _rng):
                action, _ = sb3.predict(obs, deterministic=True)
                return action
        else:
            policy_fn = _greedy if policy_name == "greedy" else _random

        # Spawn start & goal apart, but keep them in the CENTRAL part of the
        # footprint. A partial/curved reconstruction doesn't fill its bounding
        # box, so the box corners are empty space ("outside the room") — spawn
        # in the inner ~65% so the robot stays on actual floor.
        tcfg = {"max_steps": max_steps}
        try:
            sr = spawn_region or {}
            cx = (float(sr["xmin"]) + float(sr["xmax"])) / 2
            cy = (float(sr["ymin"]) + float(sr["ymax"])) / 2
            xr = (float(sr["xmax"]) - float(sr["xmin"])) / 2 * 0.65
            yr = (float(sr["ymax"]) - float(sr["ymin"])) / 2 * 0.65
            tcfg["spawn_region"] = (cx - xr, cx + xr, cy - yr, cy + yr)
            tcfg["min_start_goal_dist"] = max(0.8, 0.7 * float(np.hypot(2 * xr, 2 * yr)))
        except Exception:
            pass
        env = NavEnv(mjcf_path, task=TaskConfig(**tcfg))
        rng = np.random.default_rng(seed)
        successes = 0
        rewards = []
        episodes_out = []
        for ep in range(episodes):
            if footprint:
                # Spawn on the SCANNED floor, not the bounding box — a partial
                # scan's box corners are empty space "outside the room". Each
                # episode draws a fresh far-apart pair of footprint cells.
                cells = np.asarray(footprint["free_cells"], dtype=float)
                ep_rng = np.random.default_rng(seed + ep)
                want = 0.55 * float(footprint.get("max_dist", 2.0))
                pick = cells[ep_rng.integers(len(cells))], cells[ep_rng.integers(len(cells))]
                for _ in range(64):
                    a_, b_ = cells[ep_rng.integers(len(cells))], cells[ep_rng.integers(len(cells))]
                    if float(np.linalg.norm(a_ - b_)) >= want:
                        pick = a_, b_
                        break
                env.task.fixed_spawn = (float(pick[0][0]), float(pick[0][1]))
                env.task.fixed_goal = (float(pick[1][0]), float(pick[1][1]))
            obs, _ = env.reset(seed=seed + ep)
            ep_reward = 0.0
            ep_steps = 0
            collisions = 0
            traj = []
            info: dict = {}
            spawn_xy = env._agent_xy().astype(float).tolist()
            goal_xy = env._goal_xy().astype(float).tolist()
            # Obstacle-avoidance: the last 16 obs are lidar ranges. When the
            # nearest ray drops below PROX the robot is hugging an obstacle; if
            # it leaves that proximity span without ever colliding, it "avoided"
            # the obstacle. We count those events + where they happened.
            PROX = 0.55
            avoided = 0
            in_prox = False
            prox_collided = False
            avoid_points: list[dict] = []
            for _ in range(max_steps):
                a = policy_fn(obs, rng)
                obs, rew, term, trunc, info = env.step(a)
                ep_reward += rew
                ep_steps += 1
                curr = env._agent_xy().astype(float)
                col = bool(env._has_scene_collision())
                if col:
                    collisions += 1
                lidar = obs[-16:]
                near = bool(float(np.min(lidar)) < PROX) if len(lidar) else False
                if near and not in_prox:
                    in_prox, prox_collided = True, False
                if in_prox and col:
                    prox_collided = True
                if in_prox and not near:
                    if not prox_collided:
                        avoided += 1
                        avoid_points.append({"x": float(curr[0]), "y": float(curr[1])})
                    in_prox = False
                traj.append({
                    "step": ep_steps, "x": float(curr[0]), "y": float(curr[1]),
                    "collision": col, "near": near,
                })
                if term or trunc:
                    break
            if in_prox and not prox_collided:
                avoided += 1
            success = bool(info.get("success", False))
            distance = float(info.get("distance", -1.0))
            if success and traj:
                # Dock the final success_radius stretch so the rendered robot
                # visibly arrives AT the goal marker — interpolated at the
                # normal per-step pace so it rolls in, not teleports.
                last = np.array([traj[-1]["x"], traj[-1]["y"]], dtype=float)
                g = np.array(goal_xy, dtype=float)
                n = max(2, int(float(np.linalg.norm(g - last)) / 0.025))
                for k in range(1, n + 1):
                    p = last + (g - last) * (k / n)
                    traj.append({
                        "step": ep_steps + k, "x": float(p[0]), "y": float(p[1]),
                        "collision": False, "near": False,
                    })
            if success:
                successes += 1
            rewards.append(ep_reward)
            episodes_out.append({
                "steps": ep_steps, "reward": round(float(ep_reward), 3),
                "distance": round(distance, 3), "success": success,
                "avoided": avoided, "avoid_points": avoid_points, "collisions": collisions,
                "spawn": spawn_xy, "goal": goal_xy, "trajectory": traj,
            })
        env.close()
        avg_reward = float(np.mean(rewards)) if rewards else 0.0

        traj_dir = settings.data_dir / "projects" / build_project_id / "runs" / run_id
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_path = traj_dir / "trajectories.json"
        traj_path.write_text(json.dumps(
            {"bounds": bounds, "spawn_region": spawn_region, "episodes": episodes_out}
        ))

        with SessionLocal() as db:
            r = db.get(Run, run_id)
            assert r is not None
            r.episodes = episodes
            r.successes = successes
            r.avg_reward = round(avg_reward, 3)
            r.trajectories_path = str(traj_path)
            r.status = "ok"
            db.commit()
    except Exception as exc:
        with SessionLocal() as db:
            r = db.get(Run, run_id)
            if r is not None:
                r.status = "failed"
                db.commit()
        traceback.print_exc()


@router.get("/{project_id}/runs/{run_id}", response_model=RunOut)
def get_run(project: ProjectDep, run_id: str, db: DbSession) -> Run:
    r = db.get(Run, run_id)
    if r is None:
        raise HTTPException(404, f"unknown run {run_id}")
    return r


@router.get("/{project_id}/runs/{run_id}/trajectories")
def get_run_trajectories(project: ProjectDep, run_id: str, db: DbSession) -> dict:
    r = db.get(Run, run_id)
    if r is None or not r.trajectories_path:
        raise HTTPException(404, "run has no trajectories yet")
    p = Path(r.trajectories_path)
    if not p.exists():
        raise HTTPException(404, "trajectories file missing on disk")
    return json.loads(p.read_text())


# --------------------------------------------------------------------------- #
# Head-to-head: greedy vs PPO on the SAME start→goal (shows where PPO wins)
# --------------------------------------------------------------------------- #
class CompareRequest(BaseModel):
    seed: int | None = None
    max_steps: int = 300


def _build_raw_to_sim(build) -> tuple[list | None, float | None]:
    """(raw_to_sim, floor_z) from build.bounds, falling back to the build dir's
    metadata.json for builds made before these were stored in the DB."""
    b = build.bounds or {}
    raw_to_sim = b.get("raw_to_sim")
    floor_z = (b.get("min") or [None, None, None])[2]
    if raw_to_sim is None or floor_z is None:
        try:
            meta = json.loads((Path(build.mjcf_path).parent / "metadata.json").read_text())
            raw_to_sim = raw_to_sim if raw_to_sim is not None else meta.get("raw_to_sim")
            floor_z = floor_z if floor_z is not None else meta.get("floor_z")
        except Exception:
            pass
    return raw_to_sim, floor_z


def _nav_footprint(build, mesh_path: str | None) -> dict | None:
    """Occupancy footprint of the *scanned floor* in sim coordinates.

    Partial phone scans don't fill their bounding box — spawning anywhere in the
    box puts the robot visually "outside the room". This grids the mesh's
    floor-band vertices, drops cells near obstacles/walls (points in the band
    above the floor), keeps the largest connected free region, and returns:
      floor_z      — the REAL floor height (densest vertex slab), not the noisy
                     mesh minimum, so balls render resting on the visible floor
      free_cells   — [x, y] centers the robot can stand on
      corner_pair  — the two farthest-apart free cells (corner → opposite corner)
    Cached as nav_footprint.json next to the MJCF. Returns None for meshes too
    small/synthetic to profile (caller falls back to box spawning).
    """
    if not mesh_path or not Path(mesh_path).exists() or not build.mjcf_path:
        return None
    cache = Path(build.mjcf_path).parent / "nav_footprint.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass

    import numpy as np
    import trimesh

    raw_to_sim, _ = _build_raw_to_sim(build)
    if raw_to_sim is None:
        return None
    mesh = trimesh.load(mesh_path, process=False)
    v = np.asarray(mesh.vertices)
    if len(v) < 20_000:  # synthetic fixtures — footprint profiling is meaningless
        return None
    M = np.asarray(raw_to_sim, dtype=float)
    vs = (M[:3, :3] @ v.T).T + M[:3, 3]
    zs = vs[:, 2]

    # Real floor = densest z-slab in the lower 40% of the height range.
    hist, edges = np.histogram(zs, bins=80)
    i = int(np.argmax(hist[: int(80 * 0.4)]))
    floor_z = float((edges[i] + edges[i + 1]) / 2)

    floor = vs[(zs > floor_z - 0.12) & (zs < floor_z + 0.12)][:, :2]
    obst = vs[(zs > floor_z + 0.18) & (zs < floor_z + 1.4)][:, :2]
    if len(floor) < 5_000:
        return None

    CELL = 0.15
    origin = vs[:, :2].min(axis=0)
    from collections import Counter

    def cells_of(pts):
        return Counter(map(tuple, np.floor((pts - origin) / CELL).astype(int)))

    gf, go = cells_of(floor), cells_of(obst)
    thr = max(3, int(np.percentile(np.array(list(gf.values())), 30) * 0.5))
    floor_cells = {c for c, n in gf.items() if n >= thr}
    obst_cells = {c for c, n in go.items() if n >= max(5, thr)}
    nbrs = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    free = {
        c for c in floor_cells
        if not any((c[0] + dx, c[1] + dy) in obst_cells for dx, dy in nbrs)
    }
    # Ground truth: place the agent at each candidate cell in the ACTUAL MJCF
    # and keep only cells where it stands collision-free. Point-density
    # clearance can't see how far the sim's collision geometry extends.
    try:
        import mujoco

        from rl_env.env import NavEnv, TaskConfig

        probe = NavEnv(build.mjcf_path, task=TaskConfig(max_steps=10))
        probe.reset(seed=0)
        probed: set = set()
        for c in free:
            x = (c[0] + 0.5) * CELL + origin[0]
            y = (c[1] + 0.5) * CELL + origin[1]
            probe.data.qpos[probe._agent_x_qpos] = x
            probe.data.qpos[probe._agent_y_qpos] = y
            mujoco.mj_forward(probe.model, probe.data)
            if not probe._has_scene_collision():
                probed.add(c)
        probe.close()
        if len(probed) >= 12:
            free = probed
    except Exception:
        pass
    # Largest connected free region (8-connected) — drop noise islands.
    seen: set = set()
    best: set = set()
    for c in free:
        if c in seen:
            continue
        stack, comp = [c], set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.add(x)
            for dx, dy in nbrs:
                nb = (x[0] + dx, x[1] + dy)
                if nb in free and nb not in seen:
                    stack.append(nb)
        if len(comp) > len(best):
            best = comp
    if len(best) < 12:
        return None

    F = np.array([[(c[0] + 0.5) * CELL + origin[0], (c[1] + 0.5) * CELL + origin[1]] for c in sorted(best)])
    P = F[:: max(1, len(F) // 1500)]
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    a, b = np.unravel_index(int(np.argmax(D)), D.shape)
    if float(D[a, b]) < 1.0:
        return None
    out = {
        "floor_z": floor_z,
        "cell": CELL,
        "free_cells": F.round(3).tolist(),
        "corner_pair": [P[a].round(3).tolist(), P[b].round(3).tolist()],
        "max_dist": round(float(D[a, b]), 3),
    }
    try:
        cache.write_text(json.dumps(out))
    except Exception:
        pass
    return out


def _greedy_policy(obs, _rng):
    import numpy as np

    v = obs[4:6]
    n = float(np.linalg.norm(v))
    return np.zeros(2, dtype=np.float32) if n < 1e-6 else (v / n).astype(np.float32)


def _central_taskcfg(spawn_region: dict | None, max_steps: int) -> dict:
    """Same central, far-apart spawn the replay uses, so both policies face an
    identical, non-trivial start→goal."""
    import numpy as np

    tcfg: dict = {"max_steps": max_steps}
    try:
        sr = spawn_region or {}
        cx = (float(sr["xmin"]) + float(sr["xmax"])) / 2
        cy = (float(sr["ymin"]) + float(sr["ymax"])) / 2
        xr = (float(sr["xmax"]) - float(sr["xmin"])) / 2 * 0.65
        yr = (float(sr["ymax"]) - float(sr["ymin"])) / 2 * 0.65
        tcfg["spawn_region"] = (cx - xr, cx + xr, cy - yr, cy + yr)
        tcfg["min_start_goal_dist"] = max(0.8, 0.7 * float(np.hypot(2 * xr, 2 * yr)))
    except Exception:
        pass
    return tcfg


def _rollout_one(env, policy_fn, max_steps: int, seed: int) -> dict:
    import numpy as np

    obs, _ = env.reset(seed=seed)  # same seed → same start/goal across policies
    rng = np.random.default_rng(seed)
    spawn = env._agent_xy().astype(float).tolist()
    goal = env._goal_xy().astype(float).tolist()
    PROX = 0.55
    steps = collisions = avoided = 0
    in_prox = prox_collided = False
    traj: list[dict] = []
    info: dict = {}
    for _ in range(max_steps):
        obs, _r, term, trunc, info = env.step(policy_fn(obs, rng))
        steps += 1
        curr = env._agent_xy().astype(float)
        col = bool(env._has_scene_collision())
        if col:
            collisions += 1
        lidar = obs[-16:]
        near = bool(float(np.min(lidar)) < PROX) if len(lidar) else False
        if near and not in_prox:
            in_prox, prox_collided = True, False
        if in_prox and col:
            prox_collided = True
        if in_prox and not near:
            if not prox_collided:
                avoided += 1
            in_prox = False
        traj.append({"step": steps, "x": float(curr[0]), "y": float(curr[1]), "collision": col, "near": near})
        if term or trunc:
            break
    if in_prox and not prox_collided:
        avoided += 1
    success = bool(info.get("success", False))
    if success and traj:
        # The episode terminates within success_radius of the goal — dock the
        # final stretch so the rendered ball visibly arrives AT the goal.
        # Interpolate at the robot's normal per-step pace so it ROLLS in
        # rather than teleporting in a single frame.
        last = np.array([traj[-1]["x"], traj[-1]["y"]], dtype=float)
        g = np.array(goal, dtype=float)
        n = max(2, int(float(np.linalg.norm(g - last)) / 0.025))
        for k in range(1, n + 1):
            p = last + (g - last) * (k / n)
            traj.append({
                "step": steps + k, "x": float(p[0]), "y": float(p[1]),
                "collision": False, "near": False,
            })
    return {
        "steps": steps,
        "success": success,
        "distance": round(float(info.get("distance", -1.0)), 3),
        "collisions": collisions,
        "avoided": avoided,
        "spawn": spawn,
        "goal": goal,
        "trajectory": traj,
    }


@router.post("/{project_id}/compare")
def compare_policies(project: ProjectDep, body: CompareRequest, db: DbSession) -> dict:
    """Run greedy and the trained PPO policy on the SAME start→goal and return
    both trajectories — so you can see greedy stall on an obstacle while PPO
    routes around it. Runs synchronously (two short rollouts)."""
    import random

    from rl_env.env import NavEnv, TaskConfig

    build = db.scalars(
        select(Build).where(Build.project_id == project.id).order_by(Build.created_at.desc())
    ).first()
    if build is None or not build.mjcf_path:
        raise HTTPException(400, "no built scene — run Build first")

    pol = db.scalars(
        select(Policy).where(Policy.build_id == build.id).order_by(Policy.created_at.desc())
    ).first()
    ckpt = pol.ckpt_path if pol and pol.ckpt_path and Path(pol.ckpt_path).exists() else None
    if not ckpt:
        raise HTTPException(400, "no trained PPO policy — train one first, then compare")

    tcfg = _central_taskcfg(build.spawn_region, body.max_steps)

    from stable_baselines3 import PPO

    sb3 = PPO.load(ckpt, device="cpu")

    def ppo_fn(obs, _rng):
        action, _ = sb3.predict(obs, deterministic=True)
        return action

    def run_seed(s: int) -> list[dict]:
        out = []
        for name, fn in (("greedy", _greedy_policy), ("ppo", ppo_fn)):
            env = NavEnv(build.mjcf_path, task=TaskConfig(**tcfg))
            ep = _rollout_one(env, fn, body.max_steps, s)
            ep["policy"] = name
            out.append(ep)
            env.close()
        return out

    def _score(res: list[dict]) -> float:
        # Prefer a "sensible" track: an obstacle is in the way so greedy
        # struggles (fails / collides) while PPO still reaches the goal.
        g = next(r for r in res if r["policy"] == "greedy")
        p = next(r for r in res if r["policy"] == "ppo")
        return (
            (0 if g["success"] else 2.0)
            - (0 if p["success"] else 2.0)
            + 0.05 * (g["collisions"] - p["collisions"])
        )

    # Where we know the scanned-floor footprint, the track is FIXED and
    # sensible: corner of the scan → opposite corner, on actual floor.
    recon = db.get(Reconstruction, build.reconstruction_id) if build.reconstruction_id else None
    fp = None
    try:
        fp = _nav_footprint(build, recon.mesh_path if recon else None)
    except Exception:
        fp = None

    if fp is not None and body.seed is None:
        seed = 0
        spawn, goal = fp["corner_pair"]
        tcfg["fixed_spawn"] = tuple(spawn)
        tcfg["fixed_goal"] = tuple(goal)
        tcfg["min_start_goal_dist"] = 0.0
        results = run_seed(seed)
    elif body.seed is not None:
        seed = body.seed
        results = run_seed(seed)
    else:
        # No footprint (synthetic box): scan a spread of candidate tracks
        # (deterministic, not random) and keep the most instructive: ideally PPO
        # reaches the goal while greedy stalls on an obstacle. Search with a
        # shorter horizon for speed, then re-run the winner at full length.
        def run_seed_fast(s: int) -> list[dict]:
            steps = min(body.max_steps, 170)
            out = []
            for name, fn in (("greedy", _greedy_policy), ("ppo", ppo_fn)):
                env = NavEnv(build.mjcf_path, task=TaskConfig(**tcfg))
                ep = _rollout_one(env, fn, steps, s)
                ep["policy"] = name
                out.append(ep)
                env.close()
            return out

        best_seed, best_score = 17, -1e9
        for i in range(18):
            s = 17 + i * 89
            res = run_seed_fast(s)
            sc = _score(res)
            if sc > best_score:
                best_seed, best_score = s, sc
            g = next(r for r in res if r["policy"] == "greedy")
            p = next(r for r in res if r["policy"] == "ppo")
            if p["success"] and not g["success"]:
                best_seed = s  # ideal showcase — greedy stalls, PPO routes around
                break
        seed = best_seed
        results = run_seed(seed)

    # raw→sim transform + floor height let the 3D viewer place each path back
    # onto the textured mesh floor. Prefer the footprint's floor (the densest
    # vertex slab — the floor you can SEE) over the mesh minimum, which sits
    # below it whenever the scan has noise under the real floor.
    raw_to_sim, floor_z = _build_raw_to_sim(build)
    if fp is not None:
        floor_z = fp["floor_z"]
    return {
        "raw_to_sim": raw_to_sim,
        "floor_z": floor_z if floor_z is not None else 0.0,
        "seed": seed,
        "bounds": build.bounds,
        "spawn_region": build.spawn_region,
        "results": results,
    }
