"""CLI entry point: `python -m rl_env <subcommand>`.

Subcommands:
  build  MESH [--out DIR]      mesh -> MJCF + collision hulls
  run    MJCF [--episodes N]   load MJCF, roll random policy, print stats
  demo                          procedural sample room -> full pipeline -> rollout
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from rl_env.build import BuildConfig, build_environment
from rl_env.env import NavEnv, TaskConfig
from rl_env.sample_room import write_sample_room


def _cmd_build(args: argparse.Namespace) -> int:
    cfg = BuildConfig(
        mesh_path=args.mesh,
        out_dir=args.out,
        target_diagonal_m=args.diag,
        up_axis=args.up,
        decompose=not args.no_decompose,
        max_hulls=args.max_hulls,
    )
    artifacts = build_environment(cfg)
    print(f"MJCF written: {artifacts.mjcf_path}")
    print(f"  hulls:        {artifacts.n_hulls}")
    print(f"  bounds (Z-up): {artifacts.bounds.tolist()}")
    print(f"  floor_z:       {artifacts.floor_z:.4f}")
    print(f"  spawn region:  {artifacts.spawn_region}")
    return 0


def _policy_random(obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


def _policy_greedy(obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sanity-check policy: walk straight toward the goal.

    Won't avoid obstacles — but in a clear room, should reach the goal in
    most episodes, proving the env's task and reward are wired correctly.
    """
    goal_vec = obs[4:6]
    norm = float(np.linalg.norm(goal_vec))
    if norm < 1e-6:
        return np.zeros(2, dtype=np.float32)
    return (goal_vec / norm).astype(np.float32)


def _cmd_run(args: argparse.Namespace) -> int:
    env = NavEnv(args.mjcf, task=TaskConfig(max_steps=args.steps))
    rng = np.random.default_rng(args.seed)
    policy = _policy_greedy if args.policy == "greedy" else _policy_random

    successes = 0
    total_steps = 0
    total_reward = 0.0
    t0 = time.time()
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep if args.seed is not None else None)
        ep_reward = 0.0
        ep_steps = 0
        for _ in range(args.steps):
            a = policy(obs, rng)
            obs, r, term, trunc, info = env.step(a)
            ep_reward += r
            ep_steps += 1
            if term or trunc:
                break
        total_steps += ep_steps
        total_reward += ep_reward
        successes += int(info.get("success", False))
        print(
            f"  ep {ep+1}/{args.episodes}: steps={ep_steps:3d} "
            f"reward={ep_reward:7.2f} dist={info.get('distance', float('nan')):.2f} "
            f"success={info.get('success', False)}"
        )
    elapsed = time.time() - t0
    print()
    print(f"{args.policy} policy over {args.episodes} eps: ")
    print(f"  successes: {successes}/{args.episodes}")
    print(f"  avg reward: {total_reward / args.episodes:.2f}")
    print(f"  total steps: {total_steps}  ({total_steps / max(elapsed, 1e-6):.0f} steps/s)")
    env.close()
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from rl_env.server import serve

    serve(host=args.host, port=args.port)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from rl_env.train import train_ppo

    train_ppo(
        mjcf_path=args.mjcf,
        total_timesteps=args.steps,
        n_envs=args.n_envs,
        max_steps=args.max_steps,
        save_path=args.ckpt,
        seed=args.seed,
        device=args.device,
    )
    return 0


def _cmd_play(args: argparse.Namespace) -> int:
    from rl_env.train import play_policy

    result = play_policy(
        mjcf_path=args.mjcf,
        ckpt_path=args.ckpt,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        deterministic=not args.stochastic,
    )
    for i, e in enumerate(result["episodes"]):
        flag = "✓" if e["success"] else "✗"
        print(f"  ep {i+1:2d}: {e['steps']:3d} steps  r={e['reward']:7.2f}  d={e['distance']:.2f}  {flag}")
    print()
    print(f"PPO policy: {result['successes']}/{result['n_episodes']} success "
          f"· avg reward {result['avg_reward']:.2f} · avg steps {result['avg_steps']:.1f}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    out = Path(args.out)
    sample_path = out / "sample_room.obj"
    write_sample_room(sample_path)
    print(f"sample mesh: {sample_path}")

    build_args = argparse.Namespace(
        mesh=str(sample_path),
        out=str(out / "build"),
        diag=None,
        up="y",
        no_decompose=False,
        max_hulls=64,
    )
    _cmd_build(build_args)
    mjcf = out / "build" / "scene.xml"

    run_args = argparse.Namespace(
        mjcf=str(mjcf),
        episodes=args.episodes,
        steps=args.steps,
        seed=args.seed,
        policy=args.policy,
    )
    return _cmd_run(run_args)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rl_env")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="mesh -> MJCF")
    pb.add_argument("mesh", help="path to .obj/.ply mesh (e.g. prototype export)")
    pb.add_argument("--out", default="build/", help="output directory")
    pb.add_argument("--diag", type=float, default=6.0, help="rescale xy-diagonal to this many meters; 0 to skip")
    pb.add_argument("--up", default="auto", choices=["auto", "y", "z"], help="up axis of input mesh")
    pb.add_argument("--no-decompose", action="store_true", help="skip VHACD; use connected-component hulls")
    pb.add_argument("--max-hulls", type=int, default=64)
    pb.set_defaults(func=_cmd_build)

    pr = sub.add_parser("run", help="rollout on a built MJCF (random or greedy policy)")
    pr.add_argument("mjcf", help="path to scene.xml")
    pr.add_argument("--episodes", type=int, default=5)
    pr.add_argument("--steps", type=int, default=500)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--policy", choices=["random", "greedy"], default="random")
    pr.set_defaults(func=_cmd_run)

    pd = sub.add_parser("demo", help="end-to-end: sample room -> MJCF -> rollout")
    pd.add_argument("--out", default="demo_out/")
    pd.add_argument("--episodes", type=int, default=3)
    pd.add_argument("--steps", type=int, default=300)
    pd.add_argument("--seed", type=int, default=0)
    pd.add_argument("--policy", choices=["random", "greedy"], default="greedy")
    pd.set_defaults(func=_cmd_demo)

    ps = sub.add_parser("serve", help="start local Flask server (browser UI)")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=5174)
    ps.set_defaults(func=_cmd_serve)

    pt = sub.add_parser("train", help="train PPO on a built MJCF")
    pt.add_argument("mjcf", help="path to scene.xml")
    pt.add_argument("--steps", type=int, default=200_000, help="total env steps")
    pt.add_argument("--n-envs", type=int, default=4)
    pt.add_argument("--max-steps", type=int, default=300)
    pt.add_argument("--ckpt", default="policy.zip")
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    pt.set_defaults(func=_cmd_train)

    pp = sub.add_parser("play", help="evaluate a trained policy")
    pp.add_argument("mjcf", help="path to scene.xml")
    pp.add_argument("--ckpt", required=True)
    pp.add_argument("--episodes", type=int, default=10)
    pp.add_argument("--max-steps", type=int, default=500)
    pp.add_argument("--seed", type=int, default=0)
    pp.add_argument("--stochastic", action="store_true", help="sample actions instead of mean")
    pp.set_defaults(func=_cmd_play)

    args = p.parse_args(argv)
    if args.cmd == "build" and args.diag == 0:
        args.diag = None
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
