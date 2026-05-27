"""Flask server: exposes the rl_env pipeline behind a small HTTP API and
serves the prototype HTML same-origin, so the browser can fetch /api/...
without CORS gymnastics.

Routes:
  GET  /                    redirect to prototype/v4.html
  GET  /prototype/<path>    static file under prototype/
  POST /api/build           multipart: file=mesh; form: up, diag, max_hulls
  POST /api/run             json: build_id, episodes, steps, policy, seed
  POST /api/depth           multipart: image=jpg/png; returns f32 depth in meters
  GET  /api/builds          list known builds (debug)
"""
from __future__ import annotations

import json
import secrets
import subprocess
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, redirect, request, send_from_directory

from rl_env.build import BuildConfig, build_environment
from rl_env.env import NavEnv, TaskConfig

# File lives at rl_env/rl_env/server.py; repo root is three parents up.
ROOT = Path(__file__).resolve().parent.parent.parent
PROTOTYPE_DIR = ROOT / "prototype"
BUILDS_DIR = ROOT / "server_builds"
MODELS_CACHE_DIR = ROOT / "server_builds" / "models"

# ONNX models pulled on first request and cached on disk. GitHub release assets
# don't serve CORS headers, so we proxy them through the same origin as the page.
MODEL_URLS = {
    "superpoint": "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v1.0.0/superpoint.onnx",
    "superpoint_lightglue": "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v0.1.3/superpoint_lightglue.onnx",
}

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB upload cap


@app.get("/")
def index():
    return redirect("/prototype/v4.html")


@app.get("/prototype/<path:p>")
def proto(p: str):
    return send_from_directory(PROTOTYPE_DIR, p)


@app.get("/api/models/<name>")
def get_model(name: str):
    if name not in MODEL_URLS:
        return jsonify({"error": f"unknown model: {name}"}), 404
    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = MODELS_CACHE_DIR / f"{name}.onnx"
    if not cache_path.exists():
        url = MODEL_URLS[name]
        tmp_path = cache_path.with_suffix(".onnx.partial")
        print(f"[models] downloading {name} from {url}")
        # Use curl rather than urllib: macOS Python installs ship without a CA
        # bundle by default, so urllib fails with CERTIFICATE_VERIFY_FAILED.
        try:
            subprocess.run(
                ["curl", "-fsSL", "-o", str(tmp_path), url], check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            tmp_path.unlink(missing_ok=True)
            return jsonify({"error": f"download failed: {e}"}), 502
        tmp_path.rename(cache_path)
        print(f"[models] cached {name} -> {cache_path} ({cache_path.stat().st_size} bytes)")
    return send_from_directory(
        cache_path.parent, cache_path.name, mimetype="application/octet-stream"
    )


# Two depth backends, both lazy-loaded and cached in memory:
#  - "indoor": Depth-Anything-V2-Metric-Indoor-Small (~25M params, ~100 MB,
#    ~500 MB RAM). Faster, used by prototype/v4.html (stable). No FOV estimate.
#  - "pro": Apple Depth-Pro (~1B params, ~1 GB, ~2 GB RAM). Stronger on flat
#    indoor surfaces and emits an estimated horizontal FOV per image. Used by
#    prototype/v4.2.html (work-in-progress).
# Cache key: model name -> dict of backend-specific handles. Both can coexist
# in RAM if the user toggles between v4 and v4.2.
_depth_cache: dict = {}
_depth_device = None


def _pick_device():
    global _depth_device
    if _depth_device is not None:
        return _depth_device
    import torch
    if torch.cuda.is_available():
        _depth_device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        _depth_device = "mps"
    else:
        _depth_device = "cpu"
    return _depth_device


def _ensure_depth_model(name: str):
    if name in _depth_cache:
        return _depth_cache[name]
    device = _pick_device()
    t = time.time()
    if name == "indoor":
        from transformers import pipeline
        print(f"[depth] loading Depth-Anything-V2-Metric-Indoor-Small on device={device}…")
        pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
            device=device,
        )
        _depth_cache[name] = {"kind": "pipeline", "pipe": pipe}
    elif name == "pro":
        import torch
        from transformers import DepthProForDepthEstimation, DepthProImageProcessor
        print(f"[depth] loading Depth-Pro (1B params, ~1 GB) on device={device}…")
        proc = DepthProImageProcessor.from_pretrained("apple/DepthPro-hf")
        model = (
            DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf")
            .to(device)
            .eval()
        )
        _depth_cache[name] = {"kind": "explicit", "proc": proc, "model": model, "device": device}
    else:
        raise ValueError(f"unknown depth model: {name}")
    print(f"[depth] loaded {name} in {time.time() - t:.1f}s")
    return _depth_cache[name]


@app.post("/api/depth")
def api_depth():
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' file field"}), 400
    name = request.args.get("model", "indoor")
    if name not in ("indoor", "pro"):
        return jsonify({"error": f"unknown model: {name}"}), 400
    import torch
    from PIL import Image
    img = Image.open(request.files["image"].stream).convert("RGB")
    handle = _ensure_depth_model(name)

    t = time.time()
    fov_deg = None
    if handle["kind"] == "pipeline":
        # Simple transformers pipeline path (Depth-Anything-V2-Metric-Indoor).
        out = handle["pipe"](img)
        pred = out["predicted_depth"]
        depth = pred.detach().cpu().numpy() if hasattr(pred, "detach") else np.asarray(pred)
    else:
        # Explicit Depth-Pro path; also surface its FOV estimate.
        proc, model, device = handle["proc"], handle["model"], handle["device"]
        inputs = proc(images=img, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        post = proc.post_process_depth_estimation(
            outputs, target_sizes=[(img.height, img.width)]
        )[0]
        depth = post["predicted_depth"].detach().cpu().numpy()
        fov_value = post.get("field_of_view") or post.get("fov")
        if fov_value is not None:
            fov_deg = float(fov_value.item() if hasattr(fov_value, "item") else fov_value)

    depth = np.asarray(depth, dtype=np.float32)
    while depth.ndim > 2:
        depth = depth.squeeze(0)
    h, w = depth.shape
    body = depth.tobytes()

    fov_log = f", fov={fov_deg:.1f}°" if fov_deg is not None else ""
    print(
        f"[depth] {name}: {img.size[0]}x{img.size[1]} -> {w}x{h} in {time.time() - t:.2f}s "
        f"(range {float(depth.min()):.2f}-{float(depth.max()):.2f} m{fov_log})"
    )
    headers = {
        "Content-Type": "application/octet-stream",
        "Content-Length": str(len(body)),
        "X-Depth-Width": str(w),
        "X-Depth-Height": str(h),
    }
    if fov_deg is not None:
        headers["X-Estimated-Fov-Deg"] = f"{fov_deg:.4f}"
        headers["Access-Control-Expose-Headers"] = "X-Depth-Width,X-Depth-Height,X-Estimated-Fov-Deg"
    return body, 200, headers


@app.get("/api/builds")
def list_builds():
    if not BUILDS_DIR.exists():
        return jsonify([])
    out = []
    for d in sorted(BUILDS_DIR.iterdir()):
        meta_path = d / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            out.append({"build_id": d.name, **meta})
    return jsonify(out)


@app.post("/api/build")
def api_build():
    if "mesh" not in request.files:
        return jsonify({"error": "missing 'mesh' file field"}), 400
    f = request.files["mesh"]
    if not f.filename:
        return jsonify({"error": "empty filename"}), 400

    up = request.form.get("up", "auto")
    diag_raw = request.form.get("diag", "6.0")
    try:
        diag = float(diag_raw)
    except ValueError:
        return jsonify({"error": f"invalid diag={diag_raw!r}"}), 400
    max_hulls = int(request.form.get("max_hulls", "64"))

    bid = secrets.token_hex(6)
    out = BUILDS_DIR / bid
    # Save the upload outside the build dir — `build_environment` rmtrees
    # `out_dir` before writing artifacts, which would also blow away the input.
    uploads_dir = BUILDS_DIR / "_uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(f.filename).suffix.lower() or ".obj"
    mesh_path = uploads_dir / f"{bid}{suffix}"
    f.save(str(mesh_path))

    t0 = time.time()
    cfg = BuildConfig(
        mesh_path=str(mesh_path),
        out_dir=str(out),
        up_axis=up,
        target_diagonal_m=diag if diag > 0 else None,
        max_hulls=max_hulls,
    )
    try:
        artifacts = build_environment(cfg)
    except Exception as e:
        return jsonify({"error": f"build failed: {e}"}), 500
    elapsed = time.time() - t0

    return jsonify(
        {
            "build_id": bid,
            "n_hulls": artifacts.n_hulls,
            "bounds_min": artifacts.bounds[0].tolist(),
            "bounds_max": artifacts.bounds[1].tolist(),
            "floor_z": artifacts.floor_z,
            "spawn_region": list(artifacts.spawn_region),
            "elapsed_s": round(elapsed, 3),
        }
    )


def _policy_random(obs, rng):
    return rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


def _policy_greedy(obs, rng):
    goal_vec = obs[4:6]
    norm = float(np.linalg.norm(goal_vec))
    if norm < 1e-6:
        return np.zeros(2, dtype=np.float32)
    return (goal_vec / norm).astype(np.float32)


@app.post("/api/run")
def api_run():
    payload = request.get_json(silent=True) or {}
    bid = payload.get("build_id")
    if not bid:
        return jsonify({"error": "missing build_id"}), 400
    mjcf = BUILDS_DIR / bid / "scene.xml"
    if not mjcf.exists():
        return jsonify({"error": f"unknown build {bid}"}), 404
    meta_path = BUILDS_DIR / bid / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    episodes = int(payload.get("episodes", 5))
    steps = int(payload.get("steps", 500))
    seed = int(payload.get("seed", 0))
    policy_name = payload.get("policy", "greedy")
    include_trace = bool(payload.get("include_trace", False))
    trace_episode = int(payload.get("trace_episode", 0))
    policy_fn = _policy_greedy if policy_name == "greedy" else _policy_random

    if policy_name == "ppo":
        ckpt = BUILDS_DIR / bid / "policy.zip"
        if not ckpt.exists():
            return jsonify({"error": "no trained policy for this build — click Train PPO first"}), 400
        from stable_baselines3 import PPO

        ppo_model = PPO.load(str(ckpt), device="auto")

        def policy_fn(obs, _rng):
            action, _ = ppo_model.predict(obs, deterministic=True)
            return action

    env = NavEnv(str(mjcf), task=TaskConfig(max_steps=steps))
    rng = np.random.default_rng(seed)

    eps = []
    successes = 0
    replay: dict | None = None
    t0 = time.time()
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_steps = 0
        info: dict = {}
        trace = []
        if include_trace and ep == trace_episode:
            trace.append(
                {
                    "step": 0,
                    "agent_xy": env._agent_xy().astype(float).tolist(),
                    "goal_xy": env._goal_xy().astype(float).tolist(),
                    "distance": float(np.linalg.norm(env._agent_xy() - env._goal_xy())),
                    "collision": bool(env._has_scene_collision()),
                    "reward": 0.0,
                }
            )
        for _ in range(steps):
            a = policy_fn(obs, rng)
            obs, r, term, trunc, info = env.step(a)
            ep_reward += r
            ep_steps += 1
            if include_trace and ep == trace_episode:
                trace.append(
                    {
                        "step": ep_steps,
                        "agent_xy": env._agent_xy().astype(float).tolist(),
                        "goal_xy": env._goal_xy().astype(float).tolist(),
                        "distance": float(info.get("distance", -1.0)),
                        "collision": bool(env._has_scene_collision()),
                        "reward": float(r),
                    }
                )
            if term or trunc:
                break
        eps.append(
            {
                "steps": ep_steps,
                "reward": float(round(ep_reward, 3)),
                "distance": float(round(info.get("distance", -1), 3)),
                "success": bool(info.get("success", False)),
            }
        )
        if include_trace and ep == trace_episode:
            replay = {
                "episode": ep,
                "policy": policy_name,
                "steps": ep_steps,
                "success": bool(info.get("success", False)),
                "bounds_min": meta.get("bounds_min"),
                "bounds_max": meta.get("bounds_max"),
                "trace": trace,
            }
        successes += int(info.get("success", False))
    env.close()
    elapsed = time.time() - t0

    return jsonify(
        {
            "policy": policy_name,
            "episodes": eps,
            "successes": successes,
            "n_episodes": episodes,
            "avg_reward": float(round(sum(e["reward"] for e in eps) / max(len(eps), 1), 3)),
            "elapsed_s": round(elapsed, 3),
            "replay": replay,
        }
    )


@app.post("/api/frames")
def api_frames():
    import base64
    import io
    from PIL import Image

    payload = request.get_json(silent=True) or {}
    bid = payload.get("build_id")
    if not bid:
        return jsonify({"error": "missing build_id"}), 400
    mjcf = BUILDS_DIR / bid / "scene.xml"
    if not mjcf.exists():
        return jsonify({"error": f"unknown build {bid}"}), 404

    steps = int(payload.get("steps", 300))
    seed = int(payload.get("seed", 0))
    policy_name = payload.get("policy", "greedy")
    max_frames = int(payload.get("max_frames", 40))
    width = int(payload.get("width", 320))
    height = int(payload.get("height", 240))

    policy_fn = _policy_greedy if policy_name == "greedy" else _policy_random
    if policy_name == "ppo":
        ckpt = BUILDS_DIR / bid / "policy.zip"
        if not ckpt.exists():
            return jsonify({"error": "no trained policy — click Train PPO first"}), 400
        from stable_baselines3 import PPO
        ppo_model = PPO.load(str(ckpt), device="auto")
        def policy_fn(obs, _):
            action, _state = ppo_model.predict(obs, deterministic=True)
            return action

    env = NavEnv(str(mjcf), task=TaskConfig(max_steps=steps), render_mode="rgb_array", render_size=(width, height))
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)

    skip = max(1, steps // max_frames)
    frames_b64 = []
    ep_steps = 0
    info: dict = {}

    def capture():
        frame = env.render()
        if frame is None:
            return
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        frames_b64.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())

    capture()
    for i in range(steps):
        a = policy_fn(obs, rng)
        obs, _, term, trunc, info = env.step(a)
        ep_steps += 1
        if (i + 1) % skip == 0:
            capture()
        if term or trunc:
            break
    capture()

    env.close()
    return jsonify({
        "frames": frames_b64,
        "n_frames": len(frames_b64),
        "steps": ep_steps,
        "success": bool(info.get("success", False)),
        "policy": policy_name,
    })


@app.post("/api/train")
def api_train():
    payload = request.get_json(silent=True) or {}
    bid = payload.get("build_id")
    if not bid:
        return jsonify({"error": "missing build_id"}), 400
    mjcf = BUILDS_DIR / bid / "scene.xml"
    if not mjcf.exists():
        return jsonify({"error": f"unknown build {bid}"}), 404

    total_steps = int(payload.get("steps", 100_000))
    n_envs = int(payload.get("n_envs", 4))
    max_steps = int(payload.get("max_steps", 300))
    seed = int(payload.get("seed", 0))
    device = payload.get("device", "cpu")
    min_start_goal_dist = float(payload.get("min_start_goal_dist", 0.8))
    max_start_goal_dist_raw = payload.get("max_start_goal_dist", 0)
    max_start_goal_dist = float(max_start_goal_dist_raw) if float(max_start_goal_dist_raw) > 0 else None

    ckpt_path = BUILDS_DIR / bid / "policy.zip"

    from rl_env.train import train_ppo

    t0 = time.time()
    try:
        train_ppo(
            mjcf_path=str(mjcf),
            total_timesteps=total_steps,
            n_envs=n_envs,
            max_steps=max_steps,
            save_path=str(ckpt_path),
            seed=seed,
            device=device,
            verbose=0,
            min_start_goal_dist=min_start_goal_dist,
            max_start_goal_dist=max_start_goal_dist,
        )
    except Exception as e:
        return jsonify({"error": f"train failed: {e}"}), 500
    elapsed = time.time() - t0

    return jsonify(
        {
            "build_id": bid,
            "steps": total_steps,
            "elapsed_s": round(elapsed, 2),
            "fps": int(total_steps / max(elapsed, 1e-6)),
        }
    )


def serve(host: str = "127.0.0.1", port: int = 5174) -> None:
    BUILDS_DIR.mkdir(exist_ok=True)
    print(f"WorldScan server: http://{host}:{port}/")
    print(f"  prototype:   http://{host}:{port}/prototype/v4.html")
    print(f"  build dir:   {BUILDS_DIR}")
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    serve()
