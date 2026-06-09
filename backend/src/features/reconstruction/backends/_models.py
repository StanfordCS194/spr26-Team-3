"""Model loaders + inference for depth-fusion. Lazy, process-cached.

Depth uses transformers (Depth-Anything-V2-Metric-Indoor-Small by default,
Apple Depth-Pro optional). Feature extraction + matching uses SuperPoint and
LightGlue ONNX models, downloaded from the same URLs matthew's prototype
uses so the legacy Flask server and the new backend share weights.
"""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import numpy as np
from PIL import Image

# rl_env is a workspace member of the backend (see backend/pyproject.toml).
# Imports of `rl_env.server` are deferred to call sites because that module
# pulls in flask at import time, which is only needed for matthew's legacy
# server, not for backend tests of pure-math helpers.

# Process-wide caches (matches matthew's `_depth_cache` pattern). Guarded by
# locks so concurrent Inngest workers don't double-load.
_depth_lock = threading.Lock()
_onnx_lock = threading.Lock()
_onnx_sessions: dict[str, "object"] = {}


def get_depth_model(name: str = "indoor"):
    """Return matthew's lazily-loaded depth pipeline (`indoor` or `pro`).

    Wraps `rl_env.server._ensure_depth_model` so the legacy Flask `/api/depth`
    endpoint and this backend share the same in-memory weights.
    """
    from rl_env.server import _ensure_depth_model  # type: ignore[attr-defined]
    with _depth_lock:
        return _ensure_depth_model(name)


def infer_depth(image: Image.Image, name: str = "indoor") -> tuple[np.ndarray, dict]:
    """Run depth inference on a PIL image; return (depth_meters HxW float32, meta).

    Meta carries the model name, output shape, and optional FOV (Depth-Pro only).
    Mirrors matthew's `rl_env.server.api_depth` handling so outputs match the
    legacy Flask endpoint exactly.
    """
    handle = get_depth_model(name)
    meta: dict = {"depth_model": name}
    if handle["kind"] == "pipeline":
        out = handle["pipe"](image)
        pred = out["predicted_depth"]
        depth = pred.detach().cpu().numpy() if hasattr(pred, "detach") else np.asarray(pred)
    elif handle["kind"] == "explicit":
        import torch  # local: torch is in pyproject but defer importing it
        proc = handle["proc"]
        model = handle["model"]
        device = handle["device"]
        inputs = proc(images=image, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        post = proc.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)]
        )[0]
        depth = post["predicted_depth"].detach().cpu().numpy()
        fov_value = post.get("field_of_view") or post.get("fov")
        if fov_value is not None:
            meta["fov_deg"] = float(fov_value.item() if hasattr(fov_value, "item") else fov_value)
    else:
        raise ValueError(f"unknown depth handle kind {handle.get('kind')!r}")

    depth = np.asarray(depth, dtype=np.float32)
    while depth.ndim > 2:
        depth = depth.squeeze(0)
    meta["depth_shape"] = list(depth.shape)
    return depth, meta


def _models_cache_dir() -> Path:
    """Where ONNX model files live on disk. Mirrors matthew's location so
    the Flask `/api/models/<name>` endpoint can also serve the cached files
    without re-downloading.
    """
    # rl_env.server resolves ROOT relative to itself; we want the same root.
    from rl_env.server import MODELS_CACHE_DIR  # type: ignore[attr-defined]
    return Path(MODELS_CACHE_DIR)


def _download_onnx(name: str) -> Path:
    """Fetch `<name>.onnx` from MODEL_URLS into the shared cache dir if missing."""
    from rl_env.server import MODEL_URLS  # type: ignore[attr-defined]
    if name not in MODEL_URLS:
        raise ValueError(f"unknown ONNX model {name!r}; known: {list(MODEL_URLS)}")
    cache_dir = _models_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{name}.onnx"
    if cache_path.exists():
        return cache_path
    url = MODEL_URLS[name]
    tmp_path = cache_path.with_suffix(".onnx.partial")
    # curl avoids macOS Python's CA-bundle issue (matches matthew's approach).
    subprocess.run(["curl", "-fsSL", "-o", str(tmp_path), url], check=True)
    tmp_path.rename(cache_path)
    return cache_path


def _onnx_session(name: str):
    """Lazy-load an onnxruntime InferenceSession, cached for the process."""
    with _onnx_lock:
        sess = _onnx_sessions.get(name)
        if sess is not None:
            return sess
        import onnxruntime as ort
        path = _download_onnx(name)
        # CPU works everywhere; CUDA EP is preferred when available.
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess = ort.InferenceSession(str(path), providers=providers)
        _onnx_sessions[name] = sess
        return sess


def _preprocess_for_superpoint(image: Image.Image, max_side: int = 1536) -> tuple[np.ndarray, float, float]:
    """Grayscale + resize so the longer side ≤ max_side, with both dims
    quantized to a multiple of 8 (SuperPoint's ONNX export expects /8-divisible
    spatial dims, and the prototype found coarser inputs lose keypoints).

    Returns (image_array float32 0-1 shaped (1,1,H,W), scale_x, scale_y) so
    detected keypoints in the resized grid can be mapped back to original pixel
    coordinates. Mirrors `preprocessForSuperPoint` in prototype/v4.2.html
    (SP_MAX_DIM=1536, round to multiple of 8, min 32).
    """
    w, h = image.size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    new_w = max(32, int(round(w * scale / 8.0)) * 8)
    new_h = max(32, int(round(h * scale / 8.0)) * 8)
    if (new_w, new_h) != (w, h):
        image = image.resize((new_w, new_h), Image.BILINEAR)
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    # SuperPoint expects (1, 1, H, W).
    arr = arr[None, None, :, :]
    return arr, new_w / float(w), new_h / float(h)


def extract_superpoint(image: Image.Image, max_keypoints: int = 1024) -> dict:
    """Run SuperPoint ONNX on a PIL image. Returns
        {"keypoints": (N,2) in ORIGINAL image coords, "descriptors": (N,256), "scores": (N,)}.

    Caps to top-`max_keypoints` by score, matching matthew's heuristic.
    """
    sess = _onnx_session("superpoint")
    arr, scale_x, scale_y = _preprocess_for_superpoint(image)
    inputs = {sess.get_inputs()[0].name: arr}
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(out_names, inputs)
    # ONNX SuperPoint export from fabio-sim/LightGlue-ONNX returns
    # (keypoints[1,N,2], scores[1,N], descriptors[1,N,256]).
    kp = outs[0][0]
    scores = outs[1][0]
    desc = outs[2][0]
    if kp.shape[0] > max_keypoints:
        top = np.argsort(scores)[-max_keypoints:]
        kp, scores, desc = kp[top], scores[top], desc[top]
    # Map keypoints from the (independently quantized) resized grid back to
    # original pixel coordinates. x and y can have slightly different scales.
    kp = kp.astype(np.float32)
    kp[:, 0] /= scale_x
    kp[:, 1] /= scale_y
    return {"keypoints": kp, "descriptors": desc.astype(np.float32), "scores": scores.astype(np.float32)}


def match_lightglue(feat_a: dict, feat_b: dict, image_size: tuple[int, int]) -> np.ndarray:
    """Match two SuperPoint feature sets via LightGlue ONNX. Returns an (M, 2)
    array of (idx_a, idx_b) matched indices into the original feature arrays.

    `image_size` is (width, height) of the original images, used by LightGlue
    to normalize keypoint coordinates.
    """
    sess = _onnx_session("superpoint_lightglue")
    w, h = image_size
    # Normalize keypoints to [-1, 1] as LightGlue expects (per fabio-sim ONNX
    # contract).
    def norm(kp: np.ndarray) -> np.ndarray:
        kp = kp.astype(np.float32).copy()
        kp[:, 0] = (kp[:, 0] - w / 2.0) / (max(w, h) / 2.0)
        kp[:, 1] = (kp[:, 1] - h / 2.0) / (max(w, h) / 2.0)
        return kp[None, ...]  # (1, N, 2)

    inputs = {
        "kpts0": norm(feat_a["keypoints"]),
        "kpts1": norm(feat_b["keypoints"]),
        "desc0": feat_a["descriptors"][None, ...],
        "desc1": feat_b["descriptors"][None, ...],
    }
    # Filter to inputs the model actually accepts.
    expected = {i.name for i in sess.get_inputs()}
    inputs = {k: v for k, v in inputs.items() if k in expected}
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(out_names, inputs)
    # The fabio-sim LightGlue ONNX exports `matches0` (1, M, 2) of (idx_a, idx_b).
    matches = outs[0]
    if matches.ndim == 3:
        matches = matches[0]
    return matches.astype(np.int64)


__all__ = [
    "get_depth_model",
    "infer_depth",
    "extract_superpoint",
    "match_lightglue",
]
