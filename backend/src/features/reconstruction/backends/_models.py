"""Model loaders + inference for depth-fusion.

Monocular metric depth runs LOCALLY by default — Matthew's Depth-Anything-V2
(or Depth-Pro) pipeline, reused via `rl_env.server` so the legacy Flask
`/api/depth` and this backend share weights. These models are small (~95MB)
and run fine on a laptop CPU/MPS — unlike VGGT, they never crashed anything.

A cloud depth path (`name="cloud"`, Replicate Depth-Anything-V3-metric) is kept
as an option, but it costs one prediction PER FRAME, so it rate-limits hard on
low-credit accounts. Local is the default. SuperPoint + LightGlue stay local
(tiny ONNX, CPU).
"""
from __future__ import annotations

import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import get_settings
from src.features.reconstruction.backends import _replicate as rep

# Process-wide caches, guarded so concurrent Inngest workers don't double-load.
_depth_lock = threading.Lock()
_onnx_lock = threading.Lock()
_onnx_sessions: dict[str, object] = {}

# Replicate's depth-anything-v3-metric returns a "depth" array that must be
# scaled to true meters: metric_depth = focal_px * depth / 300 (per the model
# README). focal_px is derived from image width + FOV, matching the K that
# depth_fusion builds with `assume_intrinsics`.
_METRIC_DEPTH_DIVISOR = 300.0


def get_depth_model(name: str = "indoor"):
    """Return matthew's lazily-loaded local depth pipeline (`indoor` or `pro`).

    Wraps `rl_env.server._ensure_depth_model` so the legacy Flask `/api/depth`
    endpoint and this backend share the same in-memory weights.
    """
    from rl_env.server import _ensure_depth_model  # type: ignore[attr-defined]
    with _depth_lock:
        return _ensure_depth_model(name)


def infer_depth(
    image: Image.Image, name: str = "indoor", fov_deg: float = 60.0
) -> tuple[np.ndarray, dict]:
    """Run metric depth on a PIL image; return (depth_meters HxW, meta).

    Default is Matthew's LOCAL model (`name` = "indoor"/"pro"). Pass
    `name="cloud"` to use Replicate's Depth-Anything-V3-metric instead.
    """
    if name == "cloud":
        return _infer_depth_cloud(image, fov_deg)

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


def _pick_depth_array(payload: dict) -> np.ndarray:
    """Extract the 2D depth grid from a Replicate depth model's JSON output.

    Prefers an explicit depth-like key, else falls back to the first entry that
    decodes to a 2D array, so we tolerate minor schema differences.
    """
    for key in ("depth", "metric_depth", "predicted_depth", "depth_map"):
        if key in payload:
            return rep.decode_array(payload[key])
    for value in payload.values():
        try:
            arr = rep.decode_array(value)
        except Exception:
            continue
        if arr.ndim == 2:
            return arr
    raise RuntimeError(f"no depth array in Replicate output keys: {list(payload)}")


def _infer_depth_cloud(image: Image.Image, fov_deg: float) -> tuple[np.ndarray, dict]:
    """Cloud depth via Replicate (one prediction per frame — rate-limits)."""
    settings = get_settings()
    model_ref = settings.replicate_depth_model
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        image.save(tmp.name, format="PNG")
        with open(tmp.name, "rb") as fh:
            output = rep.run_model(
                model_ref,
                {"images": [fh], "to_base64": True, "return_depth": True,
                 "output_format": "json"},
            )
    data_uris = output.get("data") if isinstance(output, dict) else None
    if not data_uris:
        raise RuntimeError(f"depth model (Replicate) returned no data: {output!r}")
    raw = _pick_depth_array(rep.fetch_json(data_uris[0])).astype(np.float32)
    while raw.ndim > 2:
        raw = raw.squeeze(0)
    # Scale to metric meters using the focal implied by image width + FOV.
    focal_px = image.width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    depth = (raw * focal_px / _METRIC_DEPTH_DIVISOR).astype(np.float32)
    return depth, {
        "depth_model": model_ref,
        "depth_shape": list(depth.shape),
        "focal_px": float(focal_px),
        "metric_scale": float(focal_px / _METRIC_DEPTH_DIVISOR),
    }


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


def _preprocess_for_superpoint(image: Image.Image, max_side: int = 512) -> tuple[np.ndarray, float]:
    """Grayscale + resize-down to keep the longer side ≤ max_side. Returns
    (image_array float32 0-1, scale_factor) so detected keypoints can be
    mapped back to original coordinates.
    """
    w, h = image.size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    if scale < 1.0:
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    # SuperPoint expects (1, 1, H, W).
    arr = arr[None, None, :, :]
    return arr, scale


def extract_superpoint(image: Image.Image, max_keypoints: int = 1024) -> dict:
    """Run SuperPoint ONNX on a PIL image. Returns
        {"keypoints": (N,2) in ORIGINAL image coords, "descriptors": (N,256), "scores": (N,)}.

    Caps to top-`max_keypoints` by score, matching matthew's heuristic.
    """
    sess = _onnx_session("superpoint")
    arr, scale = _preprocess_for_superpoint(image)
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
    if scale < 1.0:
        kp = kp.astype(np.float32) / scale
    return {"keypoints": kp.astype(np.float32), "descriptors": desc.astype(np.float32), "scores": scores.astype(np.float32)}


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
    matches0 = outs[0]
    # fabio-sim's LightGlue-ONNX export emits `matches0` as a per-keypoint
    # assignment vector of shape (1, N0): entry k is the index of kpts0[k]'s
    # match in kpts1, or -1 if unmatched. Convert to (M, 2) (idx_a, idx_b) pairs.
    if matches0.ndim == 3 and matches0.shape[-1] == 2:
        # Some exports already give (1, M, 2) index pairs — pass through.
        return matches0[0].astype(np.int64)
    assignment = matches0[0] if matches0.ndim == 2 else matches0
    idx_a = np.nonzero(assignment >= 0)[0]
    idx_b = assignment[idx_a]
    return np.stack([idx_a, idx_b], axis=1).astype(np.int64)


__all__ = [
    "get_depth_model",
    "infer_depth",
    "extract_superpoint",
    "match_lightglue",
]
