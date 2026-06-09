"""depth_fusion_da3 — Matthew's depth-fusion pipeline with Depth-Anything-3 depth.

Reuses the FROZEN depth_fusion fusion loop verbatim (SuperPoint + LightGlue +
scale-only calibration + RANSAC/Umeyama + back-projection) but swaps the
per-frame depth front-end from the local Depth-Pro/Depth-Anything-V2 to
Depth-Anything-3 metric depth on Replicate.

It does NOT edit depth_fusion: for the duration of one reconstruct() it rebinds
that module's `infer_depth`/`get_depth_model` names (under a lock, so it can't
leak into a concurrent plain depth_fusion run), so all of Matthew's frozen
fusion logic runs unchanged.

Caveat: DA3 depth is one Replicate prediction PER FRAME — on a low-credit
account that rate-limits hard (6/min). Needs Replicate credit to run smoothly;
for local, use the plain `depth_fusion` backend.
"""
from __future__ import annotations

import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import get_settings
from src.features.reconstruction.backends import _replicate as rep
from src.features.reconstruction.backends import depth_fusion as _df
from src.features.reconstruction.backends import register
from src.features.reconstruction.backends.base import (
    ReconstructionBackend,
    ReconstructionInput,
    ReconstructionOutput,
)

_FOV_DEG_DEFAULT = 60.0
_METRIC_DEPTH_DIVISOR = 300.0  # DA3: metric_depth = focal_px * raw / 300
_DEPTH_KEYS = ("depth", "metric_depth", "predicted_depth", "depth_map")
# Serializes the global rebind so it can't bleed into a concurrent depth_fusion.
_patch_lock = threading.Lock()


def _da3_get_depth_model(name: str = "indoor"):
    """No-op stand-in: DA3 depth is cloud, no local model to preload.
    depth_fusion's preflight calls this only to validate loadability."""
    return {"kind": "cloud-da3"}


def _pick_depth(payload: dict) -> np.ndarray:
    for k in _DEPTH_KEYS:
        if k in payload:
            return rep.decode_array(payload[k])
    for v in payload.values():
        try:
            a = rep.decode_array(v)
        except Exception:
            continue
        if a.ndim == 2:
            return a
    raise RuntimeError(f"no depth array in DA3 output keys: {list(payload)}")


def _da3_infer_depth(image: Image.Image, name: str = "indoor", **_kw):
    """Depth-Anything-3 metric depth via Replicate; returns (depth_m HxW, meta)."""
    settings = get_settings()
    model_ref = settings.replicate_depth_model
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        image.save(tmp.name, format="PNG")
        with open(tmp.name, "rb") as fh:
            out = rep.run_model(
                model_ref,
                {"images": [fh], "to_base64": True, "return_depth": True,
                 "output_format": "json"},
            )
    data = out.get("data") if isinstance(out, dict) else None
    if not data:
        raise RuntimeError(f"DA3 returned no data: {out!r}")
    raw = _pick_depth(rep.fetch_json(data[0])).astype(np.float32)
    while raw.ndim > 2:
        raw = raw.squeeze(0)
    # Scale to metric meters: focal from width + fallback FOV, matching the K
    # depth_fusion builds when no per-frame FOV estimate is available.
    focal_px = image.width / (2.0 * np.tan(np.radians(_FOV_DEG_DEFAULT) / 2.0))
    depth = (raw * focal_px / _METRIC_DEPTH_DIVISOR).astype(np.float32)
    return depth, {"depth_model": model_ref, "depth_shape": list(depth.shape)}


@register
class DepthFusionDA3Backend(ReconstructionBackend):
    name = "depth_fusion_da3"
    requires_gpu = False  # DA3 depth on cloud; SuperPoint/LightGlue + numpy local
    implemented = rep.replicate_available()  # needs a token for DA3 cloud depth

    def reconstruct(
        self,
        inp: ReconstructionInput,
        out_dir: Path,
        progress_cb: Callable[[float, str], None],
    ) -> ReconstructionOutput:
        with _patch_lock:
            orig_infer, orig_get = _df.infer_depth, _df.get_depth_model
            _df.infer_depth = _da3_infer_depth
            _df.get_depth_model = _da3_get_depth_model
            try:
                out = _df.DepthFusionBackend().reconstruct(inp, out_dir, progress_cb)
            finally:
                _df.infer_depth = orig_infer
                _df.get_depth_model = orig_get
        out.backend_meta["actual_backend"] = "depth_fusion_da3"
        out.backend_meta["depth_model"] = get_settings().replicate_depth_model
        return out
