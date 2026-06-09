"""Shared Replicate cloud-inference helpers for reconstruction backends.

All heavy reconstruction models (VGGT, Depth-Anything, COLMAP, gaussian
splatting) run on Replicate's GPUs, NOT on the dev machine. Loading them
locally OOMs and crashes laptops, so every neural backend goes through here.

`replicate_available()` gates a backend's `implemented` flag on the token being
present, so the API advertises a backend as runnable only when cloud inference
is actually configured.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from src.config import get_settings


def replicate_available() -> bool:
    """True when a Replicate token is configured. Backends use this for their
    `implemented` flag so they never claim to run without cloud access."""
    return bool(get_settings().replicate_api_token)


def _client():
    """Return an authenticated Replicate client, or raise a clear error.

    `replicate` is imported lazily so importing a backend module never requires
    the SDK to be installed (e.g. during unit tests of pure-math helpers).
    """
    token = get_settings().replicate_api_token
    if not token:
        raise RuntimeError(
            "REPLICATE_API_TOKEN is not set. Cloud reconstruction backends need "
            "it — add it to backend/.env. Pick the `demo_fixture` backend to demo "
            "the pipeline without cloud access."
        )
    try:
        import replicate
    except ImportError as e:  # pragma: no cover - dependency is declared
        raise RuntimeError(
            f"replicate SDK not installed: {e}. Run `uv sync` in backend/."
        ) from e
    return replicate.Client(api_token=token)


def run_model(model_ref: str, inputs: dict[str, Any]) -> Any:
    """Run a Replicate model to completion and return its output.

    Blocks until the prediction finishes (Replicate handles the GPU queue).
    File-typed inputs may be passed as open binary handles or Paths; the SDK
    uploads them. Output URIs come back as strings / FileOutput objects.
    """
    return _client().run(model_ref, input=inputs)


def open_files(paths: list[Path]) -> list:
    """Open image/video paths as binary handles for a Replicate file input.

    Caller is responsible for closing them (use within a `with` or close after
    `run_model` returns). Kept separate so backends control the lifetime.
    """
    return [p.open("rb") for p in paths]


def _uri_str(uri: Any) -> str:
    """Coerce a Replicate output entry (str URL or FileOutput) to a URL string."""
    # FileOutput exposes the URL via str() / .url depending on SDK version.
    return getattr(uri, "url", None) or str(uri)


def fetch_json(uri: Any, *, timeout: float = 120.0) -> dict:
    """Download a JSON output file from a Replicate output URI."""
    resp = httpx.get(_uri_str(uri), timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    return resp.json()


def download(uri: Any, dest: Path, *, timeout: float = 600.0) -> Path:
    """Stream a Replicate output file (e.g. .ply/.glb) to `dest`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream(
        "GET", _uri_str(uri), timeout=timeout, follow_redirects=True
    ) as resp:
        resp.raise_for_status()
        with dest.open("wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)
    return dest


def decode_array(obj: Any) -> np.ndarray:
    """Decode a base64-array dict from a Replicate JSON output into an ndarray.

    The vufinder models emit arrays (when `to_base64=true`) as
    `{"data": "<base64>", "shape": [...], "dtype": "float32"}`. Already-decoded
    nested lists are passed straight to `np.asarray`.
    """
    if isinstance(obj, dict) and "data" in obj and "shape" in obj:
        raw = base64.b64decode(obj["data"])
        arr = np.frombuffer(raw, dtype=np.dtype(obj.get("dtype", "float32")))
        return arr.reshape(obj["shape"])
    return np.asarray(obj)


__all__ = [
    "replicate_available",
    "run_model",
    "open_files",
    "fetch_json",
    "download",
    "decode_array",
]
