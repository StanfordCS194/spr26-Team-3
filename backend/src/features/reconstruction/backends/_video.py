"""Shared ffmpeg frame->mp4 encoding for video-input cloud reconstruction models.

A neutral, backend-agnostic helper (mirrors `_replicate.py` / `_geometry.py`
naming). Extracted from the old SfM backend so that deleting it is
decoupled from any consumer that still needs to encode sampled frames into a
video. Kept even if no current backend uses it, so a future video-input model
can reuse it.
"""
from __future__ import annotations

from pathlib import Path


def frames_to_video(frames_dir: Path, fps: float, dest: Path) -> Path:
    """Re-encode sampled frames into an mp4 for video-input cloud models."""
    import ffmpeg  # ffmpeg-python, declared in pyproject

    pattern = None
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        if next(frames_dir.glob(ext), None) is not None:
            pattern = ext
            break
    if pattern is None:
        raise RuntimeError(f"no frames in {frames_dir} to build a video from")

    dest.parent.mkdir(parents=True, exist_ok=True)
    (
        ffmpeg
        .input(str(frames_dir / pattern), pattern_type="glob", framerate=max(fps, 1.0))
        .output(str(dest), vcodec="libx264", pix_fmt="yuv420p", r=max(fps, 1.0))
        .overwrite_output()
        .run(quiet=True)
    )
    return dest
