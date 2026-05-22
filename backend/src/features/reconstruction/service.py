"""Frame extraction + reconstruction orchestration."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def extract_frames(video_path: Path, out_dir: Path, n_frames: int = 24, fps_target: float = 4.0) -> list[Path]:
    """Sample `n_frames` evenly from the video using ffmpeg.

    Returns a list of frame paths on disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Wipe any previous frames
    for p in out_dir.iterdir():
        if p.is_file():
            p.unlink()

    # Get duration to compute "every N seconds" if possible. Fall back to fps.
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, check=True,
        )
        duration = float(probe.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        duration = 0.0

    if duration > 0:
        step = max(duration / n_frames, 1.0 / fps_target)
        vf = f"fps=1/{step}"
    else:
        vf = f"fps={fps_target}"

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", vf,
        "-frames:v", str(n_frames),
        "-q:v", "3",
        str(out_dir / "frame_%04d.jpg"),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode(errors='ignore')[-500:]}") from e

    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        # Last-ditch: maybe input is already an image — copy it
        if video_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            target = out_dir / f"frame_0001{video_path.suffix.lower()}"
            shutil.copy(video_path, target)
            frames = [target]
    if not frames:
        raise RuntimeError("no frames produced from input")
    return frames


def write_thumbnail(frame_path: Path, out_path: Path) -> None:
    """Copy or convert the first frame to a thumbnail PNG."""
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(frame_path)
    img.thumbnail((256, 192))
    img.save(out_path, "PNG")
