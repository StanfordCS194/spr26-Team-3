"""ReconstructionBackend ABC + dataclasses.

Every backend takes a directory of sampled frames and produces a mesh and
optional point cloud. Implementations live in sibling modules.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ReconstructionInput:
    frames_dir: Path
    fps_sampled: float
    intrinsics_hint: dict | None = None


@dataclass
class ReconstructionOutput:
    mesh_path: Path
    point_cloud_path: Path | None = None
    camera_poses: dict | None = None
    backend_meta: dict = field(default_factory=dict)


class ReconstructionBackend(ABC):
    name: str = ""  # overridden per subclass
    requires_gpu: bool = False
    implemented: bool = True

    @abstractmethod
    def reconstruct(
        self,
        inp: ReconstructionInput,
        out_dir: Path,
        progress_cb: Callable[[float, str], None],
    ) -> ReconstructionOutput: ...
