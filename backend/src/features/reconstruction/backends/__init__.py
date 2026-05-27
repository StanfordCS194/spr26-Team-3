"""Reconstruction backend registry.

Add a new backend by creating `backends/<name>.py` with a class decorated
with `@register`. The class is auto-discovered by the import side-effects
in this file.
"""
from __future__ import annotations

from src.features.reconstruction.backends.base import ReconstructionBackend

_BACKENDS: dict[str, type[ReconstructionBackend]] = {}


def register(cls: type[ReconstructionBackend]) -> type[ReconstructionBackend]:
    _BACKENDS[cls.name] = cls
    return cls


def list_backends() -> list[dict]:
    return [
        {"name": cls.name, "implemented": cls.implemented, "requires_gpu": cls.requires_gpu}
        for cls in _BACKENDS.values()
    ]


def get_backend(name: str) -> ReconstructionBackend:
    if name not in _BACKENDS:
        raise ValueError(f"unknown backend {name!r}; have {list(_BACKENDS)}")
    return _BACKENDS[name]()


# Import side-effect: registers all known backends.
from src.features.reconstruction.backends import (  # noqa: E402, F401
    colmap,
    depth_fusion,
    splat,
    vggt,
)
