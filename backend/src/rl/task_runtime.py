"""Load and validate generated task modules (restricted exec lands in PR-2)."""
from __future__ import annotations

from src.rl.task_abc import Task


def load(code: str) -> type[Task]:
    """Instantiate a GeneratedTask class from source. Not implemented until PR-2."""
    raise NotImplementedError("task codegen and sandbox load land in PR-2")
