"""Load and validate generated task modules in a restricted namespace."""
from __future__ import annotations

import concurrent.futures
import re
import traceback
from typing import Any, Callable, TypeVar

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from numpy.random import Generator

from src.rl import mujoco_view
from src.rl.task_abc import Task, TaskContext, nav_action_space, nav_obs_space

T = TypeVar("T")

# Curated builtins — no import, open, eval, exec, compile, __import__.
_SAFE_BUILTINS: dict[str, Any] = {
    "__build_class__": __build_class__,
    "None": None,
    "True": True,
    "False": False,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "type": type,
    "Exception": Exception,
    "ValueError": ValueError,
    "RuntimeError": RuntimeError,
    "super": super,
}

_CALL_TIMEOUT_S = 0.05
_DRY_RUN_STEPS = 10


class TaskRuntimeError(Exception):
    pass


class TaskTimeoutError(TaskRuntimeError):
    pass


def _call_timed(fn: Callable[[], T], label: str, timeout_s: float = _CALL_TIMEOUT_S) -> T:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(fn)
        try:
            return fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError as e:
            raise TaskTimeoutError(f"{label} exceeded {timeout_s}s") from e


def extract_python_module(text: str) -> str:
    """Pull Python from a markdown fenced block if the model wrapped it."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def load(code: str) -> type[Task]:
    """Exec generated source and return the GeneratedTask class."""
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            raise TaskRuntimeError("generated code must not use import statements")
        namespace: dict[str, Any] = {
            "__name__": "generated_task",
            "__builtins__": _SAFE_BUILTINS,
            "np": np,
        "numpy": np,
        "mujoco": mujoco,
        "gym": gym,
        "spaces": spaces,
        "Task": Task,
        "TaskContext": TaskContext,
        "nav_obs_space": nav_obs_space,
        "nav_action_space": nav_action_space,
        "mujoco_view": mujoco_view,
        "Generator": Generator,
    }
    try:
        exec(code, namespace)  # noqa: S102 — intentional sandboxed exec
    except Exception as e:
        raise TaskRuntimeError(f"exec failed: {e}") from e

    cls = namespace.get("GeneratedTask")
    if cls is None:
        raise TaskRuntimeError("module must define class GeneratedTask(Task)")
    if not isinstance(cls, type) or not issubclass(cls, Task):
        raise TaskRuntimeError("GeneratedTask must subclass Task")
    return cls


def dry_run(
    code: str,
    mjcf_path: str,
    ctx: TaskContext,
    *,
    steps: int = _DRY_RUN_STEPS,
    seed: int = 0,
) -> list[str]:
    """Load code, instantiate, reset + N physics steps. Returns validate() warnings."""
    task_cls = load(code)
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    mj_data = mujoco.MjData(model)
    mujoco.mj_resetData(model, mj_data)
    rng = np.random.default_rng(seed)

    task = task_cls(ctx)
    task._model = model  # runtime injects MjModel before hooks (codegen uses self._model)

    def _reset():
        return task.reset(mj_data, rng)

    _call_timed(_reset, "reset")
    mujoco_view.forward(model, mj_data)

    zero = np.zeros(int(task.action_space.shape[0]), dtype=np.float32)
    for step_i in range(1, steps + 1):

        def _step_hooks():
            task.observe(mj_data)
            task.reward(mj_data, zero)
            task.terminated(mj_data)
            task.truncated(mj_data, step_i)

        _call_timed(_step_hooks, f"step-{step_i}")
        mujoco_view.step_zero_ctrl(model, mj_data)

    warnings = task.validate(mjcf_path)
    return warnings


def validate_and_dry_run(code: str, mjcf_path: str, ctx: TaskContext) -> None:
    """Raises TaskRuntimeError on failure."""
    try:
        dry_run(code, mjcf_path, ctx)
    except TaskRuntimeError:
        raise
    except Exception as e:
        raise TaskRuntimeError(traceback.format_exc()[-1000:]) from e
