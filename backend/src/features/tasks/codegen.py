"""Claude-backed task codegen — builds prompts and returns Python modules."""
from __future__ import annotations

import inspect
import logging
from pathlib import Path

from src.config import get_settings
from src.rl.legacy_tasks import NavToGoalTask
from src.rl.task_abc import Task, TaskContext
from src.rl.task_runtime import extract_python_module

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-20250514"


def _read_reference_source() -> str:
    task_abc = Path(inspect.getfile(Task)).read_text()
    legacy = Path(inspect.getfile(NavToGoalTask)).read_text()
    return f"### task_abc.py\n{task_abc}\n\n### legacy_tasks.py (reference)\n{legacy}"


def build_system_prompt() -> str:
    return f"""You write Python for WorldScan RL task authoring.

Output ONE module only. Requirements:
- Define exactly: class GeneratedTask(Task)
- Do NOT use import or from statements (numpy, mujoco, Task, spaces, mujoco_view are pre-injected)
- Use self._model (mujoco.MjModel) set by the runtime before reset(); pass it to mujoco_view.*(self._model, mj_data)
- Implement: reset, observe, reward, terminated, truncated; optional validate
- Class attributes: obs_space, action_space, horizon (int)
- Match the reference NavToGoalTask style unless the user's NL clearly requires something else

Reference sources:
{_read_reference_source()}
"""


def build_user_prompt(
    *,
    objective_nl: str,
    env_nl: str,
    agent_nl: str,
    ctx: TaskContext,
) -> str:
    goal_line = f"Fixed goal (world XY): {ctx.goal_3d}" if ctx.goal_3d else "Goal: sampled randomly in spawn region each episode."
    return f"""Write GeneratedTask for this scene.

MJCF path: {ctx.mjcf_path}
Bounds: {ctx.bounds}
Spawn region: {ctx.spawn_region}
{goal_line}

Objective: {objective_nl or '(default: reach the goal)'}

Environment constraints: {env_nl or '(default: floor navigation, static scene)'}

Agent constraints: {agent_nl or '(default: holonomic disk, 2D velocity actions, lidar obs)'}

Return only Python code for the module."""


def call_claude(system: str, user: str, model: str | None = None) -> tuple[str, str]:
    """Returns (raw_text, model_id). Raises if API key missing or request fails."""
    settings = get_settings()
    api_key = getattr(settings, "anthropic_api_key", "") or ""
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set — cannot run task codegen")

    model_id = model or getattr(settings, "anthropic_task_model", None) or DEFAULT_MODEL

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model_id,
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts = [b.text for b in msg.content if hasattr(b, "text")]
    raw = "\n".join(parts).strip()
    log.info("codegen model=%s input_chars=%d output_chars=%d", model_id, len(user), len(raw))
    return raw, model_id


def generate_module_code(ctx: TaskContext, objective_nl: str, env_nl: str, agent_nl: str) -> tuple[str, str, str]:
    """Returns (code, raw_response, model_id)."""
    system = build_system_prompt()
    user = build_user_prompt(
        objective_nl=objective_nl,
        env_nl=env_nl,
        agent_nl=agent_nl,
        ctx=ctx,
    )
    raw, model_id = call_claude(system, user, model=None)
    code = extract_python_module(raw)
    return code, raw, model_id
