# worldscan-v3-rl-authoring — Proposal

## What

Replace the hardcoded "navigate to a fixed goal" RL task with a **natural-language task authoring** flow, and replace the separate 2D trajectory viewer with **agent playback rendered inside the existing 3D mesh viewer**.

The user describes the task in plain English across three fields:

1. **Objective** — what counts as success ("the agent must reach the chair without bumping into it").
2. **Environment constraints** — what's true about the world ("the floor is the only walkable surface; the agent cannot fly or climb").
3. **Agent constraints** — what the agent can do ("the agent is a 30 cm radius circle, can move forward/back/turn, observes a depth image and its own velocity").

A backend AI agent (Claude API call) translates those fields into a structured `Task` definition: reward function, termination predicates, observation builders, action space, spawn/goal sampler. The generated code is reviewable and editable. Training (PPO) consumes that `Task` instead of the hardcoded one.

Replay no longer opens a 2D top-down chart — it plays back the agent's trajectory **inside the persistent 3D MeshViewer** (the one already in the right panel), with a draggable goal marker, a scrubbable timeline, and per-episode selection.

This subsumes the open task "Training: agent goal manipulation in the mesh" (task #22) — goal placement happens via clicking in the 3D mesh viewer during authoring, and replay paints the resulting trajectories back into the same mesh.

## Why now

- **The current task is a toy.** Point-to-goal with a fixed spawn region and a fixed goal cell gives identical-looking runs every time. Users can't ask *their* question — "can the robot reach the chair I scanned?", "does it avoid the doorframe?", "what if it can only see depth, not RGB?". The product feels like a demo, not a tool.
- **Hand-coding tasks doesn't scale.** Every new task today means a backend code change, a redeploy, a new PR. The whole point of WorldScan is that the *world* is user-provided; the *task* must be too. AI-mediated code generation lets us keep tasks in the database alongside the world.
- **The 2D trajectory viewer is a regression from the 3D viewer.** Users already trust the 3D mesh — every other stage shows it. Replay shouldn't yank them out. Mesh playback closes the loop: scan → build → train → *see the agent move through your scan*.

## What it changes

- **NEW** capability `rl-authoring`: task definition CRUD, AI codegen, code review/edit UI, task versioning per project.
- **NEW** capability `replay-mesh-playback`: trajectory rendered into the 3D mesh viewer with playback controls.
- **MODIFIED** capability `rl-training`: training consumes the generated `Task` instead of the hardcoded one; the `policy` row links to its task version.

## Non-goals

- Multi-agent. Single agent only.
- General-purpose reward learning (RLHF, preference labels). The user writes natural language; the LLM writes Python; we run PPO. No human-in-the-loop reward.
- Replacing PPO. Algo stays the same.
- Continuous editing during training. The task is frozen when training starts.

## Open questions

- **Codegen sandboxing.** The AI writes Python that runs in the backend's process. What's our story for malicious / accidentally-destructive code? Likely answer: run codegen output inside a restricted namespace with no filesystem / network access, time-out each reward call.
- **How much pre-flight validation?** Do we require the generated `Task.validate()` to pass on the actual mesh before allowing Train? Suggested yes — saves a wasted PPO run.
- **Goal placement UX.** Click in mesh → ray-cast to floor plane → set goal? Or freeform 3D point that snaps to the navmesh? Start with floor-plane ray-cast (simpler, matches the "ground-plane navigation" task class).
