/**
 * Bare `/p/$projectId` route. Picks the frontier — the last stage the user
 * is allowed to reach given the project's current completion state — and
 * redirects there so opening a project resumes where they left off.
 *
 * Mirrors the locking logic in `StepNav.computeLocked`: walk forward, stop
 * at the first incomplete step. That's the user's "current" step.
 */
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect } from "react";

import { useProjectState } from "@/lib/api";

const STEPS = ["capture", "reconstruct", "validate", "build", "train", "replay"] as const;
type StepKey = (typeof STEPS)[number];

export const Route = createFileRoute("/p/$projectId/")({
  component: ProjectIndex,
});

function ProjectIndex() {
  const { projectId } = Route.useParams();
  const navigate = useNavigate();
  const { data: state } = useProjectState(projectId);

  useEffect(() => {
    if (!state) return;
    let frontier: StepKey = "capture";
    for (const key of STEPS) {
      frontier = key;
      if (!state[key]?.complete) break;
    }
    navigate({
      to: `/p/$projectId/${frontier}` as "/p/$projectId/capture",
      params: { projectId },
      replace: true,
    });
  }, [state, projectId, navigate]);

  return null;
}
