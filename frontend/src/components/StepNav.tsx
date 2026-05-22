/**
 * Stage breadcrumb across a project. Transitive locking: as soon as one
 * step's prerequisite is incomplete, every step after it is locked too.
 * Shows the project name as a small chip above the stages.
 */
import { Link } from "@tanstack/react-router";
import { Lock } from "lucide-react";

import { useProject, useProjectState, type ProjectState } from "@/lib/api";
import { cn } from "@/lib/utils";

const STEPS = [
  { key: "capture", label: "Capture" },
  { key: "reconstruct", label: "Reconstruct" },
  { key: "validate", label: "Validate" },
  { key: "build", label: "Build" },
  { key: "train", label: "Train" },
  { key: "replay", label: "Replay" },
] as const;

type StepKey = (typeof STEPS)[number]["key"];

/** Step N is reachable only if every previous step's `complete` flag is true. */
function computeLocked(state: ProjectState | undefined): Set<StepKey> {
  const locked = new Set<StepKey>();
  if (!state) {
    // Until we know, lock everything but Capture so users land in the right place.
    for (const s of STEPS.slice(1)) locked.add(s.key);
    return locked;
  }
  let earlierLocked = false;
  for (const s of STEPS) {
    if (earlierLocked) {
      locked.add(s.key);
      continue;
    }
    const done = state[s.key]?.complete ?? false;
    if (!done && s.key !== "capture") {
      // The first incomplete step is the "current frontier" — reachable but
      // its descendants are locked.
    }
    if (!done) {
      // Everything past this point is locked.
      earlierLocked = true;
    }
  }
  return locked;
}

export function StepNav({ projectId, current }: { projectId: string; current: StepKey }) {
  const { data: state } = useProjectState(projectId);
  const { data: project } = useProject(projectId);
  const locked = computeLocked(state);
  // Allow the user to stay on the current page even if it's "after" the
  // frontier — otherwise resuming where you left off bounces you.
  locked.delete(current);

  return (
    <header className="border-b border-border bg-background">
      {/* Current project chip — plain label, not a link. Was previously a
          <Link to="/p/$projectId/build"> which sat directly above the
          breadcrumb and stole near-miss clicks aimed at the Reconstruct tab,
          bouncing the user to /build on the first click. Project-level
          navigation is handled by the sidebar. */}
      {project && (
        <div className="px-4 pt-3 pb-1 text-[11px] mono uppercase tracking-wider text-muted-foreground select-none">
          {project.name}
        </div>
      )}

      {/* Stage breadcrumb — wraps on narrow widths instead of clipping */}
      <nav className="flex items-center flex-wrap gap-y-1 gap-x-1.5 px-4 pb-3 mono text-[10px] tracking-wider">
        {STEPS.map((s, i) => {
          const isCurrent = current === s.key;
          const isLocked = locked.has(s.key);

          return (
            <div key={s.key} className="flex items-center">
              {i > 0 && <span className="text-muted-foreground/30 mr-1.5">›</span>}
              {isLocked ? (
                <span
                  title="Complete earlier steps first"
                  className="uppercase text-muted-foreground/40 inline-flex items-center gap-1 cursor-not-allowed"
                >
                  <Lock size={9} strokeWidth={2} />
                  {s.label}
                </span>
              ) : (
                <Link
                  to={`/p/$projectId/${s.key}` as any}
                  params={{ projectId }}
                  className={cn(
                    "uppercase hover:text-foreground transition-colors",
                    isCurrent ? "text-primary" : "text-muted-foreground",
                  )}
                >
                  {s.label}
                </Link>
              )}
            </div>
          );
        })}
      </nav>
    </header>
  );
}
