import { Link } from "@tanstack/react-router";
import { Lock } from "lucide-react";

import { useProjectState } from "@/lib/api";
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

// Each step's prerequisite — what must be `complete` upstream before this
// step unlocks. Capture has no prerequisite; Build allows a fixture mesh in
// PR-A so it's also unconditionally reachable.
const PREREQ: Record<StepKey, StepKey | null> = {
  capture: null,
  reconstruct: "capture",
  validate: "reconstruct",
  build: null, // allowed unconditionally — uses fixture room if no reconstruction
  train: "build",
  replay: "build",
};

/** Breadcrumb across the linear flow with stage gating. Future steps are
 * locked (non-navigable, dimmed, lock icon) until their prerequisite stage
 * reports complete via /api/projects/:id/state. */
export function StepNav({ projectId, current }: { projectId: string; current: StepKey }) {
  const { data: state } = useProjectState(projectId);

  return (
    <nav className="flex items-center gap-0 border-b border-border bg-background h-10 px-4 mono text-[11px] tracking-wider">
      {STEPS.map((s, i) => {
        const isCurrent = current === s.key;
        const prereq = PREREQ[s.key];
        const prereqComplete = prereq === null || (state?.[prereq]?.complete ?? false);
        const locked = !prereqComplete && !isCurrent;
        const blockReason = locked ? state?.[prereq!]?.reason : undefined;

        return (
          <div key={s.key} className="flex items-center">
            {i > 0 && <span className="text-muted-foreground/30 mx-2.5">›</span>}
            {locked ? (
              <span
                title={blockReason ?? `Complete ${prereq} first`}
                className="uppercase text-muted-foreground/40 inline-flex items-center gap-1.5 cursor-not-allowed"
              >
                <Lock size={10} strokeWidth={2} />
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
  );
}
