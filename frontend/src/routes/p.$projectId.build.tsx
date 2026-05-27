import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";

import { StatusDot } from "@/components/StatusDot";
import { StepNav } from "@/components/StepNav";
import { useBuild, useProject } from "@/lib/api";

export const Route = createFileRoute("/p/$projectId/build")({
  component: BuildScreen,
});

function BuildScreen() {
  const { projectId } = Route.useParams();
  const { data: project } = useProject(projectId);
  const build = useBuild(projectId);
  const [latestBuild, setLatestBuild] = useState<Awaited<ReturnType<typeof build.mutateAsync>> | null>(null);

  return (
    <>
      <StepNav projectId={projectId} current="build" />
      <div className="flex-1 p-10 overflow-auto">
        <header className="mb-8">
          <h1 className="text-2xl">{project?.name ?? "…"}</h1>
          <p className="text-sm text-muted-foreground mt-1.5">
            Build an MJCF physics scene. In PR-A the project's mesh is the procedural sample
            room. PR-B replaces this with the project's latest reconstruction.
          </p>
        </header>

        <button
          disabled={build.isPending}
          onClick={async () => setLatestBuild(await build.mutateAsync())}
          className="px-3 py-1.5 rounded-sm bg-primary text-primary-foreground text-sm font-medium hover:opacity-90 disabled:opacity-50"
        >
          {build.isPending ? "Building…" : "Build env"}
        </button>

        {build.error && (
          <p className="mt-4 text-sm text-[var(--status-fail)] mono">
            {String((build.error as Error).message)}
          </p>
        )}

        {latestBuild && (
          <section className="mt-8 max-w-xl border border-border rounded-sm">
            <header className="px-4 py-2 border-b border-border flex items-center justify-between">
              <span className="mono text-xs uppercase tracking-wider text-muted-foreground">
                Latest build
              </span>
              <StatusDot status="ok" label="ready" />
            </header>
            <dl className="grid grid-cols-2 gap-y-2 gap-x-4 p-4 text-sm mono">
              <dt className="text-muted-foreground">id</dt>
              <dd>{latestBuild.id}</dd>
              <dt className="text-muted-foreground">hulls</dt>
              <dd>{latestBuild.n_hulls}</dd>
              <dt className="text-muted-foreground">bounds</dt>
              <dd>
                [{latestBuild.bounds.min.map((n: number) => n.toFixed(2)).join(", ")}] →
                [{latestBuild.bounds.max.map((n: number) => n.toFixed(2)).join(", ")}]
              </dd>
              <dt className="text-muted-foreground">spawn</dt>
              <dd>
                x ∈ [{latestBuild.spawn_region.xmin.toFixed(2)}, {latestBuild.spawn_region.xmax.toFixed(2)}],
                {" "}
                y ∈ [{latestBuild.spawn_region.ymin.toFixed(2)}, {latestBuild.spawn_region.ymax.toFixed(2)}]
              </dd>
            </dl>
          </section>
        )}
      </div>
    </>
  );
}
