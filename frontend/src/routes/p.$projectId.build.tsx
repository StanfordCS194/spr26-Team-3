import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState } from "react";

import { ProjectSceneLayout } from "@/components/ProjectSceneLayout";
import { StatusDot } from "@/components/StatusDot";
import { useBuild, useLatestBuild } from "@/lib/api";

export const Route = createFileRoute("/p/$projectId/build")({
  component: BuildScreen,
});

function BuildScreen() {
  const { projectId } = Route.useParams();
  const build = useBuild(projectId);
  const { data: latest } = useLatestBuild(projectId);
  const navigate = useNavigate();
  const [enclose, setEnclose] = useState(true);

  const running = latest?.status === "pending" || latest?.status === "running";

  return (
    <ProjectSceneLayout projectId={projectId} current="build">
      <header>
        <h1 className="text-2xl">Build</h1>
        <p className="text-sm text-muted-foreground mt-1.5">
          Turn the validated mesh into an MJCF physics scene with convex
          collision hulls.
        </p>
      </header>

      <label className="mt-6 flex items-start gap-2 text-sm max-w-md cursor-pointer">
        <input
          type="checkbox"
          checked={enclose}
          disabled={running}
          onChange={(e) => setEnclose(e.target.checked)}
          className="mt-0.5 accent-[var(--primary)]"
        />
        <span>
          Enclose with invisible boundary walls
          <span className="block text-xs text-muted-foreground">
            Keeps the agent inside partial/open scans (e.g. a half-recorded
            room). Recommended.
          </span>
        </span>
      </label>

      <button
        disabled={build.isPending || running}
        onClick={() => build.mutate({ enclose })}
        className="mt-4 px-4 py-2 rounded-sm border-2 border-primary text-primary text-sm font-medium hover:bg-primary/10 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
      >
        {running ? "Building…" : build.isPending ? "Queuing…" : latest ? "Re-build env" : "Build env"}
      </button>

      {build.error && (
        <p className="mt-3 text-sm text-[var(--status-fail)] mono">
          {String((build.error as Error).message)}
        </p>
      )}

      {latest && (
        <section className="mt-8 border border-border rounded-sm">
          <header className="px-4 py-2 border-b border-border flex items-center justify-between">
            <span className="mono text-[11px] uppercase tracking-wider text-muted-foreground">
              latest build
            </span>
            <StatusDot
              status={
                latest.status === "ok"
                  ? "ok"
                  : latest.status === "failed"
                  ? "fail"
                  : "warn"
              }
              label={latest.status}
            />
          </header>
          {latest.status === "failed" && (
            <p className="px-4 py-3 text-xs text-[var(--status-fail)] mono">
              {latest.error ?? "unknown error"}
            </p>
          )}
          {latest.status === "ok" && (
            <>
              <dl className="grid grid-cols-2 gap-y-2 gap-x-4 p-4 text-xs mono">
                <dt className="text-muted-foreground">id</dt>
                <dd className="break-all">{latest.id}</dd>
                <dt className="text-muted-foreground">hulls</dt>
                <dd>{latest.n_hulls}</dd>
                {latest.bounds && (
                  <>
                    <dt className="text-muted-foreground">bounds</dt>
                    <dd className="text-[10px]">
                      [{latest.bounds.min.map((n) => n.toFixed(2)).join(", ")}] → [
                      {latest.bounds.max.map((n) => n.toFixed(2)).join(", ")}]
                    </dd>
                  </>
                )}
                {latest.spawn_region && (
                  <>
                    <dt className="text-muted-foreground">spawn</dt>
                    <dd className="text-[10px]">
                      x ∈ [{latest.spawn_region.xmin.toFixed(2)},{" "}
                      {latest.spawn_region.xmax.toFixed(2)}], y ∈ [
                      {latest.spawn_region.ymin.toFixed(2)},{" "}
                      {latest.spawn_region.ymax.toFixed(2)}]
                    </dd>
                  </>
                )}
              </dl>
              <footer className="px-4 py-3 border-t border-border">
                <button
                  onClick={() => navigate({ to: "/p/$projectId/train", params: { projectId } })}
                  className="px-3 py-1.5 rounded-sm border border-border text-xs hover:bg-accent"
                >
                  Continue to Train →
                </button>
              </footer>
            </>
          )}
        </section>
      )}
    </ProjectSceneLayout>
  );
}
