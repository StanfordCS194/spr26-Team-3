import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Lock } from "lucide-react";
import { useState } from "react";

import { MeshViewer } from "@/components/MeshViewer";
import { StatusDot } from "@/components/StatusDot";
import { StepNav } from "@/components/StepNav";
import {
  useBackends,
  useLatestReconstruction,
  useProjectState,
  useReconstruct,
} from "@/lib/api";

export const Route = createFileRoute("/p/$projectId/reconstruct")({
  component: Reconstruct,
});

function Reconstruct() {
  const { projectId } = Route.useParams();
  const navigate = useNavigate();
  const { data: state } = useProjectState(projectId);
  const { data: backends = [] } = useBackends();
  const { data: latest } = useLatestReconstruction(projectId);
  const reconstruct = useReconstruct(projectId);
  const [picked, setPicked] = useState<string>("demo_fixture");

  const captured = state?.capture.complete ?? false;
  const status = latest?.status ?? "none";
  const running = status === "pending" || status === "running";

  const meshUrl = latest?.mesh_path
    ? meshPathToUrl(latest.mesh_path, projectId)
    : null;

  if (!captured) {
    return (
      <>
        <StepNav projectId={projectId} current="reconstruct" />
        <div className="flex-1 p-10 overflow-auto">
          <LockedNotice
            reason={state?.reconstruct.reason ?? "No video uploaded."}
            onNav={() => navigate({ to: "/p/$projectId/capture", params: { projectId } })}
            target="Capture"
          />
        </div>
      </>
    );
  }

  return (
    <>
      <StepNav projectId={projectId} current="reconstruct" />
      <div className="flex-1 p-10 overflow-auto">
        <header className="mb-8">
          <h1 className="text-2xl">Reconstruct</h1>
          <p className="text-sm text-muted-foreground mt-1.5 max-w-xl">
            Turn the uploaded video into a triangle mesh. Pick a backend below.
          </p>
        </header>

        <div className="max-w-xl mb-6">
          <label className="mono text-[11px] uppercase tracking-wider text-muted-foreground">
            Backend
          </label>
          <div className="mt-2 grid grid-cols-2 gap-2">
            {backends.map((b) => {
              const disabled = !b.implemented;
              const selected = picked === b.name;
              return (
                <button
                  key={b.name}
                  disabled={disabled}
                  onClick={() => setPicked(b.name)}
                  title={disabled ? `${b.name} is planned for a future change` : undefined}
                  className={
                    "text-left px-3 py-2 rounded-sm border text-sm transition-colors " +
                    (selected
                      ? "border-primary bg-primary/10"
                      : disabled
                      ? "border-border/40 text-muted-foreground/40 cursor-not-allowed"
                      : "border-border hover:bg-accent")
                  }
                >
                  <div className="flex items-center justify-between">
                    <span className="mono text-xs">{b.name}</span>
                    {b.requires_gpu && (
                      <span className="text-[10px] text-muted-foreground mono">GPU</span>
                    )}
                  </div>
                  {!b.implemented && (
                    <span className="text-[10px] text-muted-foreground">coming soon</span>
                  )}
                </button>
              );
            })}
          </div>
        </div>

        <button
          disabled={running || reconstruct.isPending}
          onClick={() => reconstruct.mutate({ backend: picked, params: {} })}
          className="px-3 py-1.5 rounded-sm bg-primary text-primary-foreground text-sm hover:opacity-90 disabled:opacity-50"
        >
          {running ? "Running…" : latest ? "Re-run reconstruction" : "Reconstruct"}
        </button>

        {reconstruct.error && (
          <p className="mt-3 text-sm text-[var(--status-fail)] mono">
            {String((reconstruct.error as Error).message)}
          </p>
        )}

        {latest && (
          <section className="mt-8 max-w-3xl">
            <header className="flex items-center justify-between mb-3">
              <span className="mono text-[11px] uppercase tracking-wider text-muted-foreground">
                Latest run · {latest.backend}
              </span>
              <StatusDot
                status={
                  latest.status === "ok"
                    ? "ok"
                    : latest.status === "failed"
                    ? "fail"
                    : "idle"
                }
                label={latest.status}
              />
            </header>

            {latest.status === "failed" && (
              <p className="text-sm text-[var(--status-fail)] mono mb-3">
                {latest.error ?? "unknown error"}
              </p>
            )}
            {running && (
              <p className="text-xs text-muted-foreground mono mb-3">
                Job running — this page refreshes every 1.5s.
              </p>
            )}

            {meshUrl && latest.status === "ok" && (
              <>
                <div className="relative">
                  <MeshViewer url={meshUrl} />
                </div>
                <div className="mt-4 flex items-center gap-3">
                  <button
                    onClick={() =>
                      navigate({ to: "/p/$projectId/validate", params: { projectId } })
                    }
                    className="px-3 py-1.5 rounded-sm bg-primary text-primary-foreground text-sm hover:opacity-90"
                  >
                    Continue to Validate →
                  </button>
                  {latest.elapsed_s && (
                    <span className="text-xs text-muted-foreground mono">
                      ran in {latest.elapsed_s.toFixed(1)}s
                    </span>
                  )}
                </div>
              </>
            )}
          </section>
        )}
      </div>
    </>
  );
}

function meshPathToUrl(meshPath: string, projectId: string): string {
  // mesh_path is on the worker's filesystem (e.g. /data/projects/<pid>/reconstruction/mesh.ply).
  // The API exposes /data via StaticFiles. Translate to a URL relative to the data mount.
  const i = meshPath.indexOf("/projects/");
  if (i < 0) return `/data/projects/${projectId}/reconstruction/mesh.ply`;
  return `/data${meshPath.slice(i)}`;
}

function LockedNotice({
  reason,
  onNav,
  target,
}: {
  reason: string;
  onNav: () => void;
  target: string;
}) {
  return (
    <section className="max-w-lg border border-border rounded-sm p-6">
      <div className="flex items-start gap-3">
        <Lock size={16} className="text-muted-foreground mt-0.5" />
        <div>
          <p className="text-sm">This step is locked.</p>
          <p className="text-xs text-muted-foreground mt-1">{reason}</p>
          <button
            onClick={onNav}
            className="mt-4 px-3 py-1.5 rounded-sm bg-primary text-primary-foreground text-sm hover:opacity-90"
          >
            Go to {target}
          </button>
        </div>
      </div>
    </section>
  );
}
