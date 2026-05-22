import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Lock } from "lucide-react";
import { useEffect, useState } from "react";

import { ProjectSceneLayout } from "@/components/ProjectSceneLayout";
import { StatusDot } from "@/components/StatusDot";
import { RadioCardGroup } from "@/components/ui/RadioCardGroup";
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

  // If the user's pick stops being available (e.g. vggt flips false after a
  // backend restart), snap to the first actually-implemented backend.
  useEffect(() => {
    if (backends.length === 0) return;
    const valid = backends.filter((b) => b.implemented).map((b) => b.name);
    if (valid.length > 0 && !valid.includes(picked)) {
      setPicked(valid[0]);
    }
  }, [backends, picked]);

  if (!captured) {
    return (
      <ProjectSceneLayout projectId={projectId} current="reconstruct">
        <h1 className="text-2xl">Reconstruct</h1>
        <section className="mt-6 border border-border rounded-sm p-4">
          <div className="flex items-start gap-3">
            <Lock size={16} className="text-muted-foreground mt-0.5" />
            <div>
              <p className="text-sm">Locked.</p>
              <p className="text-xs text-muted-foreground mt-1">
                {state?.reconstruct.reason ?? "No video uploaded."}
              </p>
              <button
                onClick={() =>
                  navigate({ to: "/p/$projectId/capture", params: { projectId } })
                }
                className="mt-3 px-3 py-1.5 rounded-sm bg-primary text-primary-foreground text-sm hover:opacity-90"
              >
                Go to Capture
              </button>
            </div>
          </div>
        </section>
      </ProjectSceneLayout>
    );
  }

  return (
    <ProjectSceneLayout projectId={projectId} current="reconstruct">
      <header>
        <h1 className="text-2xl">Reconstruct</h1>
        <p className="text-sm text-muted-foreground mt-1.5">
          Pick a reconstruction method and run it once. Only one runs at a time.
        </p>
      </header>

      <div className="mt-6">
        <div className="mono text-[11px] uppercase tracking-wider text-muted-foreground mb-2">
          Method
        </div>
        <RadioCardGroup
          name="reconstruction-method"
          value={picked}
          onValueChange={setPicked}
          disabled={running}
          options={backends.map((b) => ({
            value: b.name,
            label: b.name,
            tag: b.requires_gpu ? "GPU" : undefined,
            sublabel: !b.implemented ? "not available" : undefined,
            disabled: !b.implemented,
          }))}
        />
      </div>

      <button
        disabled={running || reconstruct.isPending}
        onClick={() => reconstruct.mutate({ backend: picked, params: {} })}
        className="mt-6 px-4 py-2 rounded-sm border-2 border-primary text-primary text-sm font-medium hover:bg-primary/10 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
      >
        {running
          ? "Reconstruction running…"
          : reconstruct.isPending
          ? "Queuing…"
          : latest
          ? "Re-run reconstruction"
          : "Run reconstruction"}
      </button>

      {reconstruct.error && (
        <p className="mt-3 text-sm text-[var(--status-fail)] mono">
          {String((reconstruct.error as Error).message)}
        </p>
      )}

      {latest && (
        <section className="mt-8">
          <header className="flex items-center justify-between mb-3">
            <span className="mono text-[11px] uppercase tracking-wider text-muted-foreground">
              latest · {latest.backend}
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
            <p className="text-xs text-[var(--status-fail)] mono">
              {latest.error ?? "unknown error"}
            </p>
          )}
          {running && (
            <p className="text-xs text-muted-foreground mono">
              Running on Inngest · live polling
            </p>
          )}
          {latest.status === "ok" && (
            <div className="flex flex-col gap-2">
              {latest.elapsed_s && (
                <span className="text-xs text-muted-foreground mono">
                  ran in {latest.elapsed_s.toFixed(1)}s
                </span>
              )}
              <button
                onClick={() =>
                  navigate({ to: "/p/$projectId/validate", params: { projectId } })
                }
                className="px-3 py-1.5 rounded-sm border border-border text-sm hover:bg-accent self-start"
              >
                Continue to Validate →
              </button>
            </div>
          )}
        </section>
      )}
    </ProjectSceneLayout>
  );
}
