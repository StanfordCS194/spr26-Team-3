import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Lock } from "lucide-react";
import { useState } from "react";

import { ProjectSceneLayout } from "@/components/ProjectSceneLayout";
import { StatusDot } from "@/components/StatusDot";
import { ValidationReport, type Report } from "@/components/ValidationReport";
import {
  useCancelValidation,
  useLatestReconstruction,
  useLatestValidation,
  useProjectState,
  useValidate,
} from "@/lib/api";

export const Route = createFileRoute("/p/$projectId/validate")({
  component: Validate,
});

function Validate() {
  const { projectId } = Route.useParams();
  const navigate = useNavigate();
  const { data: state } = useProjectState(projectId);
  const { data: recon } = useLatestReconstruction(projectId);
  const { data: latest } = useLatestValidation(projectId);
  const validate = useValidate(projectId);
  const cancel = useCancelValidation(projectId);
  const [override, setOverride] = useState(false);

  const reconReady = recon?.status === "ok";
  const running = latest?.status === "pending" || latest?.status === "running";
  // Suppress the report panel while a new validation is in-flight so the
  // user doesn't see the previous run's verdict flash next to "Validating…".
  const report = running ? null : ((latest?.report ?? null) as Report | null);
  const canBuild = report?.overall === "pass" || report?.overall === "warn" || override;

  if (!reconReady) {
    return (
      <ProjectSceneLayout projectId={projectId} current="validate">
        <h1 className="text-2xl">Validate</h1>
        <section className="mt-6 border border-border rounded-sm p-4">
          <div className="flex items-start gap-3">
            <Lock size={16} className="text-muted-foreground mt-0.5" />
            <div>
              <p className="text-sm">Locked.</p>
              <p className="text-xs text-muted-foreground mt-1">
                {state?.validate.reason ?? "No successful reconstruction yet."}
              </p>
              <button
                onClick={() =>
                  navigate({ to: "/p/$projectId/reconstruct", params: { projectId } })
                }
                className="mt-3 px-3 py-1.5 rounded-sm bg-primary text-primary-foreground text-sm hover:opacity-90"
              >
                Go to Reconstruct
              </button>
            </div>
          </div>
        </section>
      </ProjectSceneLayout>
    );
  }

  return (
    <ProjectSceneLayout projectId={projectId} current="validate">
      <header>
        <h1 className="text-2xl">Validate</h1>
        <p className="text-sm text-muted-foreground mt-1.5">
          Six sanity checks on the mesh. Pass / warn lets Build proceed; fail
          blocks unless you override.
        </p>
      </header>

      <div className="mt-6 flex items-center gap-3">
        <button
          disabled={validate.isPending || running}
          onClick={() => validate.mutate()}
          className="px-4 py-2 rounded-sm border-2 border-primary text-primary text-sm font-medium hover:bg-primary/10 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {running
            ? "Validating…"
            : validate.isPending
            ? "Queuing…"
            : latest
            ? "Re-validate"
            : "Run validation"}
        </button>
        {running && (
          <button
            onClick={() => cancel.mutate()}
            disabled={cancel.isPending}
            className="px-3 py-2 rounded-sm border border-[var(--status-fail)] text-[var(--status-fail)] text-sm hover:bg-[var(--status-fail)]/10 disabled:opacity-40 transition-colors"
          >
            {cancel.isPending ? "Cancelling…" : "Cancel"}
          </button>
        )}
      </div>

      {validate.error && (
        <p className="mt-3 text-sm text-[var(--status-fail)] mono">
          {String((validate.error as Error).message)}
        </p>
      )}

      {latest?.status === "failed" && (
        <p className="mt-3 text-sm text-[var(--status-fail)] mono">
          {latest.error ?? "unknown error"}
        </p>
      )}

      {report && (
        <section className="mt-6">
          <header className="flex items-center justify-between mb-3">
            <span className="mono text-[11px] uppercase tracking-wider text-muted-foreground">
              latest
            </span>
            <StatusDot
              status={
                report.overall === "pass"
                  ? "ok"
                  : report.overall === "warn"
                  ? "warn"
                  : "fail"
              }
              label={report.overall}
            />
          </header>
          <ValidationReport report={report} />

          <div className="mt-4 flex flex-col gap-2">
            {report.overall === "fail" && !override && (
              <label className="flex items-center gap-2 text-xs text-muted-foreground">
                <input
                  type="checkbox"
                  checked={override}
                  onChange={(e) => setOverride(e.target.checked)}
                />
                Build anyway (override validation)
              </label>
            )}
            <button
              disabled={!canBuild}
              onClick={() =>
                navigate({ to: "/p/$projectId/build", params: { projectId } })
              }
              className="self-start px-3 py-1.5 rounded-sm border border-border text-sm hover:bg-accent disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Continue to Build →
            </button>
          </div>
        </section>
      )}
    </ProjectSceneLayout>
  );
}
