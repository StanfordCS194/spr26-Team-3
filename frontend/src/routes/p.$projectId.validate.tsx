import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Lock } from "lucide-react";
import { useState } from "react";

import { StatusDot } from "@/components/StatusDot";
import { StepNav } from "@/components/StepNav";
import { ValidationReport, type Report } from "@/components/ValidationReport";
import {
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
  const [override, setOverride] = useState(false);

  const reconReady = recon?.status === "ok";
  const report = (latest?.report ?? null) as Report | null;
  const canBuild = report?.overall === "pass" || report?.overall === "warn" || override;

  if (!reconReady) {
    return (
      <>
        <StepNav projectId={projectId} current="validate" />
        <div className="flex-1 p-10 overflow-auto">
          <section className="max-w-lg border border-border rounded-sm p-6">
            <div className="flex items-start gap-3">
              <Lock size={16} className="text-muted-foreground mt-0.5" />
              <div>
                <p className="text-sm">Validate is locked.</p>
                <p className="text-xs text-muted-foreground mt-1">
                  {state?.validate.reason ?? "No successful reconstruction yet."}
                </p>
                <button
                  onClick={() =>
                    navigate({ to: "/p/$projectId/reconstruct", params: { projectId } })
                  }
                  className="mt-4 px-3 py-1.5 rounded-sm bg-primary text-primary-foreground text-sm hover:opacity-90"
                >
                  Go to Reconstruct
                </button>
              </div>
            </div>
          </section>
        </div>
      </>
    );
  }

  return (
    <>
      <StepNav projectId={projectId} current="validate" />
      <div className="flex-1 p-10 overflow-auto">
        <header className="mb-8">
          <h1 className="text-2xl">Validate</h1>
          <p className="text-sm text-muted-foreground mt-1.5 max-w-xl">
            Run six sanity checks on the latest mesh before turning it into an
            RL environment. Pass / warn lets Build proceed; fail blocks unless
            you override.
          </p>
        </header>

        <button
          disabled={validate.isPending}
          onClick={() => validate.mutate()}
          className="px-3 py-1.5 rounded-sm bg-primary text-primary-foreground text-sm hover:opacity-90 disabled:opacity-50"
        >
          {validate.isPending ? "Running…" : latest ? "Re-validate" : "Run validation"}
        </button>

        {validate.error && (
          <p className="mt-3 text-sm text-[var(--status-fail)] mono">
            {String((validate.error as Error).message)}
          </p>
        )}

        {report && (
          <section className="mt-8 max-w-3xl">
            <header className="flex items-center justify-between mb-3">
              <span className="mono text-[11px] uppercase tracking-wider text-muted-foreground">
                Latest report · {latest && new Date(latest.created_at).toLocaleString()}
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

            <div className="mt-6 flex items-center gap-4">
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
                className="px-3 py-1.5 rounded-sm bg-primary text-primary-foreground text-sm hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Continue to Build →
              </button>
            </div>
          </section>
        )}
      </div>
    </>
  );
}
