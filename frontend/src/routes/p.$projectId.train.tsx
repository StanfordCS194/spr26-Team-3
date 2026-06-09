import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Lock } from "lucide-react";
import { useState } from "react";

import { MetricsChart } from "@/components/MetricsChart";
import { StatusDot } from "@/components/StatusDot";
import { StepNav } from "@/components/StepNav";
import {
  useCancelTraining,
  usePolicies,
  usePolicyLive,
  useProjectState,
  useTrain,
} from "@/lib/api";

export const Route = createFileRoute("/p/$projectId/train")({
  component: Train,
});

function Train() {
  const { projectId } = Route.useParams();
  const navigate = useNavigate();
  const { data: state } = useProjectState(projectId);
  const { data: policies = [] } = usePolicies(projectId);
  const train = useTrain(projectId);
  const cancelTrain = useCancelTraining(projectId);
  const [activePolicyId, setActivePolicyId] = useState<string | null>(null);
  const liveId = activePolicyId ?? policies[0]?.id ?? null;
  const { data: live } = usePolicyLive(projectId, liveId);

  const canTrain = state?.build.complete ?? false;
  const [steps, setSteps] = useState(50_000);
  const [seed, setSeed] = useState(0);
  const [nEnvs, setNEnvs] = useState(4);

  if (!canTrain) {
    return (
      <>
        <StepNav projectId={projectId} current="train" />
        <div className="flex-1 p-10 overflow-auto">
          <section className="max-w-lg border border-border rounded-sm p-6">
            <div className="flex items-start gap-3">
              <Lock size={16} className="text-muted-foreground mt-0.5" />
              <div>
                <p className="text-sm">Train is locked.</p>
                <p className="text-xs text-muted-foreground mt-1">
                  {state?.train.reason ?? "Complete Build first."}
                </p>
                <button
                  onClick={() => navigate({ to: "/p/$projectId/build", params: { projectId } })}
                  className="mt-4 px-3 py-1.5 rounded-sm bg-primary text-primary-foreground text-sm hover:opacity-90"
                >
                  Go to Build
                </button>
              </div>
            </div>
          </section>
        </div>
      </>
    );
  }

  const progress = live?.metrics?.progress ?? 0;
  const running =
    live && !live.metrics?.done && !live.metrics?.error && !live.metrics?.cancelled;
  const trace = live?.metrics?.trace ?? [];

  return (
    <>
      <StepNav projectId={projectId} current="train" />
      <div className="flex-1 p-10 overflow-auto">
        <header className="mb-8">
          <h1 className="text-2xl">Train</h1>
          <p className="text-sm text-muted-foreground mt-1.5 max-w-xl">
            Train PPO on the latest build's MJCF. Metrics stream live; you can
            navigate away and come back without losing progress.
          </p>
        </header>

        <section className="max-w-xl mb-6 grid grid-cols-3 gap-4">
          <label className="text-xs">
            <span className="block mono uppercase tracking-wider text-muted-foreground mb-1">
              total steps
            </span>
            <input
              type="number"
              value={steps}
              onChange={(e) => setSteps(parseInt(e.target.value, 10) || 0)}
              className="w-full bg-background border border-border rounded-sm px-2 py-1 mono text-sm"
            />
          </label>
          <label className="text-xs">
            <span className="block mono uppercase tracking-wider text-muted-foreground mb-1">
              n envs
            </span>
            <input
              type="number"
              value={nEnvs}
              onChange={(e) => setNEnvs(parseInt(e.target.value, 10) || 1)}
              className="w-full bg-background border border-border rounded-sm px-2 py-1 mono text-sm"
            />
          </label>
          <label className="text-xs">
            <span className="block mono uppercase tracking-wider text-muted-foreground mb-1">
              seed
            </span>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(parseInt(e.target.value, 10) || 0)}
              className="w-full bg-background border border-border rounded-sm px-2 py-1 mono text-sm"
            />
          </label>
        </section>

        <div className="flex items-center gap-3">
          <button
            disabled={train.isPending || !!running}
            onClick={async () => {
              const p = await train.mutateAsync({ total_steps: steps, n_envs: nEnvs, seed });
              setActivePolicyId(p.id);
            }}
            className="px-4 py-2 rounded-sm border-2 border-primary text-primary text-sm font-medium hover:bg-primary/10 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {running ? "Training…" : "Train PPO"}
          </button>
          {running && liveId && (
            <button
              onClick={() => cancelTrain.mutate(liveId)}
              disabled={cancelTrain.isPending}
              className="px-3 py-2 rounded-sm border border-[var(--status-fail)] text-[var(--status-fail)] text-sm hover:bg-[var(--status-fail)]/10 disabled:opacity-40 transition-colors"
            >
              {cancelTrain.isPending ? "Cancelling…" : "Cancel"}
            </button>
          )}
        </div>

        {train.error && (
          <p className="mt-3 text-sm text-[var(--status-fail)] mono">
            {String((train.error as Error).message)}
          </p>
        )}

        {live && (
          <section className="mt-8 max-w-3xl space-y-4">
            <header className="flex items-center justify-between">
              <span className="mono text-[11px] uppercase tracking-wider text-muted-foreground">
                policy · {live.id}
              </span>
              <div className="flex items-center gap-4">
                <StatusDot
                  status={
                    live.metrics?.error
                      ? "fail"
                      : live.metrics?.cancelled
                      ? "idle"
                      : live.metrics?.done
                      ? "ok"
                      : "warn"
                  }
                  label={
                    live.metrics?.error
                      ? "error"
                      : live.metrics?.cancelled
                      ? `cancelled · ${(progress * 100).toFixed(0)}%`
                      : live.metrics?.done
                      ? `done · ${live.metrics?.elapsed_s?.toFixed(1)}s`
                      : `${(progress * 100).toFixed(0)}% · ${live.metrics?.fps ?? 0} fps`
                  }
                />
              </div>
            </header>

            <div className="h-2 w-full bg-muted rounded-sm overflow-hidden">
              <div
                className="h-full bg-primary transition-all"
                style={{ width: `${(progress * 100).toFixed(1)}%` }}
              />
            </div>

            {trace.length > 1 && <MetricsChart trace={trace} />}

            {live.metrics?.error && (
              <p className="text-sm text-[var(--status-fail)] mono">
                {live.metrics.error}
              </p>
            )}

            {live.metrics?.done && !live.metrics?.error && (
              <button
                onClick={() => navigate({ to: "/p/$projectId/replay", params: { projectId } })}
                className="px-3 py-1.5 rounded-sm bg-primary text-primary-foreground text-sm hover:opacity-90"
              >
                Continue to Replay →
              </button>
            )}
          </section>
        )}

        {policies.length > 1 && (
          <section className="mt-10 max-w-3xl">
            <h2 className="text-sm mono uppercase tracking-wider text-muted-foreground mb-2">
              Prior policies
            </h2>
            <ul className="divide-y divide-border border border-border rounded-sm">
              {policies.map((p) => (
                <li key={p.id} className="px-4 py-2 flex items-center justify-between text-sm">
                  <span className="mono text-xs">{p.id}</span>
                  <span className="text-xs text-muted-foreground">
                    {p.total_steps} steps · avg r={p.metrics?.avg_reward ?? "—"}
                  </span>
                  <button
                    onClick={() => setActivePolicyId(p.id)}
                    className="text-xs px-2 py-0.5 rounded-sm border border-border hover:bg-accent"
                  >
                    inspect
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </>
  );
}
