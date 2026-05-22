import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Lock } from "lucide-react";
import { useState } from "react";

import { ProjectSceneLayout } from "@/components/ProjectSceneLayout";
import { StatusDot } from "@/components/StatusDot";
import { TrajectoryViewer, type TrajectoryRun } from "@/components/TrajectoryViewer";
import {
  useProjectRuns,
  useProjectState,
  useRun,
  useRunTrajectories,
  useStartReplay,
} from "@/lib/api";

export const Route = createFileRoute("/p/$projectId/replay")({
  component: ReplayScreen,
});

type PolicyName = "random" | "greedy" | "ppo";

function ReplayScreen() {
  const { projectId } = Route.useParams();
  const navigate = useNavigate();
  const { data: state } = useProjectState(projectId);
  const startReplay = useStartReplay(projectId);
  const [runId, setRunId] = useState<string | null>(null);
  const [policy, setPolicy] = useState<PolicyName>("greedy");
  const [episodeIdx, setEpisodeIdx] = useState<number | undefined>(undefined);

  const canReplay = state?.replay.complete ?? false;
  const canPPO = state?.train.complete ?? false;
  const { data: pastRuns = [] } = useProjectRuns(projectId);

  // Live status of the currently-active run; polls on pending/running.
  const { data: live } = useRun(projectId, runId);
  const ready = live?.status === "ok";
  const { data: traj } = useRunTrajectories(projectId, runId, ready);

  const start = async (p: PolicyName) => {
    setPolicy(p);
    setEpisodeIdx(undefined);
    const r = await startReplay.mutateAsync({ policy: p, episodes: 5, max_steps: 300, seed: 0 });
    setRunId(r.id);
  };

  if (!canReplay) {
    return (
      <ProjectSceneLayout projectId={projectId} current="replay">
        <h1 className="text-2xl">Replay</h1>
        <section className="mt-6 border border-border rounded-sm p-4">
          <div className="flex items-start gap-3">
            <Lock size={16} className="text-muted-foreground mt-0.5" />
            <div>
              <p className="text-sm">Locked.</p>
              <p className="text-xs text-muted-foreground mt-1">
                {state?.replay.reason ?? "No build yet."}
              </p>
              <button
                onClick={() => navigate({ to: "/p/$projectId/build", params: { projectId } })}
                className="mt-3 px-3 py-1.5 rounded-sm bg-primary text-primary-foreground text-sm hover:opacity-90"
              >
                Go to Build
              </button>
            </div>
          </div>
        </section>
      </ProjectSceneLayout>
    );
  }

  const trajRun: TrajectoryRun | null = traj
    ? {
        policy,
        bounds: traj.bounds,
        spawn_region: traj.spawn_region,
        episodes: traj.episodes,
      }
    : null;

  return (
    <ProjectSceneLayout projectId={projectId} current="replay">
      <header>
        <h1 className="text-2xl">Replay</h1>
        <p className="text-sm text-muted-foreground mt-1.5">
          Random / greedy baselines + the trained policy.
        </p>
      </header>

      <div className="mt-6 flex flex-col gap-2">
        {(["random", "greedy", "ppo"] as PolicyName[]).map((p) => (
          <button
            key={p}
            disabled={startReplay.isPending || (p === "ppo" && !canPPO) || live?.status === "running"}
            onClick={() => start(p)}
            title={p === "ppo" && !canPPO ? "Train a PPO policy first" : undefined}
            className={
              "text-left px-4 py-2 rounded-sm border-2 text-sm transition-colors " +
              (policy === p && runId
                ? "border-primary text-primary"
                : "border-border hover:bg-accent hover:border-foreground/40") +
              (p === "ppo" && !canPPO ? " opacity-40 cursor-not-allowed" : "")
            }
          >
            Run {p} (5 episodes)
          </button>
        ))}
      </div>

      {startReplay.error && (
        <p className="mt-3 text-sm text-[var(--status-fail)] mono">
          {String((startReplay.error as Error).message)}
        </p>
      )}

      {live && (
        <section className="mt-6 space-y-3">
          <header className="flex items-center justify-between">
            <span className="mono text-[11px] uppercase tracking-wider text-muted-foreground">
              {policy} · {live.id}
            </span>
            <StatusDot
              status={
                live.status === "ok"
                  ? "ok"
                  : live.status === "failed"
                  ? "fail"
                  : "warn"
              }
              label={
                live.status === "ok"
                  ? `${live.successes}/${live.n_episodes ?? live.episodes ?? 5}`
                  : live.status
              }
            />
          </header>

          {live.status === "failed" && (
            <p className="text-xs text-[var(--status-fail)] mono">
              {live.error ?? "unknown error"}
            </p>
          )}

          {trajRun && (
            <>
              <TrajectoryViewer run={trajRun} episodeIndex={episodeIdx} height={220} />
              <div className="flex gap-1.5 flex-wrap">
                {trajRun.episodes.map((e, i) => (
                  <button
                    key={i}
                    onClick={() => setEpisodeIdx(episodeIdx === i ? undefined : i)}
                    className={
                      "px-2 py-1 rounded-sm mono text-[10px] " +
                      (episodeIdx === i
                        ? "bg-foreground text-background"
                        : "border border-border hover:bg-accent")
                    }
                  >
                    ep{i + 1} {e.success ? "✓" : "✗"} {e.failure_class}
                  </button>
                ))}
              </div>
            </>
          )}
        </section>
      )}

      {pastRuns.length > 0 && (
        <section className="mt-8">
          <h2 className="mono text-[11px] uppercase tracking-wider text-muted-foreground mb-2">
            Run history
          </h2>
          <div className="border border-border rounded-sm divide-y divide-border">
            {pastRuns.slice(0, 8).map((r) => {
              const pct = (r.episodes ?? 0) > 0 ? (100 * (r.successes ?? 0)) / (r.episodes ?? 1) : 0;
              return (
                <div key={r.id} className="px-3 py-1.5 flex items-center justify-between text-[11px] mono">
                  <span>{r.baseline ?? "ppo"}</span>
                  <span>
                    {(r.successes ?? 0)}/{(r.episodes ?? 0)}{" "}
                    <span className="text-muted-foreground">({pct.toFixed(0)}%)</span>
                  </span>
                </div>
              );
            })}
          </div>
        </section>
      )}
    </ProjectSceneLayout>
  );
}
