import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Lock } from "lucide-react";
import { useState } from "react";

import { StatusDot } from "@/components/StatusDot";
import { StepNav } from "@/components/StepNav";
import { TrajectoryViewer, type TrajectoryRun } from "@/components/TrajectoryViewer";
import {
  useProjectRuns,
  useProjectState,
  useReplayWithTrajectories,
  type TrajectoryReplayResponse,
} from "@/lib/api";

export const Route = createFileRoute("/p/$projectId/replay")({
  component: ReplayScreen,
});

type PolicyName = "random" | "greedy" | "ppo";

const FAILURE_LABEL: Record<string, string> = {
  success: "success",
  timeout: "timeout",
  stuck: "stuck",
  collided: "collided",
  "near-miss": "near-miss",
};

function ReplayScreen() {
  const { projectId } = Route.useParams();
  const navigate = useNavigate();
  const { data: state } = useProjectState(projectId);
  const replay = useReplayWithTrajectories(projectId);
  const [runs, setRuns] = useState<Record<PolicyName, TrajectoryReplayResponse | null>>({
    random: null,
    greedy: null,
    ppo: null,
  });
  const [focus, setFocus] = useState<PolicyName>("greedy");
  const [episodeIdx, setEpisodeIdx] = useState<number | undefined>(undefined);

  const canReplay = state?.replay.complete ?? false;
  const canPPO = state?.train.complete ?? false;
  const { data: pastRuns = [] } = useProjectRuns(projectId);

  const run = async (policy: PolicyName) => {
    setFocus(policy);
    const r = await replay.mutateAsync({ policy, episodes: 5, max_steps: 300, seed: 0 });
    setRuns((prev) => ({ ...prev, [policy]: r }));
  };

  if (!canReplay) {
    return (
      <>
        <StepNav projectId={projectId} current="replay" />
        <div className="flex-1 p-10 overflow-auto">
          <section className="max-w-lg border border-border rounded-sm p-6">
            <div className="flex items-start gap-3">
              <Lock size={16} className="text-muted-foreground mt-0.5" />
              <div>
                <p className="text-sm">Replay is locked.</p>
                <p className="text-xs text-muted-foreground mt-1">
                  {state?.replay.reason ?? "No build yet."}
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

  const focused = runs[focus];

  // failure-class buckets for the focused run
  const buckets = focused
    ? focused.episodes.reduce<Record<string, number[]>>((acc, ep, i) => {
        const k = ep.failure_class;
        (acc[k] ||= []).push(i);
        return acc;
      }, {})
    : {};

  const trajRun: TrajectoryRun | null = focused
    ? {
        policy: focused.policy,
        bounds: focused.bounds,
        spawn_region: focused.spawn_region,
        episodes: focused.episodes,
      }
    : null;

  return (
    <>
      <StepNav projectId={projectId} current="replay" />
      <div className="flex-1 p-10 overflow-auto">
        <header className="mb-8">
          <h1 className="text-2xl">Replay</h1>
          <p className="text-sm text-muted-foreground mt-1.5 max-w-2xl">
            Roll the latest build under random / greedy / PPO. Trajectories
            overlay on the floor view; failures are bucketed by reason.
          </p>
        </header>

        <div className="flex gap-2 mb-6">
          {(["random", "greedy", "ppo"] as PolicyName[]).map((p) => (
            <button
              key={p}
              disabled={replay.isPending || (p === "ppo" && !canPPO)}
              onClick={() => run(p)}
              title={p === "ppo" && !canPPO ? "Train a PPO policy first" : undefined}
              className={
                "px-3 py-1.5 rounded-sm text-sm transition-colors " +
                (focus === p
                  ? "bg-primary text-primary-foreground"
                  : "border border-border hover:bg-accent") +
                (p === "ppo" && !canPPO ? " opacity-40 cursor-not-allowed" : "")
              }
            >
              Run {p} (5 eps)
            </button>
          ))}
        </div>

        {replay.error && (
          <p className="mt-2 mb-4 text-sm text-[var(--status-fail)] mono">
            {String((replay.error as Error).message)}
          </p>
        )}

        {focused && trajRun && (
          <section className="space-y-6">
            <header className="flex items-center justify-between">
              <span className="mono text-[11px] uppercase tracking-wider text-muted-foreground">
                {focus} · seed 0
              </span>
              <div className="flex items-center gap-6">
                <StatusDot
                  status={focused.successes / focused.n_episodes >= 0.5 ? "ok" : "warn"}
                  label={`${focused.successes}/${focused.n_episodes}`}
                />
                <span className="mono text-xs text-muted-foreground">
                  avg r = {focused.avg_reward.toFixed(2)}
                </span>
              </div>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <div className="lg:col-span-2">
                <TrajectoryViewer run={trajRun} episodeIndex={episodeIdx} />
              </div>

              <div className="space-y-3">
                <h2 className="mono text-[11px] uppercase tracking-wider text-muted-foreground">
                  episodes
                </h2>
                <div className="border border-border rounded-sm divide-y divide-border max-h-[360px] overflow-auto">
                  {focused.episodes.map((ep, i) => {
                    const active = episodeIdx === i;
                    return (
                      <button
                        key={i}
                        onClick={() => setEpisodeIdx(active ? undefined : i)}
                        className={
                          "block w-full text-left px-3 py-2 text-xs transition-colors " +
                          (active ? "bg-accent" : "hover:bg-accent/30")
                        }
                      >
                        <div className="flex items-center justify-between mono">
                          <span>ep {i + 1}</span>
                          <StatusDot status={ep.success ? "ok" : "fail"} />
                        </div>
                        <div className="flex items-center justify-between mono text-muted-foreground mt-0.5">
                          <span>{ep.steps}s</span>
                          <span>r={ep.reward.toFixed(1)}</span>
                          <span>d={ep.distance.toFixed(2)}</span>
                          <span>{FAILURE_LABEL[ep.failure_class] ?? ep.failure_class}</span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            <section>
              <h2 className="mono text-[11px] uppercase tracking-wider text-muted-foreground mb-2">
                failure buckets
              </h2>
              <div className="flex gap-2 flex-wrap">
                {Object.entries(buckets).map(([cls, idxs]) => (
                  <button
                    key={cls}
                    onClick={() => setEpisodeIdx(idxs[0])}
                    className="px-3 py-1.5 border border-border rounded-sm text-xs mono hover:bg-accent"
                  >
                    {cls} · {idxs.length}
                  </button>
                ))}
                {episodeIdx !== undefined && (
                  <button
                    onClick={() => setEpisodeIdx(undefined)}
                    className="px-3 py-1.5 border border-border rounded-sm text-xs mono text-muted-foreground hover:bg-accent"
                  >
                    show all
                  </button>
                )}
              </div>
            </section>
          </section>
        )}

        {pastRuns.length > 0 && (
          <section className="mt-12">
            <h2 className="mono text-[11px] uppercase tracking-wider text-muted-foreground mb-3">
              Run history
            </h2>
            <div className="border border-border rounded-sm divide-y divide-border max-w-3xl">
              {pastRuns.map((r) => {
                const pct = r.episodes > 0 ? (100 * r.successes) / r.episodes : 0;
                return (
                  <div key={r.id} className="px-4 py-2.5 flex items-center justify-between text-xs">
                    <div className="mono flex items-center gap-4">
                      <span className="text-muted-foreground">
                        {new Date(r.created_at).toLocaleTimeString()}
                      </span>
                      <span>{r.baseline ?? "ppo"}</span>
                    </div>
                    <div className="mono flex items-center gap-6">
                      <span>
                        {r.successes}/{r.episodes}{" "}
                        <span className="text-muted-foreground">({pct.toFixed(0)}%)</span>
                      </span>
                      <span className="text-muted-foreground">avg r {r.avg_reward.toFixed(1)}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        )}
      </div>
    </>
  );
}
