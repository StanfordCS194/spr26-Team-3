import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Lock } from "lucide-react";
import { useState } from "react";

import { CompareViewer } from "@/components/CompareViewer";
import { ProjectSceneLayout } from "@/components/ProjectSceneLayout";
import { StatusDot } from "@/components/StatusDot";
import { TrajectoryViewer, type TrajectoryRun } from "@/components/TrajectoryViewer";
import {
  useComparePolicies,
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
  const compare = useComparePolicies(projectId);
  const [runId, setRunId] = useState<string | null>(null);
  const [policy, setPolicy] = useState<PolicyName>("greedy");
  const [episodeIdx, setEpisodeIdx] = useState<number | undefined>(undefined);
  const [viewMode, setViewMode] = useState<"animate" | "paths">("animate");

  const canReplay = state?.replay.complete ?? false;
  const canPPO = state?.train.complete ?? false;
  const { data: pastRuns = [] } = useProjectRuns(projectId);

  // Live status of the currently-active run; polls on pending/running.
  const { data: live } = useRun(projectId, runId);
  const ready = live?.status === "ok";
  const { data: traj } = useRunTrajectories(projectId, runId, ready);

  // One run = one robot, one random start → random goal. A fresh random seed
  // every press, so each play spawns a new start/goal. `count > 1` is the
  // benchmark mode (several episodes at once).
  const start = async (p: PolicyName, count = 1) => {
    setPolicy(p);
    setEpisodeIdx(undefined);
    const seed = Math.floor(Math.random() * 1_000_000);
    const r = await startReplay.mutateAsync({ policy: p, episodes: count, max_steps: 300, seed });
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
        <h1 className="text-2xl">Navigate</h1>
        <p className="text-sm text-muted-foreground mt-1.5">
          One robot, a random start → a random goal. Press again for a new
          start/goal. Watch it route around obstacles.
        </p>
      </header>

      <div className="mt-6 flex flex-col gap-2">
        {(["ppo", "greedy", "random"] as PolicyName[]).map((p) => (
          <button
            key={p}
            disabled={startReplay.isPending || (p === "ppo" && !canPPO) || live?.status === "running"}
            onClick={() => start(p, 1)}
            title={p === "ppo" && !canPPO ? "Train a PPO policy first" : undefined}
            className={
              "text-left px-4 py-2 rounded-sm border-2 text-sm transition-colors " +
              (policy === p && runId
                ? "border-primary text-primary"
                : "border-border hover:bg-accent hover:border-foreground/40") +
              (p === "ppo" && !canPPO ? " opacity-40 cursor-not-allowed" : "")
            }
          >
            {p === "ppo" ? "▶ Navigate (trained policy)" : `▶ Navigate (${p} baseline)`}
            <span className="text-muted-foreground"> · random start → goal</span>
          </button>
        ))}
        <button
          disabled={startReplay.isPending || live?.status === "running"}
          onClick={() => start(policy, 5)}
          className="text-left px-4 py-1.5 rounded-sm border border-dashed border-border text-xs text-muted-foreground hover:bg-accent mt-1"
        >
          Benchmark · 5 random episodes at once
        </button>
      </div>

      <section className="mt-6 border border-border rounded-sm p-3">
        <div className="flex items-center justify-between gap-2">
          <div>
            <h2 className="text-sm">Greedy vs PPO — same start &amp; goal</h2>
            <p className="text-[11px] text-muted-foreground mt-0.5">
              Both policies run the identical config. Where an obstacle is in
              the way, greedy stalls and PPO routes around it.
            </p>
          </div>
          <div className="flex gap-1.5 shrink-0">
            <button
              disabled={!canPPO || compare.isPending}
              onClick={() => compare.mutate({})}
              title={!canPPO ? "Train a PPO policy first" : undefined}
              className={
                "px-3 py-1.5 rounded-sm text-xs whitespace-nowrap " +
                (canPPO
                  ? "bg-primary text-primary-foreground hover:opacity-90"
                  : "border border-border opacity-40 cursor-not-allowed")
              }
            >
              {compare.isPending ? "Racing…" : compare.data ? "↻ New config" : "⚔ Compare"}
            </button>
          </div>
        </div>
        {compare.error && (
          <p className="mt-2 text-[11px] text-[var(--status-fail)] mono">
            {String((compare.error as Error).message)}
          </p>
        )}
        {compare.data && <div className="mt-3"><CompareViewer data={compare.data} height={280} /></div>}
      </section>

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
                  ? `${live.successes}/${live.episodes ?? 5}`
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
              <div className="flex gap-1">
                {(["animate", "paths"] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setViewMode(m)}
                    className={
                      "px-3 py-1 rounded-sm mono text-[10px] uppercase tracking-wider " +
                      (viewMode === m
                        ? "bg-foreground text-background"
                        : "border border-border hover:bg-accent")
                    }
                  >
                    {m === "animate" ? "▶ Animate" : "Paths"}
                  </button>
                ))}
              </div>
              <TrajectoryViewer
                run={trajRun}
                episodeIndex={episodeIdx}
                height={260}
                animate={viewMode === "animate"}
              />
              {(() => {
                const eps = trajRun.episodes;
                const single = eps.length === 1;
                let col = 0;
                let steps = 0;
                let succ = 0;
                let avoided = 0;
                for (const e of eps) {
                  const t = e.trajectory ?? [];
                  steps += t.length;
                  col += e.collisions ?? t.filter((p) => p.collision).length;
                  if (e.success) succ += 1;
                  avoided += e.avoided ?? 0;
                }
                const collRate = steps ? (100 * col) / steps : 0;
                const avgSteps = eps.length ? Math.round(steps / eps.length) : 0;

                if (single) {
                  const e = eps[0];
                  const n = e.trajectory?.length ?? avgSteps;
                  const shown = Math.min(avoided, 6);
                  return (
                    <div className="space-y-1.5">
                      <div className="mono text-[12px]">
                        {e.success ? (
                          <span className="text-[var(--status-ok)]">✓ reached the goal in {n} steps</span>
                        ) : (
                          <span className="text-[var(--status-fail)]">✗ did not reach the goal ({n} steps)</span>
                        )}
                      </div>
                      <div className="flex flex-col gap-0.5 mono text-[11px] text-muted-foreground border-l-2 border-border pl-2">
                        {avoided > 0 ? (
                          <>
                            {Array.from({ length: shown }).map((_, i) => (
                              <span key={i}>🛡️ avoided an obstacle</span>
                            ))}
                            {avoided > shown && <span>… and {avoided - shown} more</span>}
                          </>
                        ) : (
                          <span>· clear path — no obstacles to route around</span>
                        )}
                        {col > 0 && (
                          <span className="text-[var(--status-fail)]">
                            ⚠ bumped an obstacle on {col} step{col > 1 ? "s" : ""}
                          </span>
                        )}
                      </div>
                    </div>
                  );
                }
                return (
                  <div className="flex gap-4 mono text-[11px] text-muted-foreground">
                    <span>success <span className="text-foreground">{succ}/{eps.length}</span></span>
                    <span>collision rate <span className="text-foreground">{collRate.toFixed(1)}%</span></span>
                    <span>avoided <span className="text-foreground">{avoided}</span></span>
                    <span>avg steps <span className="text-foreground">{avgSteps}</span></span>
                  </div>
                );
              })()}
              {trajRun.episodes.length > 1 && (
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
              )}
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
