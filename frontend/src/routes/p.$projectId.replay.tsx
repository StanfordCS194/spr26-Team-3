import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Lock } from "lucide-react";
import { useState } from "react";

import { StatusDot } from "@/components/StatusDot";
import { StepNav } from "@/components/StepNav";
import { useProjectState, useReplay, type ReplayResponse } from "@/lib/api";

export const Route = createFileRoute("/p/$projectId/replay")({
  component: ReplayScreen,
});

function ReplayScreen() {
  const { projectId } = Route.useParams();
  const navigate = useNavigate();
  const { data: state } = useProjectState(projectId);
  const replay = useReplay(projectId);
  const [results, setResults] = useState<ReplayResponse[]>([]);

  const canReplay = state?.replay.complete ?? false;

  const run = async (policy: "random" | "greedy") => {
    const r = await replay.mutateAsync({ policy, episodes: 5, max_steps: 300, seed: 0 });
    setResults((prev) => [r, ...prev].slice(0, 6));
  };

  return (
    <>
      <StepNav projectId={projectId} current="replay" />
      <div className="flex-1 p-10 overflow-auto">
        <header className="mb-8">
          <h1 className="text-2xl">Replay</h1>
          <p className="text-sm text-muted-foreground mt-1.5">
            Run random / greedy baselines against the latest build. PPO baseline lands in PR-C.
          </p>
        </header>

        {!canReplay ? (
          <section className="max-w-lg border border-border rounded-sm p-6">
            <div className="flex items-start gap-3">
              <Lock size={16} className="text-muted-foreground mt-0.5" />
              <div>
                <p className="text-sm">Replay is locked.</p>
                <p className="text-xs text-muted-foreground mt-1">
                  {state?.replay.reason ?? "No build yet for this project."}
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
        ) : (
          <>
            <div className="flex gap-2">
              <button
                disabled={replay.isPending}
                onClick={() => run("random")}
                className="px-3 py-1.5 rounded-sm border border-border text-sm hover:bg-accent disabled:opacity-50"
              >
                Run random (5 eps)
              </button>
              <button
                disabled={replay.isPending}
                onClick={() => run("greedy")}
                className="px-3 py-1.5 rounded-sm bg-primary text-primary-foreground text-sm hover:opacity-90 disabled:opacity-50"
              >
                Run greedy (5 eps)
              </button>
            </div>

            {replay.error && (
              <p className="mt-4 text-sm text-[var(--status-fail)] mono">
                {String((replay.error as Error).message)}
              </p>
            )}

            <div className="mt-8 space-y-4">
              {results.map((r, i) => (
                <section key={i} className="border border-border rounded-sm">
                  <header className="px-4 py-2.5 border-b border-border flex items-center justify-between">
                    <span className="mono text-[11px] uppercase tracking-wider text-muted-foreground">
                      {r.policy}
                    </span>
                    <div className="flex items-center gap-4">
                      <StatusDot
                        status={r.successes / r.n_episodes >= 0.5 ? "ok" : "warn"}
                        label={`${r.successes}/${r.n_episodes}`}
                      />
                      <span className="mono text-xs text-muted-foreground">
                        avg r = {r.avg_reward.toFixed(2)}
                      </span>
                    </div>
                  </header>
                  <table className="w-full text-xs mono">
                    <thead className="text-muted-foreground border-b border-border/40">
                      <tr>
                        <th className="text-left px-4 py-1.5">ep</th>
                        <th className="text-left px-4 py-1.5">steps</th>
                        <th className="text-left px-4 py-1.5">reward</th>
                        <th className="text-left px-4 py-1.5">dist</th>
                        <th className="text-left px-4 py-1.5">result</th>
                      </tr>
                    </thead>
                    <tbody>
                      {r.episodes.map((e, j) => (
                        <tr key={j} className="border-b border-border/20 last:border-0">
                          <td className="px-4 py-1.5">{j + 1}</td>
                          <td className="px-4 py-1.5">{e.steps}</td>
                          <td className="px-4 py-1.5">{e.reward.toFixed(2)}</td>
                          <td className="px-4 py-1.5">{e.distance.toFixed(2)}</td>
                          <td className="px-4 py-1.5">
                            <StatusDot status={e.success ? "ok" : "fail"} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </section>
              ))}
            </div>
          </>
        )}
      </div>
    </>
  );
}
