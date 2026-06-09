/**
 * Head-to-head: greedy vs PPO on the SAME start → goal. Both paths overlaid on
 * one top-down map so you can see greedy stall on an obstacle while PPO routes
 * around it (or, in an open room, that they're the same — which is the point).
 */
import type { CompareResponse } from "@/lib/api";

const POLICY_COLOR: Record<string, string> = {
  greedy: "oklch(0.78 0.16 80)", // amber
  ppo: "oklch(0.72 0.18 230)", // blue
  random: "#6b7280",
};

export function CompareViewer({ data, height = 300 }: { data: CompareResponse; height?: number }) {
  const xmin = data.bounds.min[0] - 0.3;
  const xmax = data.bounds.max[0] + 0.3;
  const ymin = data.bounds.min[1] - 0.3;
  const ymax = data.bounds.max[1] + 0.3;
  const w = xmax - xmin;
  const h = ymax - ymin;
  const project = (x: number, y: number) => ({
    x: ((x - xmin) / w) * 100,
    y: 100 - ((y - ymin) / h) * 100,
  });

  const ref = data.results[0];
  const start = ref ? project(ref.spawn[0], ref.spawn[1]) : { x: 50, y: 50 };
  const goal = ref ? project(ref.goal[0], ref.goal[1]) : { x: 50, y: 50 };

  return (
    <div className="border border-border rounded-sm bg-card relative overflow-hidden" style={{ height }}>
      <svg viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet" className="w-full h-full" style={{ aspectRatio: w / h }}>
        <defs>
          <pattern id="cgrid" width="10" height="10" patternUnits="userSpaceOnUse">
            <path d="M 10 0 L 0 0 0 10" fill="none" stroke="#222" strokeWidth="0.15" />
          </pattern>
        </defs>
        <rect width="100" height="100" fill="url(#cgrid)" />

        {data.results.map((r) => {
          const t = r.trajectory ?? [];
          if (t.length < 2) return null;
          const pts = t.map((p) => project(p.x, p.y));
          const d = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.y}`).join(" ");
          const color = POLICY_COLOR[r.policy] ?? "#aaa";
          const cols = t.filter((p) => p.collision).map((p) => project(p.x, p.y));
          return (
            <g key={r.policy}>
              <path d={d} fill="none" stroke={color} strokeWidth="0.6" strokeLinejoin="round" strokeLinecap="round"
                opacity={r.success ? 1 : 0.85} />
              {cols.map((c, j) => (
                <circle key={j} cx={c.x} cy={c.y} r="0.5" fill="oklch(0.78 0.16 320)" />
              ))}
            </g>
          );
        })}

        {/* shared start + goal */}
        <circle cx={start.x} cy={start.y} r="1.3" fill="oklch(0.74 0.18 145)" stroke="#000" strokeWidth="0.18" />
        <g transform={`translate(${goal.x},${goal.y})`}>
          <polygon points="0,-1.6 0.45,-0.45 1.6,-0.45 0.7,0.35 1.0,1.5 0,0.8 -1.0,1.5 -0.7,0.35 -1.6,-0.45 -0.45,-0.45"
            fill="oklch(0.65 0.22 25)" stroke="#000" strokeWidth="0.18" />
        </g>
      </svg>

      <div className="absolute top-2 left-3 right-3 flex flex-wrap gap-x-4 gap-y-1 mono text-[10px]">
        {data.results.map((r) => (
          <span key={r.policy} className="flex items-center gap-1.5">
            <span className="inline-block w-2.5 h-1 rounded" style={{ background: POLICY_COLOR[r.policy] ?? "#aaa" }} />
            <span className="text-foreground uppercase">{r.policy}</span>
            <span className={r.success ? "text-[var(--status-ok)]" : "text-[var(--status-fail)]"}>
              {r.success ? `✓ ${r.steps} steps` : `✗ stuck (${r.steps})`}
            </span>
            <span className="text-muted-foreground">· {r.collisions} hits</span>
          </span>
        ))}
      </div>
    </div>
  );
}
