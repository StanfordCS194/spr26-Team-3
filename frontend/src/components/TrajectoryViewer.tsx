/**
 * Top-down SVG trajectory viewer.
 *
 * Renders the build's bounds + spawn region as backdrop, then overlays
 * agent paths from one or more replay runs. Each run gets a color.
 */
import type { TrajectoryEpisode } from "@/lib/api";

const COLORS: Record<string, string> = {
  random: "#6b7280",
  greedy: "oklch(0.78 0.16 80)",
  ppo: "oklch(0.72 0.18 230)",
};

const POINT_COLORS = {
  start: "oklch(0.74 0.18 145)",
  goal: "oklch(0.65 0.22 25)",
  collision: "oklch(0.78 0.16 320)",
};

type Bounds = { min: number[]; max: number[] };

export type TrajectoryRun = {
  policy: string;
  bounds: Bounds;
  spawn_region: { xmin: number; xmax: number; ymin: number; ymax: number };
  episodes: TrajectoryEpisode[];
};

export function TrajectoryViewer({
  run,
  episodeIndex,
  height = 360,
}: {
  run: TrajectoryRun;
  episodeIndex?: number;
  height?: number;
}) {
  const xmin = run.bounds.min[0] - 0.3;
  const xmax = run.bounds.max[0] + 0.3;
  const ymin = run.bounds.min[1] - 0.3;
  const ymax = run.bounds.max[1] + 0.3;
  const w = xmax - xmin;
  const h = ymax - ymin;
  const aspect = w / h;

  const project = (x: number, y: number) => ({
    x: ((x - xmin) / w) * 100,
    y: 100 - ((y - ymin) / h) * 100, // flip Y so up=up
  });

  const episodes = episodeIndex !== undefined ? [run.episodes[episodeIndex]] : run.episodes;
  const color = COLORS[run.policy] ?? "oklch(0.72 0.18 230)";

  return (
    <div
      className="border border-border rounded-sm bg-card relative overflow-hidden"
      style={{ height }}
    >
      <header className="absolute top-2 left-3 z-10">
        <span className="mono text-[11px] uppercase tracking-wider text-muted-foreground">
          {run.policy} · {episodes.length}/{run.episodes.length} ep
        </span>
      </header>

      <svg
        viewBox="0 0 100 100"
        preserveAspectRatio="xMidYMid meet"
        className="w-full h-full"
        style={{ aspectRatio: aspect }}
      >
        {/* grid */}
        <defs>
          <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
            <path d="M 10 0 L 0 0 0 10" fill="none" stroke="#222" strokeWidth="0.15" />
          </pattern>
        </defs>
        <rect width="100" height="100" fill="url(#grid)" />

        {/* spawn region */}
        {(() => {
          const a = project(run.spawn_region.xmin, run.spawn_region.ymin);
          const b = project(run.spawn_region.xmax, run.spawn_region.ymax);
          return (
            <rect
              x={Math.min(a.x, b.x)}
              y={Math.min(a.y, b.y)}
              width={Math.abs(b.x - a.x)}
              height={Math.abs(b.y - a.y)}
              fill="none"
              stroke="oklch(0.65 0.22 25)"
              strokeWidth="0.25"
              strokeDasharray="1.2 1.2"
              opacity="0.4"
            />
          );
        })()}

        {/* trajectories */}
        {episodes.map((ep, i) => {
          if (!ep.trajectory || ep.trajectory.length < 2) return null;
          const points = ep.trajectory.map((p) => project(p.x, p.y));
          const d = points.map((p, idx) => `${idx === 0 ? "M" : "L"}${p.x},${p.y}`).join(" ");
          const start = project(ep.spawn[0], ep.spawn[1]);
          const goal = project(ep.goal[0], ep.goal[1]);
          const collisions = ep.trajectory.filter((p) => p.collision).map((p) => project(p.x, p.y));
          return (
            <g key={i} opacity={ep.success ? 1 : 0.55}>
              <path d={d} fill="none" stroke={color} strokeWidth="0.45" strokeLinejoin="round" />
              <circle cx={start.x} cy={start.y} r="1.1" fill={POINT_COLORS.start} stroke="#000" strokeWidth="0.15" />
              <g transform={`translate(${goal.x},${goal.y})`}>
                <polygon
                  points="0,-1.4 0.4,-0.4 1.4,-0.4 0.6,0.3 0.9,1.3 0,0.7 -0.9,1.3 -0.6,0.3 -1.4,-0.4 -0.4,-0.4"
                  fill={POINT_COLORS.goal}
                  stroke="#000"
                  strokeWidth="0.15"
                />
              </g>
              {collisions.map((c, j) => (
                <circle key={j} cx={c.x} cy={c.y} r="0.4" fill={POINT_COLORS.collision} />
              ))}
            </g>
          );
        })}
      </svg>

      <div className="absolute bottom-2 right-3 mono text-[10px] text-muted-foreground/70">
        x: [{xmin.toFixed(1)}, {xmax.toFixed(1)}] · y: [{ymin.toFixed(1)}, {ymax.toFixed(1)}]
      </div>
    </div>
  );
}
