import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export type MetricPoint = { step: number; reward: number };

export function MetricsChart({ trace, height = 220 }: { trace: MetricPoint[]; height?: number }) {
  return (
    <div className="border border-border rounded-sm p-4 bg-card">
      <header className="mb-3 flex items-baseline justify-between">
        <span className="mono text-[11px] uppercase tracking-wider text-muted-foreground">
          PPO · avg reward
        </span>
        <span className="mono text-[11px] text-muted-foreground">
          {trace.length} points
        </span>
      </header>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={trace} margin={{ top: 8, right: 8, bottom: 8, left: -16 }}>
          <XAxis
            dataKey="step"
            tick={{ fill: "#666", fontSize: 10, fontFamily: "var(--font-mono)" }}
            stroke="#333"
            tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
          />
          <YAxis
            tick={{ fill: "#666", fontSize: 10, fontFamily: "var(--font-mono)" }}
            stroke="#333"
            width={50}
          />
          <Tooltip
            contentStyle={{
              background: "#0f0f0f",
              border: "1px solid #2a2a2a",
              fontFamily: "var(--font-mono)",
              fontSize: 11,
            }}
          />
          <Line
            type="monotone"
            dataKey="reward"
            stroke="oklch(0.72 0.18 230)"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
