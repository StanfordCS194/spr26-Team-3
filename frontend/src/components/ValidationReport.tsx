import { CheckCircle2, CircleAlert, XCircle } from "lucide-react";

import { cn } from "@/lib/utils";

export type CheckResult = {
  name: string;
  status: "pass" | "warn" | "fail";
  message: string;
  fix: string;
};

export type Report = {
  checks: CheckResult[];
  overall: "pass" | "warn" | "fail";
};

const ICONS = {
  pass: CheckCircle2,
  warn: CircleAlert,
  fail: XCircle,
} as const;

const COLORS = {
  pass: "text-[var(--status-ok)]",
  warn: "text-[var(--status-warn)]",
  fail: "text-[var(--status-fail)]",
} as const;

export function ValidationReport({ report }: { report: Report }) {
  return (
    <div className="border border-border rounded-sm divide-y divide-border">
      {report.checks.map((c) => {
        const Icon = ICONS[c.status];
        return (
          <div key={c.name} className="px-4 py-3 flex items-start gap-3">
            <Icon size={16} className={cn("mt-0.5 shrink-0", COLORS[c.status])} />
            <div className="flex-1 min-w-0">
              <div className="flex items-baseline gap-3">
                <span className="mono text-[11px] uppercase tracking-wider text-muted-foreground">
                  {c.name.replace(/_/g, " ")}
                </span>
                <span className="text-sm">{c.message}</span>
              </div>
              {c.fix && (
                <p className="text-xs text-muted-foreground mt-1">
                  {c.fix}
                </p>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
