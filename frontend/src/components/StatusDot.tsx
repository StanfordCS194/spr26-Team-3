import { cn } from "@/lib/utils";

type Status = "ok" | "warn" | "fail" | "idle";

export function StatusDot({ status, label }: { status: Status; label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 mono text-xs">
      <span
        className={cn(
          "inline-block size-2 rounded-full",
          status === "ok" && "bg-[var(--status-ok)]",
          status === "warn" && "bg-[var(--status-warn)]",
          status === "fail" && "bg-[var(--status-fail)]",
          status === "idle" && "bg-muted-foreground/40",
        )}
      />
      {label && <span className="text-muted-foreground">{label}</span>}
    </span>
  );
}
