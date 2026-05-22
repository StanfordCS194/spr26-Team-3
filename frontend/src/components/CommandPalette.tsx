/**
 * ⌘K project switcher. Lists projects + jump-to-stage shortcuts.
 */
import { Command } from "cmdk";
import { useNavigate } from "@tanstack/react-router";
import { useEffect, useState } from "react";

import { useProjectsSummary } from "@/lib/api";

const STAGES = ["capture", "reconstruct", "validate", "build", "train", "replay"] as const;

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [projectId, setProjectId] = useState<string | null>(null);
  const { data: projects = [] } = useProjectsSummary();
  const navigate = useNavigate();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setOpen((o) => !o);
        setProjectId(null);
      }
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-32 bg-black/40 backdrop-blur-sm"
      onClick={() => setOpen(false)}
    >
      <div onClick={(e) => e.stopPropagation()}>
        <Command
          label="command palette"
          className="w-[520px] rounded-sm border border-border bg-card shadow-xl overflow-hidden"
        >
          <Command.Input
            placeholder={projectId ? "Jump to step…" : "Search projects…"}
            className="w-full bg-background px-4 py-3 text-sm focus:outline-none border-b border-border"
          />
          <Command.List className="max-h-[360px] overflow-y-auto p-1">
            <Command.Empty className="px-4 py-6 text-xs text-muted-foreground text-center">
              no results
            </Command.Empty>

            {!projectId &&
              projects.map((p) => (
                <Command.Item
                  key={p.id}
                  value={`${p.name} ${p.id}`}
                  onSelect={() => setProjectId(p.id)}
                  className="px-3 py-2 text-sm rounded-sm cursor-pointer aria-selected:bg-accent flex items-center justify-between"
                >
                  <span>{p.name}</span>
                  <span className="mono text-[10px] text-muted-foreground">{p.status_pill}</span>
                </Command.Item>
              ))}

            {projectId &&
              STAGES.map((s) => (
                <Command.Item
                  key={s}
                  value={s}
                  onSelect={() => {
                    navigate({ to: `/p/$projectId/${s}` as any, params: { projectId } });
                    setOpen(false);
                  }}
                  className="px-3 py-2 text-sm rounded-sm cursor-pointer aria-selected:bg-accent flex items-center justify-between"
                >
                  <span className="capitalize">{s}</span>
                  <span className="mono text-[10px] text-muted-foreground">↵</span>
                </Command.Item>
              ))}
          </Command.List>
          <footer className="px-3 py-2 border-t border-border mono text-[10px] text-muted-foreground flex items-center justify-between">
            <span>⌘K to toggle · esc to close</span>
            {projectId && (
              <button
                onClick={() => setProjectId(null)}
                className="hover:text-foreground"
              >
                ← projects
              </button>
            )}
          </footer>
        </Command>
      </div>
    </div>
  );
}
