import { Link, useNavigate } from "@tanstack/react-router";
import { Download, MoreHorizontal, Plus, Trash2 } from "lucide-react";
import { useState } from "react";

import { useCreateProject, useDeleteProject, useProjectsSummary } from "@/lib/api";
import { cn } from "@/lib/utils";

const PILL_COLORS: Record<string, string> = {
  New: "text-muted-foreground/60",
  Captured: "text-[var(--status-warn)]",
  Reconstructed: "text-[var(--status-warn)]",
  Validated: "text-[var(--status-warn)]",
  "Validation failed": "text-[var(--status-fail)]",
  Built: "text-[var(--status-ok)]",
  Trained: "text-[var(--status-ok)]",
};

function pillClass(pill: string): string {
  if (pill.startsWith("Trained")) return PILL_COLORS["Trained"];
  return PILL_COLORS[pill] ?? "text-muted-foreground/60";
}

export function Sidebar() {
  const { data: projects = [], isLoading } = useProjectsSummary();
  const createProject = useCreateProject();
  const deleteProject = useDeleteProject();
  const navigate = useNavigate();
  const [adding, setAdding] = useState(false);
  const [name, setName] = useState("");
  const [menuFor, setMenuFor] = useState<string | null>(null);

  const handleDelete = async (id: string, projectName: string) => {
    const confirm = window.prompt(
      `Type "${projectName}" to confirm delete (cascades, can't be undone):`,
    );
    if (confirm === projectName) {
      await deleteProject.mutateAsync(id);
      setMenuFor(null);
      navigate({ to: "/" });
    }
  };

  const handleExport = (id: string, projectName: string) => {
    // Trigger streaming download by navigating to the export endpoint
    const link = document.createElement("a");
    link.href = `/api/projects/${id}/export`;
    link.download = `${projectName}-${id}.zip`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setMenuFor(null);
  };

  return (
    <aside className="w-72 border-r border-border bg-card flex flex-col h-full">
      <div className="px-4 py-3 border-b border-border flex items-center justify-between">
        <Link to="/" className="text-xs uppercase tracking-wider text-muted-foreground mono hover:text-foreground">
          WorldScan
        </Link>
        <button
          aria-label="new project"
          onClick={() => setAdding(true)}
          className="text-muted-foreground hover:text-foreground"
        >
          <Plus size={14} />
        </button>
      </div>

      {adding && (
        <form
          className="px-4 py-2 border-b border-border"
          onSubmit={async (e) => {
            e.preventDefault();
            if (!name.trim()) return;
            const p = await createProject.mutateAsync(name.trim());
            setName("");
            setAdding(false);
            navigate({ to: "/p/$projectId/capture", params: { projectId: p.id } });
          }}
        >
          <input
            autoFocus
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="project name"
            onBlur={() => !name && setAdding(false)}
            className="w-full bg-transparent text-sm border-b border-border focus:outline-none focus:border-primary px-1 py-1"
          />
        </form>
      )}

      <nav className="flex-1 overflow-y-auto">
        {isLoading && <p className="px-4 py-3 text-xs text-muted-foreground">loading…</p>}
        {!isLoading && projects.length === 0 && (
          <p className="px-4 py-3 text-xs text-muted-foreground">
            no projects yet — click + to create one
          </p>
        )}
        {projects.map((p) => (
          <div key={p.id} className="relative group">
            <Link
              to="/p/$projectId/build"
              params={{ projectId: p.id }}
              className={({ isActive }: { isActive: boolean }) =>
                cn(
                  "block px-4 py-2.5 text-sm border-b border-border/40 hover:bg-accent transition-colors",
                  isActive && "bg-accent",
                )
              }
            >
              <div className="flex items-center justify-between gap-2 pr-6">
                <div className="truncate flex-1">{p.name}</div>
                {p.n_runs > 0 && (
                  <span className="mono text-[9px] text-muted-foreground">{p.n_runs}r</span>
                )}
              </div>
              <div className="flex items-center justify-between mt-0.5">
                <div className="text-[10px] text-muted-foreground mono truncate">{p.id}</div>
                <div className={cn("text-[10px] mono", pillClass(p.status_pill))}>
                  {p.status_pill}
                </div>
              </div>
            </Link>
            <button
              aria-label="project actions"
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setMenuFor(menuFor === p.id ? null : p.id);
              }}
              className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded-sm hover:bg-background"
            >
              <MoreHorizontal size={12} className="text-muted-foreground" />
            </button>
            {menuFor === p.id && (
              <div
                className="absolute z-20 top-7 right-2 bg-card border border-border rounded-sm shadow-md min-w-[140px] py-1"
                onMouseLeave={() => setMenuFor(null)}
              >
                <button
                  onClick={() => handleExport(p.id, p.name)}
                  className="w-full text-left px-3 py-1.5 text-xs hover:bg-accent flex items-center gap-2"
                >
                  <Download size={11} />
                  Export
                </button>
                <button
                  onClick={() => handleDelete(p.id, p.name)}
                  className="w-full text-left px-3 py-1.5 text-xs hover:bg-accent flex items-center gap-2 text-[var(--status-fail)]"
                >
                  <Trash2 size={11} />
                  Delete
                </button>
              </div>
            )}
          </div>
        ))}
      </nav>
    </aside>
  );
}
