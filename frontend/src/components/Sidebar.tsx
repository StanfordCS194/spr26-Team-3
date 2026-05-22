import { Link, useNavigate } from "@tanstack/react-router";
import { Download, MoreHorizontal, Pencil, Plus, Trash2 } from "lucide-react";
import { useState } from "react";

import { InlineRename } from "@/components/ui/InlineRename";
import {
  useCreateProject,
  useDeleteProject,
  useProjectsSummary,
  useRenameProject,
} from "@/lib/api";
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
  const [renamingId, setRenamingId] = useState<string | null>(null);

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
      <div className="px-4 py-4 border-b border-border flex items-center justify-between">
        {/* Logo matches archive/legacy/v3.html: World + <span accent>Scan</span> */}
        <Link
          to="/"
          className="text-[18px] font-bold tracking-[-0.02em] hover:opacity-80 transition-opacity"
        >
          World<span className="text-primary">Scan</span>
        </Link>
        <button
          aria-label="new project"
          onClick={() => setAdding(true)}
          title="new project"
          className="text-muted-foreground hover:text-foreground p-1 rounded-sm hover:bg-accent transition-colors"
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
        {isLoading && (
          <p className="px-4 py-3 text-xs text-muted-foreground">loading…</p>
        )}
        {!isLoading && projects.length === 0 && (
          <p className="px-4 py-3 text-xs text-muted-foreground">
            no projects yet — click + to create one
          </p>
        )}
        {projects.map((p) => (
          <SidebarItem
            key={p.id}
            project={p}
            isRenaming={renamingId === p.id}
            menuOpen={menuFor === p.id}
            onStartRename={() => {
              setRenamingId(p.id);
              setMenuFor(null);
            }}
            onStopRename={() => setRenamingId(null)}
            onToggleMenu={() => setMenuFor(menuFor === p.id ? null : p.id)}
            onCloseMenu={() => setMenuFor(null)}
            onExport={() => handleExport(p.id, p.name)}
            onDelete={() => handleDelete(p.id, p.name)}
          />
        ))}
      </nav>
    </aside>
  );
}

function SidebarItem({
  project,
  isRenaming,
  menuOpen,
  onStartRename,
  onStopRename,
  onToggleMenu,
  onCloseMenu,
  onExport,
  onDelete,
}: {
  project: { id: string; name: string; status_pill: string; n_runs: number };
  isRenaming: boolean;
  menuOpen: boolean;
  onStartRename: () => void;
  onStopRename: () => void;
  onToggleMenu: () => void;
  onCloseMenu: () => void;
  onExport: () => void;
  onDelete: () => void;
}) {
  const rename = useRenameProject(project.id);

  return (
    <div className="relative group">
      <Link
        to="/p/$projectId/build"
        params={{ projectId: project.id }}
        className="block px-4 py-2.5 text-sm border-b border-border/40 hover:bg-accent transition-colors"
        activeProps={{
          className:
            "block px-4 py-2.5 text-sm border-b border-border/40 bg-accent transition-colors",
        }}
      >
        <div className="flex items-center justify-between gap-2 pr-6">
          {isRenaming ? (
            <InlineRename
              value={project.name}
              onSave={async (next) => {
                await rename.mutateAsync(next);
                onStopRename();
              }}
              inputClassName="text-sm px-1 py-0.5"
            />
          ) : (
            <div className="truncate flex-1">{project.name}</div>
          )}
          {project.n_runs > 0 && (
            <span className="mono text-[9px] text-muted-foreground shrink-0">
              {project.n_runs}r
            </span>
          )}
        </div>
        <div className="flex items-center justify-between mt-0.5">
          <div className="text-[10px] text-muted-foreground mono truncate">
            {project.id}
          </div>
          <div className={cn("text-[10px] mono", pillClass(project.status_pill))}>
            {project.status_pill}
          </div>
        </div>
      </Link>
      <button
        aria-label="project actions"
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          onToggleMenu();
        }}
        className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded-sm hover:bg-background"
      >
        <MoreHorizontal size={12} className="text-muted-foreground" />
      </button>
      {menuOpen && (
        <div
          className="absolute z-20 top-7 right-2 bg-popover border border-border rounded-sm shadow-md min-w-[140px] py-1"
          onMouseLeave={onCloseMenu}
        >
          <button
            onClick={onStartRename}
            className="w-full text-left px-3 py-1.5 text-xs hover:bg-accent flex items-center gap-2"
          >
            <Pencil size={11} />
            Rename
          </button>
          <button
            onClick={onExport}
            className="w-full text-left px-3 py-1.5 text-xs hover:bg-accent flex items-center gap-2"
          >
            <Download size={11} />
            Export
          </button>
          <button
            onClick={onDelete}
            className="w-full text-left px-3 py-1.5 text-xs hover:bg-accent flex items-center gap-2 text-[var(--status-fail)]"
          >
            <Trash2 size={11} />
            Delete
          </button>
        </div>
      )}
    </div>
  );
}
