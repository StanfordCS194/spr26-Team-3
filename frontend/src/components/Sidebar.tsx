import { Link } from "@tanstack/react-router";
import { Plus } from "lucide-react";
import { useState } from "react";

import { useCreateProject, useProjects } from "@/lib/api";
import { cn } from "@/lib/utils";

export function Sidebar() {
  const { data: projects = [], isLoading } = useProjects();
  const createProject = useCreateProject();
  const [adding, setAdding] = useState(false);
  const [name, setName] = useState("");

  return (
    <aside className="w-64 border-r border-border bg-card flex flex-col h-full">
      <div className="px-4 py-3 border-b border-border flex items-center justify-between">
        <span className="text-xs uppercase tracking-wider text-muted-foreground mono">
          WorldScan
        </span>
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
            await createProject.mutateAsync(name.trim());
            setName("");
            setAdding(false);
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
          <Link
            key={p.id}
            to="/p/$projectId/build"
            params={{ projectId: p.id }}
            className={({ isActive }: { isActive: boolean }) =>
              cn(
                "block px-4 py-2 text-sm border-b border-border/40 hover:bg-accent",
                isActive && "bg-accent",
              )
            }
          >
            <div className="truncate">{p.name}</div>
            <div className="text-[10px] text-muted-foreground mono mt-0.5">{p.id}</div>
          </Link>
        ))}
      </nav>
    </aside>
  );
}
