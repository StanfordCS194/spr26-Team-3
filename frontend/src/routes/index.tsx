import { createFileRoute } from "@tanstack/react-router";

import { useProjects } from "@/lib/api";

export const Route = createFileRoute("/")({
  component: ProjectList,
});

function ProjectList() {
  const { data: projects = [], isLoading } = useProjects();

  return (
    <div className="flex-1 p-8 flex flex-col">
      <header className="mb-6">
        <h1 className="text-xl font-medium">Projects</h1>
        <p className="text-sm text-muted-foreground mt-1">
          One row per captured scene. Use the sidebar to create a new one.
        </p>
      </header>

      {isLoading && <p className="text-sm text-muted-foreground">loading…</p>}
      {!isLoading && projects.length === 0 && (
        <div className="border border-dashed border-border rounded-sm p-12 text-center text-sm text-muted-foreground">
          no projects yet
        </div>
      )}

      <ul className="space-y-1">
        {projects.map((p) => (
          <li
            key={p.id}
            className="flex items-center justify-between px-4 py-2 hover:bg-accent rounded-sm border border-border/40"
          >
            <div>
              <div className="text-sm">{p.name}</div>
              <div className="text-[10px] mono text-muted-foreground">{p.id}</div>
            </div>
            <div className="text-[10px] mono text-muted-foreground">
              {new Date(p.created_at).toLocaleString()}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
