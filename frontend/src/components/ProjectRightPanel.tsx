/**
 * Right-side panel for project stages. Lives in the root layout so the
 * MeshViewer doesn't remount when the user navigates between Capture /
 * Reconstruct / Validate / Build / Train / Replay.
 *
 * Reads the current project id from the route match, fetches the latest
 * successful reconstruction, and shows the mesh (or a placeholder).
 */
import { useMatches } from "@tanstack/react-router";

import { MeshViewer } from "@/components/MeshViewer";
import { useLatestReconstruction } from "@/lib/api";

export function ProjectRightPanel() {
  const matches = useMatches();

  // Match TanStack Router's generated route ids: "/p/$projectId/build" etc.
  const projectMatch = matches.find(
    (m) =>
      typeof m.routeId === "string" &&
      m.routeId.startsWith("/p/$projectId"),
  );
  const projectId = (projectMatch?.params as { projectId?: string } | undefined)?.projectId;

  if (!projectId) {
    return null;
  }

  return <RightPaneForProject projectId={projectId} />;
}

function RightPaneForProject({ projectId }: { projectId: string }) {
  const { data: recon } = useLatestReconstruction(projectId);

  const status = recon?.status;
  const meshUrl =
    recon?.mesh_path && status === "ok"
      ? meshPathToUrl(recon.mesh_path, projectId)
      : null;

  return (
    <aside className="hidden lg:flex flex-1 overflow-hidden p-4 border-l border-border">
      {meshUrl ? (
        <MeshViewer url={meshUrl} />
      ) : (
        <div className="w-full h-full border border-dashed border-border/60 rounded-sm flex items-center justify-center text-center px-6 text-xs mono">
          {status === "running" || status === "pending" ? (
            <span className="text-muted-foreground animate-pulse">
              reconstructing… the scene will appear here when it finishes
            </span>
          ) : status === "failed" ? (
            <span className="text-[var(--status-fail)] max-w-xs">
              reconstruction failed: {recon?.error ?? "unknown error"}
            </span>
          ) : (
            <span className="text-muted-foreground">
              no mesh yet — finish Reconstruct to see the scene here
            </span>
          )}
        </div>
      )}
    </aside>
  );
}

function meshPathToUrl(meshPath: string, projectId: string): string {
  const i = meshPath.indexOf("/projects/");
  if (i < 0) return `/data/projects/${projectId}/reconstruction/mesh.ply`;
  return `/data${meshPath.slice(i)}`;
}
