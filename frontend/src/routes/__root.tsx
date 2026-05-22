import { Outlet, createRootRouteWithContext } from "@tanstack/react-router";
import type { QueryClient } from "@tanstack/react-query";

import { ProjectRightPanel } from "@/components/ProjectRightPanel";
import { Sidebar } from "@/components/Sidebar";

export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  component: () => (
    <div className="flex h-screen w-screen">
      <Sidebar />
      {/*
        Layout: sidebar | stage controls (Outlet, fixed-width) | persistent
        MeshViewer. Stage routes only own the left column. MeshViewer lives
        in ProjectRightPanel here so navigating between stages of the same
        project doesn't unmount the viewer or restart the PLY load.
      */}
      <main className="w-[480px] shrink-0 flex flex-col overflow-hidden border-r border-border">
        <Outlet />
      </main>
      <ProjectRightPanel />
    </div>
  ),
});
