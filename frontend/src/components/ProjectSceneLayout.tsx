/**
 * Per-stage layout for a project. Renders the StepNav header + the
 * stage-specific controls below. The persistent MeshViewer lives in
 * __root.tsx (ProjectRightPanel) so navigating between stages doesn't
 * unmount the 3D scene.
 */
import { ReactNode } from "react";

import { StepNav } from "@/components/StepNav";

type Stage = Parameters<typeof StepNav>[0]["current"];

export function ProjectSceneLayout({
  projectId,
  current,
  children,
}: {
  projectId: string;
  current: Stage;
  children: ReactNode;
}) {
  return (
    <>
      <StepNav projectId={projectId} current={current} />
      <section className="flex-1 overflow-y-auto p-8">{children}</section>
    </>
  );
}
