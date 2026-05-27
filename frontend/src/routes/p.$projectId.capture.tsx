import { createFileRoute } from "@tanstack/react-router";

import { StepNav } from "@/components/StepNav";

export const Route = createFileRoute("/p/$projectId/capture")({
  component: CapturePlaceholder,
});

function CapturePlaceholder() {
  const { projectId } = Route.useParams();
  return (
    <>
      <StepNav projectId={projectId} current="capture" />
      <div className="flex-1 flex items-center justify-center text-sm text-muted-foreground mono">
        Capture screen lands in PR-B.
      </div>
    </>
  );
}
