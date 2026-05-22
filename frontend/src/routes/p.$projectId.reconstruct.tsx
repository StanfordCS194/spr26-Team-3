import { createFileRoute } from "@tanstack/react-router";

import { StepNav } from "@/components/StepNav";

export const Route = createFileRoute("/p/$projectId/reconstruct")({
  component: ReconstructPlaceholder,
});

function ReconstructPlaceholder() {
  const { projectId } = Route.useParams();
  return (
    <>
      <StepNav projectId={projectId} current="reconstruct" />
      <div className="flex-1 flex items-center justify-center text-sm text-muted-foreground mono">
        Reconstruct screen lands in PR-B.
      </div>
    </>
  );
}
