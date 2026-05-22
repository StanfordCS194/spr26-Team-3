import { createFileRoute } from "@tanstack/react-router";

import { StepNav } from "@/components/StepNav";

export const Route = createFileRoute("/p/$projectId/train")({
  component: TrainPlaceholder,
});

function TrainPlaceholder() {
  const { projectId } = Route.useParams();
  return (
    <>
      <StepNav projectId={projectId} current="train" />
      <div className="flex-1 flex items-center justify-center text-sm text-muted-foreground mono">
        Train screen lands in PR-C.
      </div>
    </>
  );
}
