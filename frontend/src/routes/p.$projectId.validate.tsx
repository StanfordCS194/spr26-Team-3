import { createFileRoute } from "@tanstack/react-router";

import { StepNav } from "@/components/StepNav";

export const Route = createFileRoute("/p/$projectId/validate")({
  component: ValidatePlaceholder,
});

function ValidatePlaceholder() {
  const { projectId } = Route.useParams();
  return (
    <>
      <StepNav projectId={projectId} current="validate" />
      <div className="flex-1 flex items-center justify-center text-sm text-muted-foreground mono">
        Validate screen lands in PR-B.
      </div>
    </>
  );
}
