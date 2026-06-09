import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Upload } from "lucide-react";
import { useRef, useState } from "react";

import { StatusDot } from "@/components/StatusDot";
import { StepNav } from "@/components/StepNav";
import { useProject, useUploadVideo } from "@/lib/api";

export const Route = createFileRoute("/p/$projectId/capture")({
  component: Capture,
});

function Capture() {
  const { projectId } = Route.useParams();
  const navigate = useNavigate();
  const { data: project } = useProject(projectId);
  const upload = useUploadVideo(projectId);
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [picked, setPicked] = useState<File | null>(null);

  const hasVideo = !!project?.video_path;

  const onPick = (f: File | null | undefined) => {
    if (!f) return;
    setPicked(f);
  };

  const send = async () => {
    if (!picked) return;
    await upload.mutateAsync(picked);
    navigate({ to: "/p/$projectId/reconstruct", params: { projectId } });
  };

  return (
    <>
      <StepNav projectId={projectId} current="capture" />
      <div className="flex-1 p-10 overflow-auto">
        <header className="mb-8">
          <h1 className="text-2xl">Capture</h1>
          <p className="text-sm text-muted-foreground mt-1.5 max-w-xl">
            Upload a short video of a room (mp4, mov, webm) — or a photo (jpg,
            png). The backend samples ~24 frames from a video (a photo is used
            as a single frame) and feeds them to the reconstruction backend on
            the next step.
          </p>
        </header>

        <section
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            onPick(e.dataTransfer.files?.[0]);
          }}
          onClick={() => inputRef.current?.click()}
          className={
            "max-w-xl border border-dashed rounded-sm p-10 text-center cursor-pointer transition-colors " +
            (dragging ? "border-primary bg-accent/40" : "border-border hover:bg-accent/20")
          }
        >
          <input
            ref={inputRef}
            type="file"
            accept="video/mp4,video/quicktime,video/webm,image/jpeg,image/png"
            className="hidden"
            onChange={(e) => onPick(e.target.files?.[0])}
          />
          <Upload size={20} className="mx-auto text-muted-foreground mb-3" />
          {picked ? (
            <div>
              <p className="text-sm">{picked.name}</p>
              <p className="text-xs text-muted-foreground mono mt-1">
                {(picked.size / 1024 / 1024).toFixed(1)} MB · {picked.type}
              </p>
            </div>
          ) : (
            <div>
              <p className="text-sm">Drop a video or photo here, or click to pick one</p>
              <p className="text-xs text-muted-foreground mt-1">mp4 / mov / webm / jpg / png, ≤ 200 MB</p>
            </div>
          )}
        </section>

        {picked && (
          <div className="mt-6 flex items-center gap-3">
            <button
              disabled={upload.isPending}
              onClick={send}
              className="px-3 py-1.5 rounded-sm bg-primary text-primary-foreground text-sm hover:opacity-90 disabled:opacity-50"
            >
              {upload.isPending ? "Uploading…" : "Upload & continue"}
            </button>
            <button
              onClick={() => {
                setPicked(null);
                if (inputRef.current) inputRef.current.value = "";
              }}
              className="px-3 py-1.5 rounded-sm border border-border text-sm hover:bg-accent"
            >
              Pick another
            </button>
          </div>
        )}

        {hasVideo && (
          <div className="mt-8 max-w-xl border border-border rounded-sm p-4">
            <header className="flex items-center justify-between mb-2">
              <span className="mono text-[11px] uppercase tracking-wider text-muted-foreground">
                Current
              </span>
              <StatusDot status="ok" label="uploaded" />
            </header>
            <p className="text-xs mono break-all">{project!.video_path}</p>
            <button
              onClick={() => navigate({ to: "/p/$projectId/reconstruct", params: { projectId } })}
              className="mt-3 px-3 py-1.5 rounded-sm border border-border text-sm hover:bg-accent"
            >
              Go to Reconstruct →
            </button>
          </div>
        )}

        {upload.error && (
          <p className="mt-4 text-sm text-[var(--status-fail)] mono">
            {String((upload.error as Error).message)}
          </p>
        )}
      </div>
    </>
  );
}
