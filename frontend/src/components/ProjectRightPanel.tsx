/**
 * Right-side panel for project stages. Lives in the root layout so the
 * MeshViewer doesn't remount when the user navigates between Capture /
 * Reconstruct / Validate / Build / Train / Replay.
 *
 * Reads the current project id from the route match, fetches the latest
 * successful reconstruction, shows the mesh (or a placeholder), and lets the
 * user rotate / move / scale the environment and bake that placement into the
 * mesh so the downstream Build step uses the corrected scene.
 */
import { type ReactNode, useEffect, useRef, useState } from "react";
import { useMatches } from "@tanstack/react-router";

import { IDENTITY_PLACEMENT, MeshViewer, type Placement } from "@/components/MeshViewer";
import { useApplyReconstructionTransform, useLatestReconstruction } from "@/lib/api";

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

function isIdentity(p: Placement): boolean {
  return (
    p.rx === 0 && p.ry === 0 && p.rz === 0 &&
    p.tx === 0 && p.ty === 0 && p.tz === 0 &&
    p.scale === 1
  );
}

function RightPaneForProject({ projectId }: { projectId: string }) {
  const { data: recon } = useLatestReconstruction(projectId);
  const apply = useApplyReconstructionTransform(projectId);

  const [placement, setPlacement] = useState<Placement>(IDENTITY_PLACEMENT);
  const [meshBump, setMeshBump] = useState(0);
  const matrixRef = useRef<number[]>([]);

  const status = recon?.status;
  const reconId = recon?.id;

  // A new reconstruction (or freshly baked mesh) → start from a clean placement.
  useEffect(() => {
    setPlacement(IDENTITY_PLACEMENT);
  }, [reconId]);

  const meshUrl =
    recon?.mesh_path && status === "ok"
      ? `${meshPathToUrl(recon.mesh_path, projectId)}?v=${reconId}_${meshBump}`
      : null;

  const onApply = async () => {
    if (!matrixRef.current.length) return;
    try {
      await apply.mutateAsync(matrixRef.current);
      setPlacement(IDENTITY_PLACEMENT); // baked in — reset the live transform
      setMeshBump((n) => n + 1); // reload the re-exported mesh
    } catch {
      /* surfaced via apply.isError below */
    }
  };

  return (
    <aside className="hidden lg:flex flex-col flex-1 overflow-hidden p-4 border-l border-border gap-3">
      <div className="flex-1 min-h-0">
        {meshUrl ? (
          <MeshViewer
            url={meshUrl}
            placement={placement}
            onMatrix={(m) => {
              matrixRef.current = m;
            }}
          />
        ) : (
          <div className="w-full h-full border border-dashed border-border/60 rounded-sm flex items-center justify-center text-center px-6 text-xs mono">
            {status === "running" || status === "pending" ? (
              <span className="text-muted-foreground max-w-xs animate-pulse">
                Building your 3D scene… this usually takes a minute or two and
                appears here automatically when it's ready.
              </span>
            ) : status === "failed" ? (
              <span className="text-[var(--status-fail)] max-w-xs">
                Reconstruction didn't finish: {recon?.error ?? "something went wrong."}
              </span>
            ) : (
              <span className="text-muted-foreground max-w-xs">
                No 3D scene yet — run Reconstruct to generate one.
              </span>
            )}
          </div>
        )}
      </div>

      {meshUrl && (
        <PlacementControls
          placement={placement}
          onChange={setPlacement}
          onReset={() => setPlacement(IDENTITY_PLACEMENT)}
          onApply={onApply}
          applying={apply.isPending}
          error={apply.isError}
          dirty={!isIdentity(placement)}
        />
      )}
    </aside>
  );
}

function PlacementControls({
  placement,
  onChange,
  onReset,
  onApply,
  applying,
  error,
  dirty,
}: {
  placement: Placement;
  onChange: (p: Placement) => void;
  onReset: () => void;
  onApply: () => void;
  applying: boolean;
  error: boolean;
  dirty: boolean;
}) {
  const set = (k: keyof Placement, v: number) => onChange({ ...placement, [k]: v });

  return (
    <div className="border border-border rounded-sm p-3 text-xs flex flex-col gap-2 max-h-[42%] overflow-auto">
      <div className="flex items-center justify-between">
        <span className="mono text-[11px] uppercase tracking-wider text-muted-foreground">
          Adjust placement
        </span>
        <div className="flex items-center gap-2">
          <button
            onClick={onReset}
            disabled={!dirty || applying}
            className="px-2 py-0.5 rounded-sm border border-border hover:bg-accent disabled:opacity-40"
          >
            Reset
          </button>
          <button
            onClick={onApply}
            disabled={!dirty || applying}
            className="px-2 py-0.5 rounded-sm bg-primary text-primary-foreground hover:opacity-90 disabled:opacity-40"
          >
            {applying ? "Applying…" : "Apply"}
          </button>
        </div>
      </div>

      <Group label="Rotate °">
        <Row label="X" value={placement.rx} min={-180} max={180} step={1} onChange={(v) => set("rx", v)} fmt={(v) => `${v}°`} />
        <Row label="Y" value={placement.ry} min={-180} max={180} step={1} onChange={(v) => set("ry", v)} fmt={(v) => `${v}°`} />
        <Row label="Z" value={placement.rz} min={-180} max={180} step={1} onChange={(v) => set("rz", v)} fmt={(v) => `${v}°`} />
      </Group>

      <Group label="Move">
        <Row label="X" value={placement.tx} min={-5} max={5} step={0.05} onChange={(v) => set("tx", v)} fmt={(v) => v.toFixed(2)} />
        <Row label="Y" value={placement.ty} min={-5} max={5} step={0.05} onChange={(v) => set("ty", v)} fmt={(v) => v.toFixed(2)} />
        <Row label="Z" value={placement.tz} min={-5} max={5} step={0.05} onChange={(v) => set("tz", v)} fmt={(v) => v.toFixed(2)} />
      </Group>

      <Group label="Scale">
        <Row label="×" value={placement.scale} min={0.1} max={3} step={0.05} onChange={(v) => set("scale", v)} fmt={(v) => `${v.toFixed(2)}×`} />
      </Group>

      <p className="text-[10px] text-muted-foreground/80 leading-snug">
        Apply bakes this into the scene used for Build. Rotate/scale pivot around
        the scene's centre.
      </p>
      {error && (
        <p className="text-[10px] text-[var(--status-fail)]">Couldn't apply — please try again.</p>
      )}
    </div>
  );
}

function Group({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[10px] text-muted-foreground/70 mono">{label}</span>
      {children}
    </div>
  );
}

function Row({
  label, value, min, max, step, onChange, fmt,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  fmt: (v: number) => string;
}) {
  return (
    <label className="flex items-center gap-2">
      <span className="w-3 text-muted-foreground mono">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="flex-1 accent-[var(--primary)] h-1"
      />
      <span className="w-12 text-right mono text-muted-foreground tabular-nums">{fmt(value)}</span>
    </label>
  );
}

function meshPathToUrl(meshPath: string, projectId: string): string {
  const i = meshPath.indexOf("/projects/");
  if (i < 0) return `/data/projects/${projectId}/reconstruction/mesh.ply`;
  return `/data${meshPath.slice(i)}`;
}
