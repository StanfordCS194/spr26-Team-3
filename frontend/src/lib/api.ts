/**
 * TanStack Query hooks. Thin wrappers around `openapi-fetch` so screens
 * can `const { data } = useProjects()` without learning React Query
 * primitives.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

// Inngest writes terminal status to the DB the moment a function finishes,
// but the frontend only sees it on the next poll. Keep this tight so the
// "running → ok" transition feels instant on fast jobs (demo_fixture is
// ~2s). For long jobs the 600ms tick is still well under network noise.
const RUNNING_POLL_MS = 600;

export type Project = {
  id: string;
  name: string;
  video_path: string | null;
  thumbnail_path: string | null;
  created_at: string;
};

export type Build = {
  id: string;
  project_id: string;
  reconstruction_id: string | null;
  mjcf_path: string;
  n_hulls: number;
  bounds: { min: number[]; max: number[] };
  spawn_region: { xmin: number; xmax: number; ymin: number; ymax: number };
  created_at: string;
};

export type ReplayEpisode = {
  steps: number;
  reward: number;
  distance: number;
  success: boolean;
};

export type ReplayResponse = {
  policy: string;
  successes: number;
  n_episodes: number;
  avg_reward: number;
  episodes: ReplayEpisode[];
};

async function get<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

async function send<T>(url: string, method: "POST" | "PATCH" | "DELETE", body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(`${url}: ${r.status} ${await r.text()}`);
  return method === "DELETE" ? (undefined as T) : (r.json() as Promise<T>);
}

export const useProjects = () =>
  useQuery({ queryKey: ["projects"], queryFn: () => get<Project[]>("/api/projects") });

export type ProjectSummary = {
  id: string;
  name: string;
  created_at: string;
  thumbnail_path: string | null;
  status_pill: string;
  n_runs: number;
};

export const useProjectsSummary = () =>
  useQuery({
    queryKey: ["projects-summary"],
    queryFn: () => get<ProjectSummary[]>("/api/projects/summary"),
    refetchInterval: 4000,
  });

export type RunRow = {
  id: string;
  policy_id: string | null;
  baseline: string | null;
  episodes: number;
  successes: number;
  avg_reward: number;
  created_at: string;
};

export const useProjectRuns = (projectId: string) =>
  useQuery({
    queryKey: ["runs", projectId],
    queryFn: () => get<RunRow[]>(`/api/projects/${projectId}/runs`),
    enabled: !!projectId,
  });

export const useProject = (id: string) =>
  useQuery({
    queryKey: ["project", id],
    queryFn: () => get<Project>(`/api/projects/${id}`),
    enabled: !!id,
  });

export type StageState = { complete: boolean; reason: string | null };
export type ProjectState = {
  capture: StageState;
  reconstruct: StageState;
  validate: StageState;
  build: StageState;
  train: StageState;
  replay: StageState;
};

export const useProjectState = (id: string) =>
  useQuery({
    queryKey: ["project-state", id],
    queryFn: () => get<ProjectState>(`/api/projects/${id}/state`),
    enabled: !!id,
    // Refetch when mutations succeed elsewhere; the mutation hooks call
    // queryClient.invalidateQueries(["project-state", id]).
  });

export const useCreateProject = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (name: string) => send<Project>("/api/projects", "POST", { name }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["projects"] }),
  });
};

export const useDeleteProject = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => send<void>(`/api/projects/${id}`, "DELETE"),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["projects"] });
      qc.invalidateQueries({ queryKey: ["projects-summary"] });
    },
  });
};

export const useRenameProject = (id: string) => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (name: string) => send<Project>(`/api/projects/${id}`, "PATCH", { name }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["projects"] });
      qc.invalidateQueries({ queryKey: ["projects-summary"] });
      qc.invalidateQueries({ queryKey: ["project", id] });
    },
  });
};

// ── Build (Inngest-backed) ────────────────────────────────────────────────

export type BuildRow = {
  id: string;
  project_id: string;
  reconstruction_id: string | null;
  mjcf_path: string | null;
  n_hulls: number | null;
  bounds: { min: number[]; max: number[] } | null;
  spawn_region: { xmin: number; xmax: number; ymin: number; ymax: number } | null;
  status: "pending" | "running" | "ok" | "failed";
  error: string | null;
  created_at: string;
};

export const useBuild = (projectId: string) => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () =>
      send<BuildRow>(`/api/projects/${projectId}/build`, "POST", { up_axis: "y" }),
    onSuccess: (created) => {
      // Seed the cache with the new pending row so the polling query
      // re-engages immediately instead of holding the previous status=ok.
      qc.setQueryData(["latest-build", projectId], created);
      qc.invalidateQueries({ queryKey: ["latest-build", projectId] });
      qc.invalidateQueries({ queryKey: ["project-state", projectId] });
    },
  });
};

export const useLatestBuild = (projectId: string) => {
  const qc = useQueryClient();
  return useQuery({
    queryKey: ["latest-build", projectId],
    queryFn: async () => {
      const next = await get<BuildRow | null>(`/api/projects/${projectId}/build/latest`);
      const prev = qc.getQueryData<BuildRow | null>(["latest-build", projectId]);
      if (
        (next?.status === "ok" || next?.status === "failed") &&
        (prev?.status === "pending" || prev?.status === "running")
      ) {
        qc.invalidateQueries({ queryKey: ["project-state", projectId] });
        qc.invalidateQueries({ queryKey: ["projects-summary"] });
      }
      return next;
    },
    refetchInterval: (q) => {
      const status = (q.state.data as BuildRow | null)?.status;
      return status === "pending" || status === "running" ? RUNNING_POLL_MS : false;
    },
    enabled: !!projectId,
  });
};

export type BackendInfo = { name: string; implemented: boolean; requires_gpu: boolean };

export const useBackends = () =>
  useQuery({
    queryKey: ["backends"],
    queryFn: () => get<BackendInfo[]>("/api/reconstruction/backends"),
    // Backend availability flips at startup (vggt depends on pip + cuda),
    // so refetch eagerly instead of caching.
    staleTime: 0,
    refetchOnMount: true,
    refetchOnWindowFocus: true,
  });

export type Reconstruction = {
  id: string;
  project_id: string;
  backend: string;
  params: Record<string, unknown>;
  mesh_path: string | null;
  status: "pending" | "running" | "ok" | "failed";
  error: string | null;
  elapsed_s: number | null;
  inngest_run_id: string | null;
  created_at: string;
};

export const useUploadVideo = (projectId: string) => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (file: File) => {
      const fd = new FormData();
      fd.append("file", file);
      const r = await fetch(`/api/projects/${projectId}/upload-video`, {
        method: "POST",
        body: fd,
      });
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      return r.json() as Promise<{ project_id: string; video_path: string; size_bytes: number }>;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["project-state", projectId] });
      qc.invalidateQueries({ queryKey: ["project", projectId] });
    },
  });
};

export const useReconstruct = (projectId: string) => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { backend: string; params?: Record<string, unknown> }) =>
      send<Reconstruction>(`/api/projects/${projectId}/reconstruct`, "POST", body),
    onSuccess: (created) => {
      // Seed the latest-reconstruction cache with the new pending row so
      // the polling query immediately flips into running mode (refetching
      // every 1.5s). Without this it would keep showing the previous
      // `status=ok` row and never notice the new job.
      qc.setQueryData(["reconstruction", projectId], created);
      qc.invalidateQueries({ queryKey: ["reconstruction", projectId] });
      qc.invalidateQueries({ queryKey: ["project-state", projectId] });
    },
  });
};

export const useLatestReconstruction = (projectId: string) => {
  const qc = useQueryClient();
  return useQuery({
    queryKey: ["reconstruction", projectId],
    queryFn: async () => {
      const next = await get<Reconstruction | null>(
        `/api/projects/${projectId}/reconstruction`,
      );
      const prev = qc.getQueryData<Reconstruction | null>(["reconstruction", projectId]);
      if (
        (next?.status === "ok" || next?.status === "failed") &&
        (prev?.status === "pending" || prev?.status === "running")
      ) {
        qc.invalidateQueries({ queryKey: ["project-state", projectId] });
        qc.invalidateQueries({ queryKey: ["projects-summary"] });
      }
      return next;
    },
    refetchInterval: (q) => {
      const status = (q.state.data as Reconstruction | null)?.status;
      return status === "pending" || status === "running" ? RUNNING_POLL_MS : false;
    },
    enabled: !!projectId,
  });
};

export type Validation = {
  id: string;
  reconstruction_id: string;
  report: import("@/components/ValidationReport").Report | null;
  user_override: boolean;
  status: "pending" | "running" | "ok" | "failed";
  error: string | null;
  created_at: string;
};

export const useLatestValidation = (projectId: string) => {
  const qc = useQueryClient();
  return useQuery({
    queryKey: ["validation", projectId],
    queryFn: async () => {
      const next = await get<Validation | null>(
        `/api/projects/${projectId}/validate/latest`,
      );
      const prev = qc.getQueryData<Validation | null>(["validation", projectId]);
      if (
        (next?.status === "ok" || next?.status === "failed") &&
        (prev?.status === "pending" || prev?.status === "running")
      ) {
        qc.invalidateQueries({ queryKey: ["project-state", projectId] });
        qc.invalidateQueries({ queryKey: ["projects-summary"] });
      }
      return next;
    },
    enabled: !!projectId,
    refetchInterval: (q) => {
      const status = (q.state.data as Validation | null)?.status;
      return status === "pending" || status === "running" ? RUNNING_POLL_MS : false;
    },
  });
};

export const useValidate = (projectId: string) => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => send<Validation>(`/api/projects/${projectId}/validate`, "POST"),
    onSuccess: (created) => {
      // Seed the latest-validation cache with the pending row so polling
      // kicks back in (was stuck on the previous status=ok).
      qc.setQueryData(["validation", projectId], created);
      qc.invalidateQueries({ queryKey: ["validation", projectId] });
      qc.invalidateQueries({ queryKey: ["project-state", projectId] });
    },
  });
};

// ── PR-C: training + trajectories ──────────────────────────────────────────

export type Policy = {
  id: string;
  build_id: string;
  algo: string;
  ckpt_path: string;
  total_steps: number;
  metrics: {
    progress?: number;
    steps?: number;
    avg_reward?: number;
    fps?: number;
    elapsed_s?: number;
    trace?: Array<{ step: number; reward: number }>;
    done?: boolean;
    error?: string;
  };
  created_at: string;
};

export const usePolicies = (projectId: string) =>
  useQuery({
    queryKey: ["policies", projectId],
    queryFn: () => get<Policy[]>(`/api/projects/${projectId}/policies`),
    enabled: !!projectId,
  });

export const useLatestPolicy = (projectId: string) => {
  const { data } = usePolicies(projectId);
  return data?.[0] ?? null;
};

export const usePolicyLive = (projectId: string, policyId: string | null | undefined) =>
  useQuery({
    queryKey: ["policy", policyId],
    queryFn: () => get<Policy>(`/api/projects/${projectId}/policies/${policyId}`),
    enabled: !!policyId,
    refetchInterval: (q) => {
      const m = (q.state.data as Policy | undefined)?.metrics;
      return m && !m.done ? 1000 : false;
    },
  });

export const useTrain = (projectId: string) => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { total_steps: number; n_envs: number; seed: number }) =>
      send<Policy>(`/api/projects/${projectId}/train`, "POST", body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["policies", projectId] });
      qc.invalidateQueries({ queryKey: ["project-state", projectId] });
    },
  });
};

export type TrajectoryPoint = { step: number; x: number; y: number; collision: boolean };
export type TrajectoryEpisode = ReplayEpisode & {
  failure_class: "success" | "timeout" | "stuck" | "collided" | "near-miss";
  spawn: number[];
  goal: number[];
  trajectory: TrajectoryPoint[] | null;
};
export type TrajectoryReplayResponse = Omit<ReplayResponse, "episodes"> & {
  bounds: { min: number[]; max: number[] };
  spawn_region: { xmin: number; xmax: number; ymin: number; ymax: number };
  episodes: TrajectoryEpisode[];
};

// ── Replay: Inngest-backed poll-then-fetch ────────────────────────────────

export type RunRowDetail = {
  id: string;
  policy_id: string | null;
  baseline: string | null;
  status: "pending" | "running" | "ok" | "failed";
  error: string | null;
  episodes: number | null;
  successes: number | null;
  avg_reward: number | null;
};

export const useStartReplay = (projectId: string) => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: {
      policy: "random" | "greedy" | "ppo";
      episodes: number;
      max_steps: number;
      seed: number;
      policy_id?: string;
    }) => send<RunRowDetail>(`/api/projects/${projectId}/replay`, "POST", body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["runs", projectId] }),
  });
};

export const useRun = (projectId: string, runId: string | null | undefined) => {
  const qc = useQueryClient();
  return useQuery({
    queryKey: ["run", projectId, runId],
    queryFn: async () => {
      const next = await get<RunRowDetail>(`/api/projects/${projectId}/runs/${runId}`);
      const prev = qc.getQueryData<RunRowDetail | undefined>(["run", projectId, runId]);
      if (
        (next?.status === "ok" || next?.status === "failed") &&
        (prev?.status === "pending" || prev?.status === "running")
      ) {
        qc.invalidateQueries({ queryKey: ["runs", projectId] });
        qc.invalidateQueries({ queryKey: ["project-state", projectId] });
      }
      return next;
    },
    enabled: !!projectId && !!runId,
    refetchInterval: (q) => {
      const status = (q.state.data as RunRowDetail | undefined)?.status;
      return status === "pending" || status === "running" ? RUNNING_POLL_MS : false;
    },
  });
};

export const useRunTrajectories = (
  projectId: string,
  runId: string | null | undefined,
  ready: boolean,
) =>
  useQuery({
    queryKey: ["run-trajectories", projectId, runId],
    queryFn: () =>
      get<{
        bounds: { min: number[]; max: number[] };
        spawn_region: { xmin: number; xmax: number; ymin: number; ymax: number };
        episodes: TrajectoryEpisode[];
      }>(`/api/projects/${projectId}/runs/${runId}/trajectories`),
    enabled: !!projectId && !!runId && ready,
    staleTime: Infinity,
  });
