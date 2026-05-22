/**
 * TanStack Query hooks. Thin wrappers around `openapi-fetch` so screens
 * can `const { data } = useProjects()` without learning React Query
 * primitives.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

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
    onSuccess: () => qc.invalidateQueries({ queryKey: ["projects"] }),
  });
};

export const useBuild = (projectId: string) => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => send<Build>(`/api/projects/${projectId}/build`, "POST", { up_axis: "y" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["project-state", projectId] }),
  });
};

export const useReplay = (projectId: string) =>
  useMutation({
    mutationFn: (body: { policy: "random" | "greedy"; episodes: number; max_steps: number; seed: number }) =>
      send<ReplayResponse>(`/api/projects/${projectId}/replay`, "POST", body),
  });

export type BackendInfo = { name: string; implemented: boolean; requires_gpu: boolean };

export const useBackends = () =>
  useQuery({
    queryKey: ["backends"],
    queryFn: () => get<BackendInfo[]>("/api/reconstruction/backends"),
    staleTime: 60_000,
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
    onSuccess: () => qc.invalidateQueries({ queryKey: ["project-state", projectId] }),
  });
};

export const useLatestReconstruction = (projectId: string) =>
  useQuery({
    queryKey: ["reconstruction", projectId],
    queryFn: () => get<Reconstruction | null>(`/api/projects/${projectId}/reconstruction`),
    refetchInterval: (q) => {
      const status = (q.state.data as Reconstruction | null)?.status;
      return status === "pending" || status === "running" ? 1500 : false;
    },
    enabled: !!projectId,
  });

export type Validation = {
  id: string;
  reconstruction_id: string;
  report: import("@/components/ValidationReport").Report;
  user_override: boolean;
  created_at: string;
};

export const useLatestValidation = (projectId: string) =>
  useQuery({
    queryKey: ["validation", projectId],
    queryFn: () => get<Validation | null>(`/api/projects/${projectId}/validate/latest`),
    enabled: !!projectId,
  });

export const useValidate = (projectId: string) => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => send<Validation>(`/api/projects/${projectId}/validate`, "POST"),
    onSuccess: () => {
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

export const useReplayWithTrajectories = (projectId: string) =>
  useMutation({
    mutationFn: (body: {
      policy: "random" | "greedy" | "ppo";
      episodes: number;
      max_steps: number;
      seed: number;
      policy_id?: string;
    }) =>
      send<TrajectoryReplayResponse>(`/api/projects/${projectId}/replay`, "POST", {
        ...body,
        include_trajectories: true,
      }),
  });
