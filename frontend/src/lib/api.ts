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
