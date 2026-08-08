import { apiGet } from "./client";
import type { CameraState } from "../types/camera";

export function fetchCameras(runId?: string): Promise<CameraState[]> {
  const suffix = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
  return apiGet<CameraState[]>(`/api/cameras${suffix}`);
}

export function fetchCamera(cameraId: string, runId?: string): Promise<CameraState> {
  const suffix = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
  return apiGet<CameraState>(`/api/cameras/${cameraId}${suffix}`);
}
