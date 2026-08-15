import { apiGet } from "./client";
import type { EvidenceRecord, PhysicalVehicleRecord, TrackRecord } from "../types/track";

export function fetchTracks(query: URLSearchParams): Promise<TrackRecord[]> {
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return apiGet<TrackRecord[]>(`/api/tracks${suffix}`);
}

export function fetchVehicles(query: URLSearchParams): Promise<PhysicalVehicleRecord[]> {
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return apiGet<PhysicalVehicleRecord[]>(`/api/vehicles${suffix}`);
}

export function fetchTrack(cameraId: string, trackId: string, runId?: string): Promise<TrackRecord> {
  const suffix = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
  return apiGet<TrackRecord>(`/api/tracks/${cameraId}/${trackId}${suffix}`);
}

export function fetchTrackEvidence(cameraId: string, trackId: string, runId?: string): Promise<EvidenceRecord[]> {
  const suffix = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
  return apiGet<EvidenceRecord[]>(`/api/tracks/${cameraId}/${trackId}/evidence${suffix}`);
}
