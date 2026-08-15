import { apiGet } from "./client";
import type { ExperimentalVehicleIdentityResult, PlateAssistedIdentityResult, RunSummary, StationaryRecoveryResult, TrackReconciliationResult } from "../types/run";

export function fetchRuns(): Promise<RunSummary[]> {
  return apiGet<RunSummary[]>("/api/runs");
}

export function fetchTrackReconciliation(runId: string): Promise<TrackReconciliationResult> {
  return apiGet<TrackReconciliationResult>(`/api/runs/${encodeURIComponent(runId)}/reconciliation`);
}

export function fetchExperimentalVehicles(runId: string): Promise<ExperimentalVehicleIdentityResult> {
  return apiGet<ExperimentalVehicleIdentityResult>(`/api/experimental/vehicles?run_id=${encodeURIComponent(runId)}`);
}

export function fetchStationaryRecoveredVehicles(runId: string): Promise<StationaryRecoveryResult> {
  return apiGet<StationaryRecoveryResult>(`/api/experimental/stationary-recovered-vehicles?run_id=${encodeURIComponent(runId)}`);
}

export function fetchPlateAssistedVehicles(runId: string): Promise<PlateAssistedIdentityResult> {
  return apiGet<PlateAssistedIdentityResult>(`/api/experimental/plate-assisted-vehicles?run_id=${encodeURIComponent(runId)}`);
}
