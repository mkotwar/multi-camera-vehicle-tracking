import { apiGet } from "./client";
import type { RunSummary, TrackReconciliationResult } from "../types/run";

export function fetchRuns(): Promise<RunSummary[]> {
  return apiGet<RunSummary[]>("/api/runs");
}

export function fetchTrackReconciliation(runId: string): Promise<TrackReconciliationResult> {
  return apiGet<TrackReconciliationResult>(`/api/runs/${encodeURIComponent(runId)}/reconciliation`);
}
