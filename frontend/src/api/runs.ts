import { apiGet } from "./client";
import type { RunSummary } from "../types/run";

export function fetchRuns(): Promise<RunSummary[]> {
  return apiGet<RunSummary[]>("/api/runs");
}
