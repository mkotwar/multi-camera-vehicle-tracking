import { apiGet } from "./client";
import type { FilterOptions } from "../types/filters";

export function fetchFilterOptions(runId?: string): Promise<FilterOptions> {
  const suffix = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
  return apiGet<FilterOptions>(`/api/filter-options${suffix}`);
}
