import { apiGet } from "./client";
import type { SystemStatus } from "../types/system";

export function fetchSystemStatus(): Promise<SystemStatus> {
  return apiGet<SystemStatus>("/api/system/status");
}
