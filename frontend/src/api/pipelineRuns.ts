import { apiGet, apiPost, apiPut } from "./client";
import type { DbAutoImportSetting, PipelineLaunchSummary, PipelineRunJob, PipelineRunLogs } from "../types/pipelineRun";

export function fetchPipelineRunJobs(): Promise<PipelineRunJob[]> {
  return apiGet<PipelineRunJob[]>("/api/pipeline-runs");
}

export function fetchPipelineRunJob(jobId: string): Promise<PipelineRunJob> {
  return apiGet<PipelineRunJob>(`/api/pipeline-runs/${encodeURIComponent(jobId)}`);
}

export function fetchPipelineRunLogs(jobId: string, limit = 200): Promise<PipelineRunLogs> {
  return apiGet<PipelineRunLogs>(`/api/pipeline-runs/${encodeURIComponent(jobId)}/logs?limit=${limit}`);
}

export function fetchPipelineLaunchSummary(configName: string): Promise<PipelineLaunchSummary> {
  return apiGet<PipelineLaunchSummary>(`/api/pipeline-runs/launch-summary/${encodeURIComponent(configName)}`);
}

export function startPipelineRun(configName: string): Promise<PipelineRunJob> {
  return apiPost<PipelineRunJob>("/api/pipeline-runs", { config_name: configName });
}

export function cancelPipelineRun(jobId: string): Promise<PipelineRunJob> {
  return apiPost<PipelineRunJob>(`/api/pipeline-runs/${encodeURIComponent(jobId)}/cancel`, {});
}

export function fetchDbAutoImportSetting(): Promise<DbAutoImportSetting> {
  return apiGet<DbAutoImportSetting>("/api/runtime-settings/db-auto-import");
}

export function updateDbAutoImportSetting(enabled: boolean): Promise<DbAutoImportSetting> {
  return apiPut<DbAutoImportSetting>("/api/runtime-settings/db-auto-import", { enabled });
}
