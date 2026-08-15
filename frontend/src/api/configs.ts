import { apiGet, apiPost, apiPut, apiUrl } from "./client";
import type { ConfigDetail, ConfigListItem, ConfigSaveResult, ConfigValidationResult, PipelineConfig } from "../types/config";

export async function listConfigs(): Promise<ConfigListItem[]> {
  const payload = await apiGet<{ configs: ConfigListItem[] }>("/api/configs");
  return payload.configs;
}

export function getConfig(configName: string): Promise<ConfigDetail> {
  return apiGet<ConfigDetail>(`/api/configs/${encodeURIComponent(configName)}`);
}

export function validateConfig(configName: string, config: PipelineConfig): Promise<ConfigValidationResult> {
  return apiPost<ConfigValidationResult>(`/api/configs/${encodeURIComponent(configName)}/validate`, { config });
}

export function saveConfig(configName: string, config: PipelineConfig): Promise<ConfigSaveResult> {
  return apiPut<ConfigSaveResult>(`/api/configs/${encodeURIComponent(configName)}`, { config });
}

export function cloneConfig(configName: string, newName: string, config: PipelineConfig): Promise<ConfigSaveResult> {
  return apiPost<ConfigSaveResult>(`/api/configs/${encodeURIComponent(configName)}/clone`, { new_name: newName, config });
}

export function roiPreviewUrl(configName: string, cameraId: string): string {
  return apiUrl(`/api/configs/${encodeURIComponent(configName)}/roi-preview?camera_id=${encodeURIComponent(cameraId)}`);
}
