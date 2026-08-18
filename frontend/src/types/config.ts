export type PipelineConfig = Record<string, unknown>;

export type ConfigListItem = {
  config_name: string;
  path: string;
  production: boolean;
  preferred: boolean;
};

export type ConfigInventoryItem = {
  path: string;
  type: string;
  default: unknown;
  required: boolean;
  restart_required: boolean;
  operator_level: "safe" | "advanced" | "experimental" | string;
  runtime_effect: string;
};

export type ConfigValidationError = {
  rule: string;
  path: string;
  message: string;
  expected?: unknown;
  actual?: unknown;
};

export type ConfigValidationResult = {
  valid: boolean;
  errors: ConfigValidationError[];
  warnings: Array<{ path: string; message: string }>;
};

export type ConfigDetail = {
  config_name: string;
  path: string;
  config: PipelineConfig;
  yaml_text: string;
  validation: ConfigValidationResult;
  inventory: ConfigInventoryItem[];
};

export type ConfigSaveResult = ConfigValidationResult & {
  saved_path: string;
  config_name: string;
  yaml_text: string;
};

export type ConfigVideoSourceUploadResult = {
  camera_id: string;
  filename: string;
  source_path: string;
  stored_path: string;
};
