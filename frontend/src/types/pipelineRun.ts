export type PipelineRunStatus =
  | "QUEUED"
  | "STARTING"
  | "RUNNING"
  | "COMPLETED"
  | "FAILED"
  | "CANCEL_REQUESTED"
  | "CANCELLED";

export type PipelineRunJob = {
  job_id: string;
  config_name: string;
  status: PipelineRunStatus;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  pid?: number | null;
  exit_code?: number | null;
  run_id?: string | null;
  run_directory?: string | null;
  current_stage: string;
  processed_frames?: number | null;
  error_message?: string | null;
  log_file?: string | null;
  elapsed_seconds?: number | null;
};

export type PipelineLaunchSummary = {
  config_name: string;
  valid: boolean;
  errors: Array<{ path: string; message: string }>;
  input_sources: Array<{ camera_id?: string; source_type?: string; source?: string }>;
  tracking_roi: {
    enabled: boolean;
    mode?: string;
    rectangle?: unknown;
    anchor?: string;
  };
  plate_ocr_enabled: boolean;
  plate_detector_enabled: boolean;
  physical_identity_enabled: boolean;
  stationary_recovery_enabled: boolean;
  db_import_after_run: boolean;
};

export type PipelineRunLogs = {
  job_id: string;
  log_file: string;
  lines: string[];
  limit: number;
};

export type DbAutoImportSetting = {
  key: "DB_IMPORT_AFTER_RUN";
  enabled: boolean;
  configured_value?: string | null;
  source: string;
};
