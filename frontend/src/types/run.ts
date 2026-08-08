export type RunSummary = {
  run_id: string;
  status: string;
  start_time?: string | null;
  completed_at?: string | null;
  camera_count?: number | null;
  processed_frames?: number | null;
  overall_pipeline_runtime_ms?: number | null;
  duration_seconds?: number | null;
  track_count?: number | null;
  frames_by_camera?: Record<string, number>;
  run_directory?: string | null;
  runtime?: boolean;
};
