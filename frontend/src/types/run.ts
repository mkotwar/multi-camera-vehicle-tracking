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

export type ReconciliationMetrics = {
  raw_bytetrack_unique_tracks?: number;
  reconciled_vehicle_identities?: number;
  potential_duplicate_tracks_removed?: number;
  reconciliation_attempts?: number;
  accepted_matches?: number;
  rejected_candidate_matches?: number;
  ambiguous_matches?: number;
  unmatched_tracks?: number;
  track_fragments_merged?: number;
};

export type ReconciliationAssociation = {
  old_track: string;
  new_track: string;
  vehicle_id: string;
  gap_frames?: number | string | null;
  gap_seconds?: number | string | null;
  score?: number | string | null;
  second_best_score?: number | string | null;
  colour?: string | null;
  class?: string | null;
  result?: string | null;
};

export type ReconciliationTrack = {
  run_id?: string | null;
  local_track_id: string;
  track_id?: string;
  camera_id: string;
  status?: string | null;
  vehicle_id?: string | null;
  final_class?: string | null;
  first_frame?: number | null;
  last_frame?: number | null;
  first_timestamp_seconds?: number | null;
  last_timestamp_seconds?: number | null;
  observation_count?: number | null;
  completion_reason?: string | null;
  vehicle_enrichment?: {
    vehicle_class?: string | null;
    vehicle_colour?: {
      label?: string | null;
      status?: string | null;
    } | null;
  } | null;
  reconciliation?: {
    matched?: boolean;
    previous_track_id?: string | null;
    score?: number | string | null;
    second_best_score?: number | string | null;
    time_gap_frames?: number | string | null;
    time_gap_seconds?: number | string | null;
    result?: string | null;
    reason?: string | null;
  } | null;
};

export type ManualValidationRow = {
  old_track: string;
  new_track: string;
  vehicle_id: string;
  score: string;
  manual_label: string;
  reviewer_notes: string;
};

export type ReconciliationVisualEvidence = {
  result: string;
  pair_key: string;
  contact_sheet_url?: string | null;
  before_url?: string | null;
  after_url?: string | null;
};

export type TrackReconciliationResult = {
  run_id: string;
  available: boolean;
  message?: string | null;
  metrics: ReconciliationMetrics;
  config: Record<string, unknown>;
  tracks: ReconciliationTrack[];
  accepted_associations: ReconciliationAssociation[];
  manual_validation: ManualValidationRow[];
  visual_evidence: ReconciliationVisualEvidence[];
  paths: Record<string, string>;
};
