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

export type ExperimentalVehicleIdentity = {
  vehicle_id: string;
  camera_id: string;
  member_tracks: string[];
  final_class: string;
  first_seen_frame?: number | null;
  last_seen_frame?: number | null;
  first_seen_seconds?: number | null;
  last_seen_seconds?: number | null;
  stationary?: boolean;
  contact_sheet_url?: string | null;
};

export type ExperimentalAssociationDecision = {
  track_a?: string;
  track_b?: string;
  candidate_vehicle_id?: string;
  association_mode?: string;
  decision?: string;
  association_reason?: string;
  ambiguity_reason?: string;
  score?: number | string | null;
  second_best_score?: number | string | null;
  best_member_score?: number | string | null;
  vehicle_consistency_score?: number | string | null;
  conflicting_member_count?: number | string | null;
};

export type ExperimentalVehicleIdentityResult = {
  run_id: string;
  experimental: true;
  available: boolean;
  message?: string | null;
  metrics: Record<string, unknown>;
  analytics_simulation: Record<string, unknown>;
  existing_reconciliation_baseline: Record<string, unknown>;
  config: Record<string, unknown>;
  calibration: {
    selected_config?: Record<string, unknown>;
    selected_row?: Record<string, unknown>;
    selection_policy?: string;
  };
  vehicles: ExperimentalVehicleIdentity[];
  vehicle_id_map: Record<string, string>;
  association_decisions: ExperimentalAssociationDecision[];
  paths: Record<string, string>;
};

export type StationaryRecoveredVehicle = {
  persistent_vehicle_id: string;
  source_vehicle_ids: string[];
  member_tracks: string[];
  camera_id: string;
  final_class: string;
  recovery_label?: string | null;
  recovery_confidence?: number | string | null;
  first_seen_seconds?: number | null;
  last_seen_seconds?: number | null;
  contact_sheet_url?: string | null;
};

export type StationaryRecoveryResult = {
  run_id: string;
  experimental: true;
  stage: "stationary_recovery";
  available: boolean;
  message?: string | null;
  metrics: Record<string, unknown>;
  analytics_simulation: Record<string, unknown>;
  config: Record<string, unknown>;
  calibration: {
    selected_config?: Record<string, unknown>;
    selected_row?: Record<string, unknown>;
    selection_policy?: string;
  };
  persistent_vehicles: StationaryRecoveredVehicle[];
  persistent_vehicle_id_map: Record<string, string>;
  recovery_decisions: Record<string, unknown>[];
  recovery_scores: Record<string, unknown>[];
  paths: Record<string, string>;
};
