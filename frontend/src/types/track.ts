export type TrackRecord = {
  run_id?: string | null;
  camera_id: string;
  track_id: string;
  local_track_id: string;
  status?: string | null;
  vehicle_class?: string | null;
  colour?: string | null;
  colour_status?: string | null;
  first_seen?: number | null;
  last_seen?: number | null;
  first_seen_seconds?: number | null;
  last_seen_seconds?: number | null;
  duration_seconds?: number | null;
  first_frame?: number | null;
  last_frame?: number | null;
  observation_count?: number | null;
  completion_reason?: string | null;
  colour_resolution?: ColourResolutionStep[];
  evidence?: EvidenceRecord[];
  best_crop?: string | null;
  best_crop_url?: string | null;
  available_crop_paths?: string[];
  runtime?: boolean;
};

export type MediaRef = {
  category: string;
  run_id: string;
  parts: string[];
  filename?: string | null;
};

export type EvidenceRecord = {
  local_track_id?: string;
  camera_id?: string;
  native_tracker_id?: number;
  frame_number?: number;
  timestamp_seconds?: number | null;
  source_image_path?: string | null;
  vehicle_crop_path?: string | null;
  annotated_frame_path?: string | null;
  bbox_xyxy?: number[];
  evidence_role?: string | null;
  detection_confidence?: number | null;
  crop_width?: number | null;
  crop_height?: number | null;
  crop_area?: number | null;
  sharpness_score?: number | null;
  brightness_score?: number | null;
  border_penalty?: number | null;
  clipping_ratio?: number | null;
  quality_score?: number | null;
  vehicle_class?: string | null;
  evidence_source?: string | null;
  selected_for_colour?: boolean | null;
  colour_crop_result?: string | null;
  crop_media?: MediaRef | null;
  full_frame_media?: MediaRef | null;
  crop_url?: string | null;
  full_frame_url?: string | null;
};

export type ColourResolutionStep = {
  index: number;
  label?: string | null;
  frame_number?: number | null;
  evidence_role?: string | null;
  quality_weight?: number | null;
  status?: string | null;
  reason?: string | null;
  crop_path?: string | null;
};
