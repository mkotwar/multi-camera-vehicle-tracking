export type ChatRole = "user" | "assistant";

export type VehicleEvidence = {
  vehicle_id: string;
  run_id?: string;
  camera_id: string;
  track_id: string;
  member_track_ids?: string[];
  vehicle_class: string;
  colour: string;
  plate_text?: string | null;
  first_seen_seconds?: number | null;
  last_seen_seconds?: number | null;
  best_crop_url?: string | null;
  image_url?: string | null;
  track_detail_url: string;
};

export type ChatVehicleQuery = {
  intent: string;
  selected_run_ids?: string[];
  include_camera_ids?: string[];
  exclude_camera_ids?: string[];
  include_classes: string[];
  exclude_classes: string[];
  include_colours: string[];
  exclude_colours: string[];
  start_time?: number | null;
  end_time?: number | null;
  camera_id?: string | null;
  group_by?: string | null;
  comparison?: Record<string, unknown> | null;
  show_evidence: boolean;
  context_reference?: string | null;
  evidence_navigation?: string | null;
};

export type EvidencePage = {
  matching_total: number;
  evidence_returned_count: number;
  evidence_offset: number;
  evidence_page_size: number;
  evidence_remaining_count: number;
  shown_count: number;
  next_offset: number;
};

export type VideoChatRequest = {
  message: string;
  run_id?: string;
  run_ids?: string[];
  session_id: string;
};

export type VideoChatResponse = {
  run_id: string;
  run_ids?: string[];
  session_id: string;
  original_query: string;
  answer: string;
  parser_used: string;
  llm_attempted?: boolean;
  llm_accepted?: boolean;
  llm_rejection_reason?: string | null;
  llm_raw_structured_output?: Record<string, unknown> | null;
  normalized_llm_output?: Record<string, unknown> | null;
  message_type?: string;
  context_was_available?: boolean;
  context_reference?: string | null;
  filters_before_context?: Record<string, unknown> | null;
  filters_after_context?: Record<string, unknown> | null;
  explicit_filters_detected?: Record<string, boolean>;
  filters_before_validation?: Record<string, unknown> | null;
  filters_after_validation?: Record<string, unknown> | null;
  group_by?: string | null;
  matching_vehicle_ids_count?: number;
  context_saved_vehicle_ids_count?: number;
  parsed_query: ChatVehicleQuery;
  analytics_result: Record<string, unknown>;
  matching_vehicle_ids: string[];
  evidence: VehicleEvidence[];
  evidence_page?: EvidencePage;
  context_used: boolean;
};

export type ChatMessage = {
  id: string;
  role: ChatRole;
  text: string;
  timestamp: string;
  evidence?: VehicleEvidence[];
  evidence_page?: EvidencePage;
  debug?: Pick<VideoChatResponse, "original_query" | "parser_used" | "llm_attempted" | "llm_accepted" | "llm_rejection_reason" | "llm_raw_structured_output" | "normalized_llm_output" | "message_type" | "context_was_available" | "context_reference" | "filters_before_context" | "filters_after_context" | "explicit_filters_detected" | "filters_before_validation" | "filters_after_validation" | "group_by" | "matching_vehicle_ids_count" | "context_saved_vehicle_ids_count" | "parsed_query" | "analytics_result" | "matching_vehicle_ids" | "context_used">;
};

export type PersistedVideoChatSession = {
  version: 1;
  session_id: string;
  run_id: string;
  run_ids?: string[];
  created_at: string;
  updated_at: string;
  messages: ChatMessage[];
};
