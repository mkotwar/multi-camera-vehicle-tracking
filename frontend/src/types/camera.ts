export type LiveDetection = {
  track_id: string;
  local_track_id?: string;
  vehicle_class: string | null;
  bbox: number[];
  confidence: number | null;
  colour: string | null;
  colour_status: string | null;
};

export type CameraState = {
  camera_id: string;
  name: string;
  status: string;
  frame_number: number | null;
  timestamp_seconds: number | null;
  processed_fps: number;
  input_fps: number | null;
  active_vehicle_count: number;
  active_track_ids: string[];
  detections: LiveDetection[];
  last_update: string | number | null;
  source_type?: string | null;
  source?: string | null;
  run_id?: string | null;
  frame_url?: string | null;
};
