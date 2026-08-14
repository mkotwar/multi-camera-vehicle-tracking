-- Canonical vehicle analytics v1 schema.
-- Additive migration only: no DROP, TRUNCATE, or DELETE statements.

create extension if not exists pgcrypto;

create table if not exists public.processing_runs (
    id uuid primary key default gen_random_uuid(),
    run_key text not null unique,
    status text not null,
    started_at timestamptz,
    completed_at timestamptz,
    project_name text,
    detection_backend text,
    tracking_backend text,
    enrichment_enabled boolean,
    processed_frames bigint check (processed_frames is null or processed_frames >= 0),
    total_detections bigint check (total_detections is null or total_detections >= 0),
    raw_yolo_detections bigint check (raw_yolo_detections is null or raw_yolo_detections >= 0),
    roi_filtered_detections bigint check (roi_filtered_detections is null or roi_filtered_detections >= 0),
    completed_tracks integer check (completed_tracks is null or completed_tracks >= 0),
    discarded_tracks integer check (discarded_tracks is null or discarded_tracks >= 0),
    config_path text,
    config_snapshot jsonb not null default '{}'::jsonb,
    metrics jsonb not null default '{}'::jsonb,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    check (completed_at is null or started_at is null or completed_at >= started_at)
);

create table if not exists public.run_cameras (
    id uuid primary key default gen_random_uuid(),
    run_id uuid not null references public.processing_runs(id) on delete cascade,
    camera_key text not null,
    source_uri text,
    source_type text,
    enabled boolean,
    frame_width integer check (frame_width is null or frame_width > 0),
    frame_height integer check (frame_height is null or frame_height > 0),
    fps double precision check (fps is null or fps > 0),
    frames_processed bigint check (frames_processed is null or frames_processed >= 0),
    detections_count bigint check (detections_count is null or detections_count >= 0),
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (run_id, camera_key)
);

create table if not exists public.vehicle_tracks (
    id uuid primary key default gen_random_uuid(),
    run_id uuid not null references public.processing_runs(id) on delete cascade,
    camera_id uuid not null references public.run_cameras(id) on delete cascade,
    camera_key text not null,
    local_track_id text not null,
    short_track_id text,
    tracker_namespace text,
    native_tracker_id text,
    track_status text not null,
    completion_reason text,
    first_frame bigint check (first_frame is null or first_frame >= 0),
    last_frame bigint check (last_frame is null or last_frame >= 0),
    first_seen_seconds double precision check (first_seen_seconds is null or first_seen_seconds >= 0),
    last_seen_seconds double precision check (last_seen_seconds is null or last_seen_seconds >= 0),
    observation_count integer check (observation_count is null or observation_count >= 0),
    lost_frames integer check (lost_frames is null or lost_frames >= 0),
    vehicle_class text,
    vehicle_class_confidence double precision check (vehicle_class_confidence is null or (vehicle_class_confidence >= 0 and vehicle_class_confidence <= 1)),
    vehicle_colour text,
    vehicle_colour_status text,
    body_type text,
    body_type_status text,
    plate_text text,
    plate_detected boolean,
    plate_colour text,
    registration_category text,
    class_counts jsonb not null default '{}'::jsonb,
    class_confidence_sums jsonb not null default '{}'::jsonb,
    raw_track jsonb not null default '{}'::jsonb,
    enrichment_summary jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (run_id, camera_id, local_track_id),
    check (last_frame is null or first_frame is null or last_frame >= first_frame),
    check (last_seen_seconds is null or first_seen_seconds is null or last_seen_seconds >= first_seen_seconds),
    check (plate_detected is distinct from false or plate_text is null)
);

create table if not exists public.media_assets (
    id uuid primary key default gen_random_uuid(),
    run_id uuid not null references public.processing_runs(id) on delete cascade,
    camera_id uuid references public.run_cameras(id) on delete set null,
    track_id uuid references public.vehicle_tracks(id) on delete cascade,
    media_type text not null,
    storage_provider text not null default 'local',
    bucket text,
    object_key text,
    relative_path text,
    width integer check (width is null or width > 0),
    height integer check (height is null or height > 0),
    frame_number bigint check (frame_number is null or frame_number >= 0),
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (run_id, media_type, relative_path),
    check (relative_path is not null or object_key is not null)
);

create table if not exists public.track_observations (
    id bigserial primary key,
    track_id uuid not null references public.vehicle_tracks(id) on delete cascade,
    run_id uuid not null references public.processing_runs(id) on delete cascade,
    camera_id uuid not null references public.run_cameras(id) on delete cascade,
    frame_number bigint not null check (frame_number >= 0),
    timestamp_seconds double precision not null check (timestamp_seconds >= 0),
    bbox_x1 double precision,
    bbox_y1 double precision,
    bbox_x2 double precision,
    bbox_y2 double precision,
    detection_confidence double precision check (detection_confidence is null or (detection_confidence >= 0 and detection_confidence <= 1)),
    raw_class_id integer,
    raw_class_name text,
    tracker_namespace text,
    native_tracker_id text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (track_id, frame_number),
    check (bbox_x2 is null or bbox_x1 is null or bbox_x2 >= bbox_x1),
    check (bbox_y2 is null or bbox_y1 is null or bbox_y2 >= bbox_y1)
);

create table if not exists public.track_evidence (
    id uuid primary key default gen_random_uuid(),
    track_id uuid not null references public.vehicle_tracks(id) on delete cascade,
    run_id uuid not null references public.processing_runs(id) on delete cascade,
    camera_id uuid not null references public.run_cameras(id) on delete cascade,
    evidence_role text,
    frame_number bigint check (frame_number is null or frame_number >= 0),
    timestamp_seconds double precision check (timestamp_seconds is null or timestamp_seconds >= 0),
    crop_media_id uuid references public.media_assets(id) on delete set null,
    source_frame_media_id uuid references public.media_assets(id) on delete set null,
    annotated_frame_media_id uuid references public.media_assets(id) on delete set null,
    bbox_x1 double precision,
    bbox_y1 double precision,
    bbox_x2 double precision,
    bbox_y2 double precision,
    original_bbox_x1 double precision,
    original_bbox_y1 double precision,
    original_bbox_x2 double precision,
    original_bbox_y2 double precision,
    expanded_crop_bbox_x1 double precision,
    expanded_crop_bbox_y1 double precision,
    expanded_crop_bbox_x2 double precision,
    expanded_crop_bbox_y2 double precision,
    detection_confidence double precision check (detection_confidence is null or (detection_confidence >= 0 and detection_confidence <= 1)),
    quality_score double precision,
    sharpness_score double precision,
    brightness_score double precision,
    crop_width integer check (crop_width is null or crop_width > 0),
    crop_height integer check (crop_height is null or crop_height > 0),
    resolution_tier text,
    selected_for_colour boolean,
    selected_for_body_type boolean,
    evidence_source text,
    candidate_rank integer check (candidate_rank is null or candidate_rank >= 0),
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (track_id, evidence_role, frame_number),
    check (bbox_x2 is null or bbox_x1 is null or bbox_x2 >= bbox_x1),
    check (bbox_y2 is null or bbox_y1 is null or bbox_y2 >= bbox_y1),
    check (original_bbox_x2 is null or original_bbox_x1 is null or original_bbox_x2 >= original_bbox_x1),
    check (original_bbox_y2 is null or original_bbox_y1 is null or original_bbox_y2 >= original_bbox_y1),
    check (expanded_crop_bbox_x2 is null or expanded_crop_bbox_x1 is null or expanded_crop_bbox_x2 >= expanded_crop_bbox_x1),
    check (expanded_crop_bbox_y2 is null or expanded_crop_bbox_y1 is null or expanded_crop_bbox_y2 >= expanded_crop_bbox_y1)
);

create table if not exists public.colour_predictions (
    id uuid primary key default gen_random_uuid(),
    track_id uuid not null references public.vehicle_tracks(id) on delete cascade,
    evidence_id uuid references public.track_evidence(id) on delete set null,
    media_id uuid references public.media_assets(id) on delete set null,
    predicted_colour text,
    normalized_colour text,
    status text,
    confidence double precision check (confidence is null or (confidence >= 0 and confidence <= 1)),
    model_name text,
    model_version text,
    prompt text,
    raw_response text,
    inference_time_ms double precision check (inference_time_ms is null or inference_time_ms >= 0),
    fallback_attempt integer check (fallback_attempt is null or fallback_attempt >= 0),
    selection_reason text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create table if not exists public.vehicle_attribute_predictions (
    id uuid primary key default gen_random_uuid(),
    track_id uuid not null references public.vehicle_tracks(id) on delete cascade,
    evidence_id uuid references public.track_evidence(id) on delete set null,
    media_id uuid references public.media_assets(id) on delete set null,
    attribute_type text not null,
    label text,
    normalized_label text,
    status text,
    confidence double precision check (confidence is null or (confidence >= 0 and confidence <= 1)),
    source_backend text,
    source_model text,
    raw_response text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create table if not exists public.plate_detections (
    id uuid primary key default gen_random_uuid(),
    track_id uuid not null references public.vehicle_tracks(id) on delete cascade,
    evidence_id uuid references public.track_evidence(id) on delete set null,
    media_id uuid references public.media_assets(id) on delete set null,
    frame_number bigint check (frame_number is null or frame_number >= 0),
    timestamp_seconds double precision check (timestamp_seconds is null or timestamp_seconds >= 0),
    bbox jsonb,
    confidence double precision check (confidence is null or (confidence >= 0 and confidence <= 1)),
    crop_media_id uuid references public.media_assets(id) on delete set null,
    status text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create table if not exists public.plate_readings (
    id uuid primary key default gen_random_uuid(),
    plate_detection_id uuid references public.plate_detections(id) on delete set null,
    track_id uuid not null references public.vehicle_tracks(id) on delete cascade,
    raw_text text,
    normalized_text text,
    confidence double precision check (confidence is null or (confidence >= 0 and confidence <= 1)),
    status text,
    plate_colour text,
    verified boolean not null default false,
    verification_source text,
    model_name text,
    model_version text,
    raw_response text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create table if not exists public.pipeline_artifacts (
    id uuid primary key default gen_random_uuid(),
    run_id uuid not null references public.processing_runs(id) on delete cascade,
    artifact_type text not null,
    relative_path text not null,
    format text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (run_id, relative_path)
);

create table if not exists public.pipeline_errors (
    id uuid primary key default gen_random_uuid(),
    run_id uuid not null references public.processing_runs(id) on delete cascade,
    camera_id uuid references public.run_cameras(id) on delete set null,
    track_id uuid references public.vehicle_tracks(id) on delete set null,
    stage text,
    severity text,
    error_code text,
    message text,
    details jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create table if not exists public.chat_sessions (
    id uuid primary key default gen_random_uuid(),
    session_key text not null unique,
    scope_type text not null,
    title text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.chat_session_runs (
    session_id uuid not null references public.chat_sessions(id) on delete cascade,
    run_id uuid not null references public.processing_runs(id) on delete cascade,
    created_at timestamptz not null default now(),
    primary key (session_id, run_id)
);

create table if not exists public.chat_messages (
    id uuid primary key default gen_random_uuid(),
    session_id uuid not null references public.chat_sessions(id) on delete cascade,
    message_index integer not null check (message_index >= 0),
    role text not null check (role in ('system', 'user', 'assistant', 'tool')),
    content text not null,
    query_plan jsonb not null default '{}'::jsonb,
    analytics_result jsonb not null default '{}'::jsonb,
    evidence_refs jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now(),
    unique (session_id, message_index)
);

create index if not exists idx_processing_runs_started_at on public.processing_runs(started_at desc);
create index if not exists idx_processing_runs_status on public.processing_runs(status);
create index if not exists idx_run_cameras_run_enabled on public.run_cameras(run_id, enabled);
create index if not exists idx_vehicle_tracks_run_status on public.vehicle_tracks(run_id, track_status);
create index if not exists idx_vehicle_tracks_run_class on public.vehicle_tracks(run_id, vehicle_class);
create index if not exists idx_vehicle_tracks_run_colour on public.vehicle_tracks(run_id, vehicle_colour);
create index if not exists idx_vehicle_tracks_run_class_colour on public.vehicle_tracks(run_id, vehicle_class, vehicle_colour);
create index if not exists idx_vehicle_tracks_run_time_overlap on public.vehicle_tracks(run_id, first_seen_seconds, last_seen_seconds);
create index if not exists idx_vehicle_tracks_completed_search on public.vehicle_tracks(run_id, vehicle_class, vehicle_colour, first_seen_seconds, last_seen_seconds) where track_status = 'COMPLETED';
create index if not exists idx_track_observations_run_camera_frame on public.track_observations(run_id, camera_id, frame_number);
create index if not exists idx_track_evidence_track_role on public.track_evidence(track_id, evidence_role);
create index if not exists idx_track_evidence_run_role on public.track_evidence(run_id, evidence_role);
create index if not exists idx_media_assets_run_path on public.media_assets(run_id, relative_path);
create index if not exists idx_colour_predictions_track on public.colour_predictions(track_id);
create index if not exists idx_attribute_predictions_track_type on public.vehicle_attribute_predictions(track_id, attribute_type);
create index if not exists idx_plate_readings_normalized_text on public.plate_readings(normalized_text) where normalized_text is not null;
create index if not exists idx_chat_messages_session_index on public.chat_messages(session_id, message_index);

alter table public.processing_runs enable row level security;
alter table public.run_cameras enable row level security;
alter table public.vehicle_tracks enable row level security;
alter table public.track_observations enable row level security;
alter table public.track_evidence enable row level security;
alter table public.media_assets enable row level security;
alter table public.colour_predictions enable row level security;
alter table public.vehicle_attribute_predictions enable row level security;
alter table public.plate_detections enable row level security;
alter table public.plate_readings enable row level security;
alter table public.pipeline_artifacts enable row level security;
alter table public.pipeline_errors enable row level security;
alter table public.chat_sessions enable row level security;
alter table public.chat_session_runs enable row level security;
alter table public.chat_messages enable row level security;
