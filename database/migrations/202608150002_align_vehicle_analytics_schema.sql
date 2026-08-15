-- Canonical backend <-> PostgreSQL repair migration for local PostgreSQL.
-- Additive only: targets vehicle_analytics, not public.

create schema if not exists vehicle_analytics;
create extension if not exists pgcrypto;

alter table vehicle_analytics.processing_runs add column if not exists output_directory text;
alter table vehicle_analytics.processing_runs add column if not exists project_name text;
alter table vehicle_analytics.processing_runs add column if not exists detection_backend text;
alter table vehicle_analytics.processing_runs add column if not exists tracking_backend text;
alter table vehicle_analytics.processing_runs add column if not exists enrichment_enabled boolean;
alter table vehicle_analytics.processing_runs add column if not exists processed_frames bigint;
alter table vehicle_analytics.processing_runs add column if not exists total_detections bigint;
alter table vehicle_analytics.processing_runs add column if not exists raw_yolo_detections bigint;
alter table vehicle_analytics.processing_runs add column if not exists roi_filtered_detections bigint;
alter table vehicle_analytics.processing_runs add column if not exists completed_tracks integer;
alter table vehicle_analytics.processing_runs add column if not exists discarded_tracks integer;
alter table vehicle_analytics.processing_runs add column if not exists config_path text;
alter table vehicle_analytics.processing_runs add column if not exists config_snapshot jsonb default '{}'::jsonb;
alter table vehicle_analytics.processing_runs add column if not exists summary jsonb default '{}'::jsonb;
alter table vehicle_analytics.processing_runs add column if not exists metrics jsonb default '{}'::jsonb;
alter table vehicle_analytics.processing_runs add column if not exists metadata jsonb default '{}'::jsonb;

alter table vehicle_analytics.run_cameras add column if not exists source_uri text;
alter table vehicle_analytics.run_cameras add column if not exists frames_processed bigint;
alter table vehicle_analytics.run_cameras add column if not exists detections_count bigint;
alter table vehicle_analytics.run_cameras add column if not exists frame_width integer;
alter table vehicle_analytics.run_cameras add column if not exists frame_height integer;
alter table vehicle_analytics.run_cameras add column if not exists metadata jsonb default '{}'::jsonb;

update vehicle_analytics.run_cameras
set source_uri = coalesce(source_uri, source)
where source_uri is null;

update vehicle_analytics.run_cameras
set frames_processed = coalesce(frames_processed, processed_frames)
where frames_processed is null;

alter table vehicle_analytics.vehicle_tracks add column if not exists camera_key text;
alter table vehicle_analytics.vehicle_tracks add column if not exists vehicle_class_confidence double precision;
alter table vehicle_analytics.vehicle_tracks add column if not exists vehicle_colour_status text;
alter table vehicle_analytics.vehicle_tracks add column if not exists body_type text;
alter table vehicle_analytics.vehicle_tracks add column if not exists body_type_status text;
alter table vehicle_analytics.vehicle_tracks add column if not exists plate_text text;
alter table vehicle_analytics.vehicle_tracks add column if not exists plate_detected boolean;
alter table vehicle_analytics.vehicle_tracks add column if not exists plate_colour text;
alter table vehicle_analytics.vehicle_tracks add column if not exists registration_category text;
alter table vehicle_analytics.vehicle_tracks add column if not exists final_class_reason text;
alter table vehicle_analytics.vehicle_tracks add column if not exists completion_reason text;
alter table vehicle_analytics.vehicle_tracks add column if not exists class_counts jsonb default '{}'::jsonb;
alter table vehicle_analytics.vehicle_tracks add column if not exists class_confidence_sums jsonb default '{}'::jsonb;
alter table vehicle_analytics.vehicle_tracks add column if not exists enrichment_summary jsonb default '{}'::jsonb;
alter table vehicle_analytics.vehicle_tracks add column if not exists evidence_record_count integer;
alter table vehicle_analytics.vehicle_tracks add column if not exists raw_track jsonb default '{}'::jsonb;

alter table vehicle_analytics.vehicle_tracks
    alter column native_tracker_id type text using native_tracker_id::text;

alter table vehicle_analytics.vehicle_tracks
    drop constraint if exists vehicle_tracks_run_id_camera_id_tracker_namespace_native_tr_key;

update vehicle_analytics.vehicle_tracks t
set camera_key = c.camera_key
from vehicle_analytics.run_cameras c
where t.camera_id = c.id
  and t.camera_key is null;

alter table vehicle_analytics.media_assets add column if not exists storage_provider text default 'local';
alter table vehicle_analytics.media_assets add column if not exists bucket text;
alter table vehicle_analytics.media_assets add column if not exists object_key text;
alter table vehicle_analytics.media_assets add column if not exists metadata jsonb default '{}'::jsonb;

alter table vehicle_analytics.track_observations add column if not exists bbox_x1 double precision;
alter table vehicle_analytics.track_observations add column if not exists bbox_y1 double precision;
alter table vehicle_analytics.track_observations add column if not exists bbox_x2 double precision;
alter table vehicle_analytics.track_observations add column if not exists bbox_y2 double precision;
alter table vehicle_analytics.track_observations add column if not exists detection_confidence double precision;
alter table vehicle_analytics.track_observations add column if not exists tracker_namespace text;
alter table vehicle_analytics.track_observations add column if not exists native_tracker_id text;
alter table vehicle_analytics.track_observations add column if not exists metadata jsonb default '{}'::jsonb;

update vehicle_analytics.track_observations
set bbox_x1 = coalesce(bbox_x1, x1),
    bbox_y1 = coalesce(bbox_y1, y1),
    bbox_x2 = coalesce(bbox_x2, x2),
    bbox_y2 = coalesce(bbox_y2, y2),
    detection_confidence = coalesce(detection_confidence, confidence)
where bbox_x1 is null
   or bbox_y1 is null
   or bbox_x2 is null
   or bbox_y2 is null
   or detection_confidence is null;

alter table vehicle_analytics.track_evidence add column if not exists run_id uuid references vehicle_analytics.processing_runs(id) on delete cascade;
alter table vehicle_analytics.track_evidence add column if not exists camera_id uuid references vehicle_analytics.run_cameras(id) on delete cascade;
alter table vehicle_analytics.track_evidence add column if not exists crop_media_id uuid references vehicle_analytics.media_assets(id) on delete set null;
alter table vehicle_analytics.track_evidence add column if not exists source_frame_media_id uuid references vehicle_analytics.media_assets(id) on delete set null;
alter table vehicle_analytics.track_evidence add column if not exists annotated_frame_media_id uuid references vehicle_analytics.media_assets(id) on delete set null;
alter table vehicle_analytics.track_evidence add column if not exists original_bbox_x1 double precision;
alter table vehicle_analytics.track_evidence add column if not exists original_bbox_y1 double precision;
alter table vehicle_analytics.track_evidence add column if not exists original_bbox_x2 double precision;
alter table vehicle_analytics.track_evidence add column if not exists original_bbox_y2 double precision;
alter table vehicle_analytics.track_evidence add column if not exists expanded_crop_bbox_x1 double precision;
alter table vehicle_analytics.track_evidence add column if not exists expanded_crop_bbox_y1 double precision;
alter table vehicle_analytics.track_evidence add column if not exists expanded_crop_bbox_x2 double precision;
alter table vehicle_analytics.track_evidence add column if not exists expanded_crop_bbox_y2 double precision;
alter table vehicle_analytics.track_evidence add column if not exists detection_confidence double precision;
alter table vehicle_analytics.track_evidence add column if not exists quality_score double precision;
alter table vehicle_analytics.track_evidence add column if not exists brightness_score double precision;
alter table vehicle_analytics.track_evidence add column if not exists crop_width integer;
alter table vehicle_analytics.track_evidence add column if not exists crop_height integer;
alter table vehicle_analytics.track_evidence add column if not exists resolution_tier text;
alter table vehicle_analytics.track_evidence add column if not exists selected_for_colour boolean;
alter table vehicle_analytics.track_evidence add column if not exists selected_for_body_type boolean;
alter table vehicle_analytics.track_evidence add column if not exists evidence_source text;
alter table vehicle_analytics.track_evidence add column if not exists candidate_rank integer;
alter table vehicle_analytics.track_evidence add column if not exists metadata jsonb default '{}'::jsonb;

update vehicle_analytics.track_evidence e
set run_id = t.run_id,
    camera_id = t.camera_id,
    crop_media_id = coalesce(e.crop_media_id, e.media_asset_id),
    quality_score = coalesce(e.quality_score, e.best_overall_score)
from vehicle_analytics.vehicle_tracks t
where e.track_id = t.id
  and (e.run_id is null or e.camera_id is null or e.crop_media_id is null or e.quality_score is null);

alter table vehicle_analytics.colour_predictions add column if not exists media_asset_id uuid references vehicle_analytics.media_assets(id) on delete set null;
alter table vehicle_analytics.colour_predictions add column if not exists media_id uuid references vehicle_analytics.media_assets(id) on delete set null;
alter table vehicle_analytics.colour_predictions add column if not exists normalized_colour text;
alter table vehicle_analytics.colour_predictions add column if not exists status text;
alter table vehicle_analytics.colour_predictions add column if not exists model_name text;
alter table vehicle_analytics.colour_predictions add column if not exists prompt text;
alter table vehicle_analytics.colour_predictions add column if not exists raw_response text;
alter table vehicle_analytics.colour_predictions add column if not exists inference_time_ms double precision;

update vehicle_analytics.colour_predictions
set normalized_colour = coalesce(normalized_colour, predicted_colour),
    model_name = coalesce(model_name, source_model)
where normalized_colour is null or model_name is null;

alter table vehicle_analytics.vehicle_attribute_predictions add column if not exists media_asset_id uuid references vehicle_analytics.media_assets(id) on delete set null;
alter table vehicle_analytics.vehicle_attribute_predictions add column if not exists media_id uuid references vehicle_analytics.media_assets(id) on delete set null;
alter table vehicle_analytics.vehicle_attribute_predictions add column if not exists attribute_value text;
alter table vehicle_analytics.vehicle_attribute_predictions add column if not exists label text;
alter table vehicle_analytics.vehicle_attribute_predictions add column if not exists normalized_label text;
alter table vehicle_analytics.vehicle_attribute_predictions add column if not exists status text;
alter table vehicle_analytics.vehicle_attribute_predictions add column if not exists source_backend text;
alter table vehicle_analytics.vehicle_attribute_predictions add column if not exists raw_response text;
alter table vehicle_analytics.vehicle_attribute_predictions add column if not exists evidence_frame_number bigint;
alter table vehicle_analytics.vehicle_attribute_predictions add column if not exists evidence_timestamp_seconds double precision;

update vehicle_analytics.vehicle_attribute_predictions
set attribute_value = coalesce(attribute_value, label),
    label = coalesce(label, attribute_value),
    normalized_label = coalesce(normalized_label, upper(coalesce(attribute_value, label)))
where attribute_value is null or label is null or normalized_label is null;

alter table vehicle_analytics.plate_detections add column if not exists media_id uuid references vehicle_analytics.media_assets(id) on delete set null;
alter table vehicle_analytics.plate_detections add column if not exists bbox jsonb;
alter table vehicle_analytics.plate_detections add column if not exists confidence double precision;
alter table vehicle_analytics.plate_detections add column if not exists crop_media_id uuid references vehicle_analytics.media_assets(id) on delete set null;
alter table vehicle_analytics.plate_detections add column if not exists status text;

update vehicle_analytics.plate_detections
set confidence = coalesce(confidence, detection_confidence),
    media_id = coalesce(media_id, media_asset_id),
    crop_media_id = coalesce(crop_media_id, media_asset_id)
where confidence is null or media_id is null or crop_media_id is null;

alter table vehicle_analytics.plate_readings add column if not exists track_id uuid references vehicle_analytics.vehicle_tracks(id) on delete cascade;
alter table vehicle_analytics.plate_readings add column if not exists status text;
alter table vehicle_analytics.plate_readings add column if not exists plate_colour text;
alter table vehicle_analytics.plate_readings add column if not exists registration_category text;
alter table vehicle_analytics.plate_readings add column if not exists model_name text;
alter table vehicle_analytics.plate_readings add column if not exists raw_response text;

update vehicle_analytics.plate_readings r
set track_id = d.track_id,
    model_name = coalesce(r.model_name, r.source_model)
from vehicle_analytics.plate_detections d
where r.plate_detection_id = d.id
  and (r.track_id is null or r.model_name is null);

alter table vehicle_analytics.pipeline_artifacts add column if not exists format text;

alter table vehicle_analytics.pipeline_errors add column if not exists severity text;
alter table vehicle_analytics.pipeline_errors add column if not exists error_code text;
alter table vehicle_analytics.pipeline_errors add column if not exists message text;
alter table vehicle_analytics.pipeline_errors add column if not exists details jsonb default '{}'::jsonb;

update vehicle_analytics.pipeline_errors
set message = coalesce(message, error_message),
    details = coalesce(details, metadata, '{}'::jsonb)
where message is null or details is null;

create table if not exists vehicle_analytics.physical_vehicles (
    id uuid primary key,
    run_id uuid not null references vehicle_analytics.processing_runs(id) on delete cascade,
    vehicle_key text not null,
    vehicle_class text,
    vehicle_colour text,
    first_timestamp_seconds double precision,
    last_timestamp_seconds double precision,
    identity_confidence double precision,
    identity_method text,
    identity_status text,
    consensus_plate_text text,
    plate_confidence double precision,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (run_id, vehicle_key)
);

create table if not exists vehicle_analytics.physical_vehicle_tracks (
    physical_vehicle_id uuid not null references vehicle_analytics.physical_vehicles(id) on delete cascade,
    vehicle_track_id uuid not null references vehicle_analytics.vehicle_tracks(id) on delete cascade,
    association_score double precision,
    association_method text,
    association_reason text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    primary key (physical_vehicle_id, vehicle_track_id),
    unique (vehicle_track_id)
);

create table if not exists vehicle_analytics.identity_decisions (
    id uuid primary key,
    run_id uuid not null references vehicle_analytics.processing_runs(id) on delete cascade,
    source_track_id uuid references vehicle_analytics.vehicle_tracks(id) on delete set null,
    target_track_id uuid references vehicle_analytics.vehicle_tracks(id) on delete set null,
    decision text not null,
    final_score double precision,
    plate_score double precision,
    spatial_score double precision,
    temporal_score double precision,
    motion_score double precision,
    appearance_score double precision,
    colour_score double precision,
    reason text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists idx_processing_runs_started_at on vehicle_analytics.processing_runs(started_at desc);
create index if not exists idx_processing_runs_status on vehicle_analytics.processing_runs(status);
create index if not exists idx_run_cameras_run_enabled on vehicle_analytics.run_cameras(run_id, enabled);
create index if not exists idx_vehicle_tracks_run_status on vehicle_analytics.vehicle_tracks(run_id, track_status);
create index if not exists idx_vehicle_tracks_run_class on vehicle_analytics.vehicle_tracks(run_id, vehicle_class);
create index if not exists idx_vehicle_tracks_run_colour on vehicle_analytics.vehicle_tracks(run_id, vehicle_colour);
create index if not exists idx_vehicle_tracks_run_class_colour on vehicle_analytics.vehicle_tracks(run_id, vehicle_class, vehicle_colour);
create index if not exists idx_vehicle_tracks_run_time_overlap on vehicle_analytics.vehicle_tracks(run_id, first_seen_seconds, last_seen_seconds);
create index if not exists idx_vehicle_tracks_completed_search on vehicle_analytics.vehicle_tracks(run_id, vehicle_class, vehicle_colour, first_seen_seconds, last_seen_seconds) where track_status = 'COMPLETED';
create index if not exists idx_track_observations_run_camera_frame on vehicle_analytics.track_observations(run_id, camera_id, frame_number);
create index if not exists idx_track_evidence_track_role on vehicle_analytics.track_evidence(track_id, evidence_role);
create index if not exists idx_track_evidence_run_role on vehicle_analytics.track_evidence(run_id, evidence_role);
create index if not exists idx_media_assets_run_path on vehicle_analytics.media_assets(run_id, relative_path);
create index if not exists idx_colour_predictions_track on vehicle_analytics.colour_predictions(track_id);
create index if not exists idx_attribute_predictions_track_type on vehicle_analytics.vehicle_attribute_predictions(track_id, attribute_type);
create index if not exists idx_plate_readings_normalized_text on vehicle_analytics.plate_readings(normalized_text) where normalized_text is not null;
create index if not exists idx_physical_vehicles_run on vehicle_analytics.physical_vehicles(run_id);
create index if not exists idx_physical_vehicles_plate on vehicle_analytics.physical_vehicles(consensus_plate_text) where consensus_plate_text is not null;
create index if not exists idx_physical_vehicle_tracks_track on vehicle_analytics.physical_vehicle_tracks(vehicle_track_id);
create index if not exists idx_identity_decisions_run on vehicle_analytics.identity_decisions(run_id);
create index if not exists idx_identity_decisions_decision on vehicle_analytics.identity_decisions(run_id, decision);
