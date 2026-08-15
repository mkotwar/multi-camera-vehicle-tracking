create table if not exists public.physical_vehicles (
    id uuid primary key,
    run_id uuid not null references public.processing_runs(id) on delete cascade,
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

create table if not exists public.physical_vehicle_tracks (
    physical_vehicle_id uuid not null references public.physical_vehicles(id) on delete cascade,
    vehicle_track_id uuid not null references public.vehicle_tracks(id) on delete cascade,
    association_score double precision,
    association_method text,
    association_reason text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    primary key (physical_vehicle_id, vehicle_track_id),
    unique (vehicle_track_id)
);

create table if not exists public.identity_decisions (
    id uuid primary key,
    run_id uuid not null references public.processing_runs(id) on delete cascade,
    source_track_id uuid references public.vehicle_tracks(id) on delete set null,
    target_track_id uuid references public.vehicle_tracks(id) on delete set null,
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

create index if not exists idx_physical_vehicles_run on public.physical_vehicles(run_id);
create index if not exists idx_physical_vehicles_plate on public.physical_vehicles(consensus_plate_text) where consensus_plate_text is not null;
create index if not exists idx_physical_vehicle_tracks_track on public.physical_vehicle_tracks(vehicle_track_id);
create index if not exists idx_identity_decisions_run on public.identity_decisions(run_id);
create index if not exists idx_identity_decisions_decision on public.identity_decisions(run_id, decision);

alter table public.physical_vehicles enable row level security;
alter table public.physical_vehicle_tracks enable row level security;
alter table public.identity_decisions enable row level security;
