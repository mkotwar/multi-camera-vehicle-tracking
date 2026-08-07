# Worker / Queue Architecture

## Scope

This document describes the current active runtime path for:

- multi-camera input
- fixed-size ingestion worker pool
- shared YOLO detection
- per-camera tracking state
- evidence collection
- async Florence colour enrichment
- camera-wise outputs

Frozen and intentionally unchanged in this step:

- body type
- plate detection / OCR / colour
- make / model
- global tracking
- cross-camera association

## Old vs New Ingestion Design

### Old

```text
N cameras
  -> one reader object per camera
  -> cameras permanently assigned to ingestion workers
  -> workers push directly into one shared frame queue
```

Properties:

- fairness only existed inside each worker's local camera loop
- no per-camera bounded staging buffers
- a fast camera could occupy more of the shared queue

### New

```text
CAM_001 -|
CAM_002 -|
CAM_003 -|--> Camera Source Registry
...      |
CAM_N   -|
                |
                v
        Ingestion Task Queue
                |
                v
    Fixed Ingestion Worker Pool
          workers = 3
                |
                v
    Per-Camera Bounded Buffers
       buffer size = 2 each
                |
                v
    Fair Round-Robin Scheduler
                |
                v
         Shared Detection Queue
                |
                v
          Shared YOLO Owner
                |
                v
      Per-Camera Tracker States
                |
                v
             Evidence
                |
                v
           Colour Queue
                |
                v
       One Florence Worker
                |
                v
          Camera Results
```

## Implemented Ingestion Architecture

### Camera source registry

- File: `src/ingestion_manager.py`
- One `VideoCameraReader` is still created per enabled camera
- The readers are stored in:
  - `readers_by_camera`
  - `camera_sources`
- This preserves camera identity and camera-local read state without creating one permanent worker per camera

### Fixed worker pool

- Ingestion workers are Python threads
- Count comes from `ingestion.worker_count`
- The worker count is independent from camera count
- Workers pull camera IDs dynamically from `camera_task_queue`
- Cameras are not permanently pinned to a single worker

### Per-camera bounded buffers

- Each enabled camera owns one bounded buffer in `per_camera_buffers`
- Buffer size comes from `ingestion.per_camera_buffer_size`
- Current validation config uses `2`
- This prevents uncontrolled queue growth while keeping camera identity explicit

### Fair scheduler

- Scheduler thread drains `per_camera_buffers`
- Policy comes from `ingestion.scheduler_policy`
- Current supported policy: `round_robin`
- Empty buffers are skipped without blocking other cameras
- After a frame is scheduled downstream, the scheduler re-enqueues the next read job for that camera if allowed

## Ordering and Safety Guarantees

### Per-camera order

- Guaranteed: `YES`
- Only one read may be in flight for a given camera
- Frames remain ordered within each camera because reads are serialized per source and buffers are FIFO per camera

### Same-camera concurrent reads

- Possible: `NO`
- Guarded by per-camera state:
  - `task_enqueued`
  - `read_in_flight`
  - `completed`
  - `failed`

### Camera identity preservation

- Preserved end-to-end: `YES`
- `FramePacket` keeps:
  - `camera_id`
  - `frame_number`
  - `timestamp_seconds`
  - `captured_at`
  - `source_type`
  - frame dimensions

## EOF and Failure Isolation

### EOF behavior

- EOF is isolated per camera: `YES`
- When one file source ends, only that camera is marked complete
- Remaining cameras continue until their own EOF
- Ingestion finishes only after:
  - all enabled cameras are completed or failed
  - all per-camera buffers are drained
  - workers stop cleanly
  - scheduler stops cleanly

### Camera failure isolation

- Camera failure isolated: `YES`
- A camera read exception marks only that camera failed
- The worker survives the job-level exception and continues servicing future camera tasks
- Global stop only happens when `ingestion.stop_on_camera_error = true`

## Backpressure Policy

### Per-camera buffer full

- Policy: bounded and non-expanding
- A new read job is not scheduled while the camera buffer is full
- Metrics record:
  - `buffer_full_count`
  - `buffer_full_count_by_camera`
  - `per_camera_buffer_peak`

### Shared detection queue full

- Policy: block with timeout and retry
- Frames are not intentionally dropped in current file-validation mode
- Metrics record:
  - `queue_full_events`
  - `maximum_observed_queue_size`

### Frame-drop policy

- Current offline video validation policy: no intentional frame dropping
- Future live-stream freshness policies can be added separately without changing the current file-mode behavior

## Metrics

The ingestion layer now reports:

- `ingestion_worker_count`
- `camera_count`
- `camera_source_registry_count`
- `camera_assignment_mode = dynamic_task_queue`
- `scheduler_policy`
- `per_camera_buffer_count`
- `per_camera_buffer_size`
- `camera_read_jobs`
- `camera_read_failures`
- `frames_by_camera`
- `frames_scheduled_by_camera`
- `frames_consumed_by_camera`
- `frames_by_worker`
- `per_camera_buffer_peak`
- `buffer_full_count`
- `buffer_full_count_by_camera`
- `round_robin_cycles`
- `scheduler_skipped_empty_camera`
- `max_consecutive_frames_same_camera`

## Worker Budget

This step keeps the downstream architecture unchanged:

- ingestion workers = configurable fixed pool
- detection workers = `1`
- YOLO model instances = `1`
- tracker state count = one state per camera
- colour workers = `1`
- Florence colour model instances = `1`

This means:

- `N` cameras does **not** create `N` ingestion workers
- `N` cameras still legitimately create `N` camera states and `N` reader objects
- concurrency stays intentionally small and bounded

## Scalability Benchmark

### Test conditions

- Date: `August 7, 2026`
- Source mode: one local test video path reused across logical camera IDs with independent reader handles
- Logical camera counts: `2`, `4`, `8`, `12`
- Frames per camera: `30`
- Ingestion workers: `3`
- Detection workers: `1`
- YOLO model instances: `1`
- Colour workers: `1`
- Florence model instances: `1`

### Observed saturation point

- `2` cameras: healthy, no frame loss, fairness good, colour queue peak `30/100`
- `4` cameras: healthy, no frame loss, fairness good, colour queue peak `52/100`
- `8` cameras: near saturation, no frame loss, fairness good, colour queue peak `95/100`
- `12` cameras: saturated on colour drain, no frame loss, fairness good, colour queue peak `100/100`, block count observed

### Primary bottleneck

- Primary bottleneck: `Florence colour worker` at `8+` logical cameras
- Secondary bottleneck: `shared YOLO / detection queue`
- Evidence cache health remained stable with `evidence_cache_misses = 0` for every tested load

### Recommended next optimization

- Recommended next step: evaluate `Florence` throughput optimization before changing ingestion or tracker architecture
