# OLD_REPO_REFERENCE

## 1. Purpose

This document is a rebuild reference for the old repository at:

`F:\vinfo\Final_vedio_Ai_system`

It identifies the useful working concepts and files for rebuilding a clean local pipeline with this target flow:

`video/camera input -> frame reading -> YOLO vehicle detection -> standard ByteTrack tracking -> simple track lifecycle -> final vehicle class -> evidence selection -> JSON output`

This document does **not** recommend copying the old repository structure directly.

Old repository code modified: `NO`

## 2. Old repository location

- Repository root: `F:\vinfo\Final_vedio_Ai_system`
- Main pipeline folder studied: `F:\vinfo\Final_vedio_Ai_system\tests\td_case2\multicamera_vehicle_tracking_pipeline`

## 3. Main pipeline location

Confirmed main pipeline location:

`F:\vinfo\Final_vedio_Ai_system\tests\td_case2\multicamera_vehicle_tracking_pipeline`

Top-level folders that actually exist there:

- `api/`: read-only FastAPI layer for browsing persisted runs/tracks/media
- `config/`: YAML config files for cameras, detection, tracking, persistence, evidence, workers, ANPR, Florence, colour, global matching
- `cross_camera/`: global-vehicle and cross-camera matching logic
- `data/`: local test videos and sample images
- `database/`: schema, migrations, DB config, repository layer
- `detection/`: YOLO detection config, models, filtering, detector runtime wrapper
- `docs/`: design notes and mapping docs
- `enrichment/`: ANPR, vehicle colour, body type, plate utilities, evidence selector helpers
- `evidence/`: track crop selection, packaging, artifact writing
- `frontend/`: React frontend
- `ingestion/`: camera config loading, file video source, multi-camera reader, frame packet
- `models/`: model path resolution and Florence/plate runtime factories
- `orchestration/`: sequential input/detection/tracking orchestration and worker orchestration
- `persistence/`: persistence backends, analytics client, media persistence, mapping
- `scripts/`: runnable validation, comparison, reporting, and experiment scripts
- `tests/`: Python unit/integration tests
- `tracking/`: ByteTrack integration, per-camera routing, lifecycle, class calculation
- `workers/`: threaded worker pipeline

## 4. Current high-level flow

Clean core flow that is already present in the old repo:

`CameraSource -> MultiCameraReader -> SharedVehicleDetector -> CameraDetectionRouter -> CameraTracker -> LocalTrackLifecycle -> optional TrackEvidenceCollector -> optional persistence`

Main files:

- Input orchestration: `orchestration/multi_camera_orchestrator.py`
- Detection orchestration: `orchestration/multicamera_detection_orchestrator.py`
- Tracking orchestration: `orchestration/multicamera_tracking_orchestrator.py`
- Reader: `ingestion/multi_camera_reader.py`
- Video open/read: `ingestion/camera_source.py`
- Detector: `detection/vehicle_detector.py`
- Per-camera tracker routing: `tracking/camera_detection_router.py`
- Tracker instance factory: `tracking/tracker_factory.py`
- Per-camera tracker update: `tracking/camera_tracker.py`
- Lifecycle/state machine: `tracking/track_lifecycle.py`
- Final class calculation: `tracking/class_stabilization.py`
- Evidence selection: `evidence/track_evidence_collector.py`
- Persistence backend selection: `persistence/persistence_backend_factory.py`

Important separation:

- `behavior_mode: standard_bytetrack` in `config/tracking.yaml` is the cleaner rebuild target.
- `behavior_mode: experimental_custom` enables extra logic that should not be migrated first.

## 5. Pipeline entry points

### Main sequential tracking validation

- File: `tests\td_case2\multicamera_vehicle_tracking_pipeline\scripts\validate_multicamera_tracking.py`
- Purpose: full sequential tracking pipeline with optional evidence and persistence
- Input: camera config, detection config, tracking config, optional persistence/evidence configs
- Output: JSON report in `debug_runs\multicamera_vehicle_tracking_pipeline\tracking_validation_<timestamp>\report.json`
- Type: validation
- Copy conceptually to new repo: `YES`

### Single-camera / input validation

- File: `tests\td_case2\multicamera_vehicle_tracking_pipeline\scripts\validate_multicamera_input.py`
- Purpose: validate frame reading only
- Input: `--config`, mode, frame limit
- Output: JSON report under `debug_runs\multicamera_vehicle_tracking_pipeline\input_validation_<timestamp>\report.json`
- Type: validation
- Copy conceptually to new repo: `YES`

### Detection validation

- File: `tests\td_case2\multicamera_vehicle_tracking_pipeline\scripts\validate_multicamera_detection.py`
- Purpose: validate shared vehicle detection without tracking
- Input: camera config, detection config
- Output: JSON report under `debug_runs\multicamera_vehicle_tracking_pipeline\detection_validation_<timestamp>\report.json`
- Type: validation
- Copy conceptually to new repo: `YES`

### Worker-based pipeline

- File: `tests\td_case2\multicamera_vehicle_tracking_pipeline\scripts\validate_worker_multicamera_tracking.py`
- Purpose: threaded camera/detection/tracking/persistence/evidence/enrichment pipeline
- Input: camera, detection, tracking, worker config and optional persistence/evidence/enrichment configs
- Output: JSON report under `debug_runs\multicamera_vehicle_tracking_pipeline\worker_tracking_validation_<timestamp>\report.json`
- Type: validation
- Copy conceptually to new repo: `NO`

### Persistence path

- File: `tests\td_case2\multicamera_vehicle_tracking_pipeline\scripts\validate_multicamera_tracking.py`
- Purpose: persistence is invoked from the main sequential tracking validation through `--dry-run-persistence` or `--persist-to-supabase`
- Input: completed tracks from lifecycle
- Output: analytics DB writes or dry-run metrics
- Type: validation / production-adjacent
- Copy conceptually to new repo: `LATER`

### Evidence generation path

- File: `tests\td_case2\multicamera_vehicle_tracking_pipeline\scripts\validate_multicamera_tracking.py`
- Purpose: evidence is generated inside the main tracking orchestrator when `evidence.yaml` is enabled
- Input: detection packets + track observations
- Output: artifacts under `artifacts\<RUN_ID>\<CAMERA>\track_<id>\...`
- Type: validation / production-adjacent
- Copy conceptually to new repo: `YES, simplified`

### Class-confidence experiments

- File: `tests\td_case2\multicamera_vehicle_tracking_pipeline\scripts\run_class_confidence_experiment.py`
- Purpose: compare per-class thresholds and downstream class behavior
- Input: camera/video window plus multiple threshold configurations
- Output: debug comparison JSON/Markdown
- Type: experiment
- Copy conceptually to new repo: `NO`

### Standard ByteTrack comparison

- File: `tests\td_case2\multicamera_vehicle_tracking_pipeline\scripts\run_standard_bytetrack_comparison.py`
- Purpose: compare `experimental_custom` vs `standard_bytetrack`
- Input: camera/video window and tracking overrides
- Output: comparison JSON/Markdown under `debug_runs\multicamera_vehicle_tracking_pipeline\standard_bytetrack_comparison`
- Type: experiment / validation
- Copy conceptually to new repo: `YES, as a small regression test later`

### Other scripts found but not needed for Milestone 1

- `run_tracking_confidence_experiment.py`: experiment
- `run_final_class_parameter_matrix.py`: experiment
- `run_class_consistency_experiment.py`: experiment
- `diagnose_track_identity_switch.py`: debugging
- `build_global_vehicle_objects.py`: later cross-camera stage
- `validate_anpr_on_existing_run.py`: later enrichment stage

## 6. Camera/video ingestion

Primary files:

- `ingestion/camera_config.py`
- `ingestion/camera_source.py`
- `ingestion/multi_camera_reader.py`
- `ingestion/frame_packet.py`

Verified behavior:

- Camera configuration file: `tests\td_case2\multicamera_vehicle_tracking_pipeline\config\cameras.yaml`
- Camera source path format: either `source_path` or nested `source.path`
- Video opening library: `cv2.VideoCapture`
- FPS source: `cv2.CAP_PROP_FPS`
- Frame count source: `cv2.CAP_PROP_FRAME_COUNT`
- Frame number generation: local counter starting at `0`, incremented per successful read
- Timestamp seconds generation: `frame_number / source_fps`
- Camera timestamp generation: `start_time + timedelta(seconds=video_time_seconds)` if `start_time` exists; sequential orchestrators also apply run start time for file-backed cameras when missing
- Camera ID attachment: `FramePacket.camera_code`
- End-of-video handling: `read()` failure marks end-of-stream and returns `None`
- Multi-camera reading: `MultiCameraReader`
- Reader modes: `sequential` or `round_robin`
- Default orchestration mode in validation scripts: `round_robin`
- Worker mode: separate camera reader workers, not `MultiCameraReader`

Important path issue:

- `config/cameras.yaml` currently contains absolute Windows paths under `C:\Mukul K\vinfo1\video-search-engine\...`
- Those paths do **not** exist in the current inspected environment
- Therefore current `cameras.yaml` is stale for this machine and should be treated as `UNVERIFIED` for direct execution

Current `cameras.yaml` entries:

- `CAM_001 -> C:/Mukul K/vinfo1/video-search-engine/tests/td_case2/multicamera_vehicle_tracking_pipeline/data/1test.mp4`
- `CAM_002 -> C:/Mukul K/vinfo1/video-search-engine/tests/td_case2/multicamera_vehicle_tracking_pipeline/data/2test.mp4`
- `CAM_003 -> C:/Mukul K/vinfo1/video-search-engine/tests/td_case2/multicamera_vehicle_tracking_pipeline/data/3test .mp4`

Current local data files actually present:

- `tests\td_case2\multicamera_vehicle_tracking_pipeline\data\1test.mp4`
- `tests\td_case2\multicamera_vehicle_tracking_pipeline\data\2test.mp4`
- `tests\td_case2\multicamera_vehicle_tracking_pipeline\data\1ANPR1-D.mp4`
- `tests\td_case2\multicamera_vehicle_tracking_pipeline\data\2ANPR1-D.mp4`

Simplified I/O example:

```text
Input:
camera_code + video_path

Output:
FramePacket
- camera_code
- frame_number
- timestamp_seconds
- frame
```

## 7. YOLO vehicle detection

Primary files:

- `detection/vehicle_detector.py`
- `detection/detection_config.py`
- `detection/detection_models.py`

Verified detector implementation:

- Exact detector file: `tests\td_case2\multicamera_vehicle_tracking_pipeline\detection\vehicle_detector.py`
- Model loading class: `ultralytics.YOLO`
- Shared detector instance across cameras: `YES`
- Shared detector created in sequential orchestrator: `SharedVehicleDetector(self.detection_config)`
- Shared detector created in worker orchestrator: `SharedVehicleDetector(self.detection_config)`

Verified configuration from `config/detection.yaml`:

- `model_path: models/vehicle_detection/best_old.pt`
- `fallback_model_path: yolov8n.pt`
- `allow_fallback: true`
- `device: auto`
- `iou_threshold: 0.45`
- `image_size: 640`
- Allowed classes:
  - `3wheeler`
  - `car`
  - `bus`
  - `truck`
  - `motorcycle`
- Class confidence thresholds:
  - `3wheeler: 0.65`
  - `car: 0.70`
  - `bus: 0.75`
  - `truck: 0.75`
  - `motorcycle: 0.25`

Effective runtime behavior:

- Input image format: raw OpenCV frame array from `FramePacket.frame` (`numpy` image, BGR from OpenCV)
- YOLO predict call location: `SharedVehicleDetector.detect()`
- Device selection:
  - requested: `auto`
  - resolved: `cuda:0` if `torch.cuda.is_available()` else `cpu`
- Inference confidence passed to YOLO: minimum configured class threshold
  - effective inference floor here: `0.25`
- NMS IoU threshold: `0.45`
- Image size: `640`
- Output bbox format: `xyxy`
- Detection filtering happens in `_convert_predictions()`
- Class-specific thresholds exist: `YES`
- Class name normalization uses `normalize_runtime_vehicle_class()`
- Normalized class-id mapping used by tracking:
  - `3wheeler -> 0`
  - `bus -> 1`
  - `car -> 2`
  - `motorcycle -> 3`
  - `truck -> 4`

Important model-path issue:

- Configured primary path `models/vehicle_detection/best_old.pt` does **not** exist in this repository root
- Existing similar weight file found:
  - `F:\vinfo\Final_vedio_Ai_system\object\vehical_detection\best_old.pt`
- Therefore the configured primary path is `UNVERIFIED / likely stale`
- Fallback file exists at repository root:
  - `F:\vinfo\Final_vedio_Ai_system\yolov8n.pt`

Ultralytics version:

- `requirements.txt` proves `ultralytics>=8.1.0`
- Exact installed version during the original working run: `UNVERIFIED`

Simple output example:

```text
Detection
- bbox_xyxy
- confidence
- class_id
- class_name
```

## 8. ByteTrack tracking

Primary files:

- `tracking/tracker_factory.py`
- `tracking/camera_tracker.py`
- `tracking/camera_detection_router.py`
- `tracking/supervision_conversion.py`
- `tracking/tracking_config.py`

Verified active backend from `config/tracking.yaml`:

- `behavior_mode: standard_bytetrack`
- `backend: supervision_bytetrack`

Tracker creation:

- Exact file: `tracking/tracker_factory.py`
- Active implementation: `supervision.ByteTrack`
- Tracker update call: `tracker.update_with_detections(detections)`
- Detection conversion for tracker: `tracking/supervision_conversion.py -> to_supervision_detections()`

Per-camera isolation:

- One tracker instance per camera: `YES`
- Cache location: `TrackerFactory._trackers[camera_code]`
- Camera router cache: `CameraDetectionRouter._camera_trackers`
- Tracker reset behavior: `TrackerFactory.reset()` clears cache
- Flush behavior: lifecycle flush occurs through `CameraTracker.flush()` / `CameraDetectionRouter.flush_all()`

Verified values from `config/tracking.yaml`:

- activation threshold: `0.15`
- matching threshold: `0.80`
- lost-track buffer: `30`
- frame rate: `20`
- minimum consecutive frames: `1`
- maximum lost frames: `30`
- `track_high_thresh: 0.15`
- `track_low_thresh: 0.10`
- `new_track_thresh: 0.30`
- `match_thresh: 0.80`
- `track_buffer: 30`
- `min_confirmed_observations: 3`
- `max_lost_frames: 30`

ID separation:

- Native ByteTrack ID:
  - `tracker_id` returned by Supervision ByteTrack
  - stored in `TrackObservation.native_tracker_id`
- Custom logical track ID:
  - `local_track_id`
  - in `standard_bytetrack` mode this stays equal to native ID
  - in experimental mode it can diverge because of splitting/relinking
- Global vehicle ID:
  - later cross-camera/global object layers
  - not required for first rebuild

Recommendation for clean rebuild:

Use:

`camera_code + native ByteTrack tracker_id`

Do **not** introduce logical split IDs or global IDs in Milestone 1.

## 9. Track lifecycle

Primary file:

- `tracking/track_lifecycle.py`

Verified basic responsibilities:

- create track: new `_TrackState` is created when a new tracker ID appears
- update track: observation appended, counts/confidence/class history updated
- confirm track: state becomes `active` when observation count reaches `min_confirmed_observations`
- temporarily lose track: `lost_frame_count` increments when a visible tracker ID disappears for a frame
- complete track: state becomes `completed` after lost-frame timeout if observation count is sufficient
- discard insufficient track: state becomes `discarded` after lost-frame timeout or flush if observation count is below `min_confirmed_observations`
- flush remaining tracks: `flush_camera()` / `flush_all()` finalize open tracks at end of run

Verified track states:

- `tentative`
- `active`
- `temporarily_lost`
- `completed`
- `discarded`

Custom / experimental behavior also present in the same file:

- fragment linking: `useful later`, `avoid in first rebuild`
- identity continuity checks: `useful later`, `avoid in first rebuild`
- class-conflict splitting: `experimental`, `avoid in first rebuild`
- logical track ID allocation beyond native ID: `experimental`, `avoid in first rebuild`
- relinking / reactivation metadata: `useful later`, `avoid in first rebuild`
- mixed-identity diagnostics: `useful later`, `avoid in first rebuild`

Rebuild guidance:

- Keep only create/update/confirm/lost/complete/discard
- Do not carry over conflict-split, fragment-relink, or identity-continuity code in Milestone 1

## 10. Final class calculation

Primary file:

- `tracking/class_stabilization.py`

Important mode split:

- In `standard_bytetrack` mode, `_build_standard_class_diagnostics()` is the relevant logic
- In `experimental_custom` mode, much more complicated locking/conflict/mixed-identity logic is enabled

Verified clean-mode behavior:

- Raw per-frame class source: each `TrackObservation.class_name`
- Counts accumulated in `class_counts`
- Confidence sums accumulated in `class_scores`
- Winner chosen from ranked class counts, with confidence sums retained as diagnostics
- Minimum observations threshold: from `tracking.track_class.minimum_observations`
- Minimum winner ratio threshold: from `tracking.track_class.minimum_winner_ratio`
- Configured values in current tracking config:
  - `minimum_observations: 3`
  - `minimum_winner_ratio: 0.50`
- UNKNOWN behavior:
  - no observations -> unknown / no class observations
  - too few observations -> insufficient observations
  - weak winning ratio -> ambiguous / no clear winner
- Current custom weighting/bonuses:
  - present in experimental mode only
- Count-vs-score disagreement logic:
  - present in experimental mode only
- Class lock/unlock logic:
  - present in experimental mode only

Minimum clean implementation recommended:

```text
for every raw class:
    confidence_sum[class] += observation confidence

winner = class with highest confidence sum

if observations are insufficient or winner ratio is weak:
    final_class = UNKNOWN
else:
    final_class = winner
```

## 11. Evidence collection

Primary files:

- `evidence/track_evidence_collector.py`
- `evidence/evidence_config.py`
- `config/evidence.yaml`

Verified current evidence roles:

- `first`
- `middle`
- `last`
- `highest_confidence`
- `largest`
- `sharpest`
- `best_overall`

Verified config values:

- enabled: `true`
- output root: `artifacts`
- min crop size: `40 x 40`
- minimum detection confidence: `0.20`
- padding:
  - `bbox_padding_ratio: 0.05`
  - `padding_ratio_x: 0.08`
  - `padding_ratio_y: 0.08`
  - `minimum_padding_pixels: 8`
- annotate all observations: `false`
- jpeg quality: `90`

Verified behavior:

- Crop selection exact file: `evidence/track_evidence_collector.py`
- Output directory structure:
  - `artifacts\<RUN_ID>\<CAMERA_CODE>\track_<id>\<track_uuid_sanitized>\vehicle\`
  - `...full_frames\`
  - `...annotated_frames\`
  - `...evidence_manifest.json`
- Crop clipping to frame: enabled
- Full-frame annotation format:
  - bounding boxes on selected track
  - label includes camera, local track, raw class, final class, confidence, frame, timestamp
- RAW class vs FINAL class labels:
  - both are explicitly written into the annotated label

Recommended minimum evidence behavior for new repo:

```text
best overall crop
first frame
last frame
annotated full frame
```

## 12. Persistence and database

Primary files:

- `persistence/persistence_backend_factory.py`
- `persistence/tracking_persistence_service.py`
- `persistence/analytics_persistence_service.py`
- `config/persistence.yaml`

Verified persistence entry point:

- Sequential: `orchestration/multicamera_tracking_orchestrator.py`
- Worker: `orchestration/worker_multicamera_tracking_orchestrator.py`

Verified current config values in `config/persistence.yaml`:

- backend: `dry_run`
- enabled: `true`
- write completed tracks only: `true`
- include discarded tracks: `false`
- observation mode: `all`
- observation batch size: `250`
- dry run: `true`
- persist track media: `true`
- track media roles:
  - `BEST_OVERALL`

Backend split:

- `disabled`: no persistence
- `dry_run`: analytics persistence service without DB writes
- `analytics_supabase`: analytics schema DB writes
- `old_public`: older repository-based path

Current tables / concepts used:

- camera metadata sync: yes
- track table: yes
- observation table: yes
- media/evidence persistence: yes in analytics path
- JSON reports without DB: yes, orchestrator reports already support this

Schema files found:

- `database\supabase\analytics_full_schema.sql`
- `database\migrations\001_extensions_and_schema.sql` through `030_track_media_full_frame_types.sql`
- `database\migrations\simplified_schema.sql`

Recommendation split:

- Required for initial local rebuild:
  - JSON output only
  - completed-track summary
  - optional evidence file paths
- Required later for database integration:
  - camera sync
  - track row insert
  - observation row insert
  - media persistence
- Not required initially:
  - analytics schema integration
  - old public repository backend
  - cross-camera tables
  - ANPR / colour persistence tables

## 13. Worker architecture

Primary files:

- `workers/worker_config.py`
- `workers/worker_supervisor.py`
- `workers/camera_reader_worker.py`
- `workers/detection_worker.py`
- `workers/tracking_worker.py`
- `workers/persistence_worker.py`
- `workers/vehicle_colour_worker.py`
- `workers/anpr_worker.py`
- `orchestration/worker_multicamera_tracking_orchestrator.py`

Verified queues/config:

- frame queue size: `20`
- detection queue size: `20`
- completed track queue size: `20`
- error queue size: `20`
- vehicle colour queue size: `20`
- ANPR queue size: `20`
- queue put timeout: `2.0`
- queue get timeout: `1.0`
- shutdown timeout: `30.0`

Verified worker components:

- camera reader worker: yes
- detection worker: yes
- tracking worker: yes
- persistence worker: yes
- colour worker: yes
- ANPR worker: yes

Rebuild rule:

```text
Do not migrate workers in the first rebuild.
First create a sequential pipeline.
Add workers only after the sequential pipeline is correct.
```

## 14. Configuration files

### Current files

- `config/cameras.yaml`
  - purpose: camera/video source list
  - important keys: `camera_code`, `enabled`, `source.path`, `start_time`
  - required in new repo: `YES`
  - can be simplified: `YES`

- `config/detection.yaml`
  - purpose: YOLO detector settings
  - important keys: `model_path`, `fallback_model_path`, `device`, `iou_threshold`, `image_size`, `allowed_classes`, `class_confidence_thresholds`
  - required in new repo: `YES`
  - can be simplified: `YES`

- `config/tracking.yaml`
  - purpose: ByteTrack and lifecycle/class settings
  - important keys: `behavior_mode`, `backend`, thresholds, buffers, `track_class`, `fragment_linking`, `identity_continuity`
  - required in new repo: `YES`
  - can be simplified: `YES`

- `config/persistence.yaml`
  - purpose: DB persistence behavior
  - important keys: backend, enabled, observation mode, media roles
  - required in new repo: `LATER`
  - can be simplified: `YES`

- `config/evidence.yaml`
  - purpose: crop/evidence selection rules
  - important keys: enabled, crop thresholds, padding, enabled roles
  - required in new repo: `YES`
  - can be simplified: `YES`

- `config/workers.yaml`
  - purpose: threaded pipeline queues and shutdown behavior
  - important keys: queue sizes, daemon flags, worker enable flags
  - required in new repo: `NO`
  - can be simplified: `N/A`

### Proposed minimal new config

```yaml
input:
  cameras: []

detection:
  model_path:
  device:
  confidence_threshold:
  iou_threshold:
  image_size:
  allowed_classes: []

tracking:
  backend:
  track_activation_threshold:
  minimum_matching_threshold:
  lost_track_buffer:

lifecycle:
  minimum_observations:
  maximum_lost_frames:

track_class:
  minimum_observations:
  minimum_winner_ratio:

output:
  directory:
```

## 15. Model and test data

Verified vehicle model files present:

- `F:\vinfo\Final_vedio_Ai_system\object\vehical_detection\best_old.pt`
- `F:\vinfo\Final_vedio_Ai_system\yolov8n.pt`
- `F:\vinfo\Final_vedio_Ai_system\yolo11n.pt`
- `F:\vinfo\Final_vedio_Ai_system\yolo11m.pt`

Configured active YOLO model path:

- `models/vehicle_detection/best_old.pt` from `config/detection.yaml`
- status: `UNVERIFIED / likely stale path in this checkout`

Fallback model path:

- `yolov8n.pt`
- status: exists at repository root

Whether weights are tracked by Git:

- weight files are present inside the repository tree
- actual Git tracking status: `UNVERIFIED` without checking index contents file-by-file

Verified local test videos:

- `tests\td_case2\multicamera_vehicle_tracking_pipeline\data\1test.mp4`
  - FPS: `30`
  - frame count: `256`
  - resolution: `1280x720`
- `tests\td_case2\multicamera_vehicle_tracking_pipeline\data\2test.mp4`
  - FPS: `30`
  - frame count: `256`
  - resolution: `1280x720`
- `tests\td_case2\multicamera_vehicle_tracking_pipeline\data\1ANPR1-D.mp4`
  - FPS: `60`
  - frame count: `1793`
  - resolution: `1920x1080`
- `tests\td_case2\multicamera_vehicle_tracking_pipeline\data\2ANPR1-D.mp4`
  - FPS: `60`
  - frame count: `1793`
  - resolution: `1920x1080`

CAM_002 test path:

- from stale config: `C:/Mukul K/vinfo1/video-search-engine/tests/td_case2/multicamera_vehicle_tracking_pipeline/data/2test.mp4`
- current local equivalent found: `F:\vinfo\Final_vedio_Ai_system\tests\td_case2\multicamera_vehicle_tracking_pipeline\data\2test.mp4`

Known useful frame range:

- experiment scripts often use `start-frame 0` and `end-frame 599`
- this exceeds the `256`-frame local `1test.mp4` and `2test.mp4` files
- therefore the exact intended useful range for the small local test videos is `UNVERIFIED`

## 16. Dependencies

Proven by imports and dependency files:

### Required initially

- `python>=3.11`
- `opencv-python` or `opencv-python-headless`
- `numpy`
- `ultralytics`
- `supervision`
- `pyyaml`
- `pytest`

### Required later

- `fastapi`
- `uvicorn`
- `pydantic`
- `pydantic-settings`
- `python-dotenv`
- `supabase`
- `torch`
- `transformers`
- `torchvision`
- `accelerate`

### Old repo only / not needed for first rebuild

- `google-genai`
- `qwen-vl-utils`
- `qdrant-client`
- Florence-related stack
- ANPR-related runtime stack

Notes:

- `requirements.txt` proves most Python dependencies
- `supervision` is proven by imports in tracking code, but is not pinned in the root `requirements.txt`

## 17. Tests worth recreating

Best conceptual tests to preserve:

- `tests/test_camera_source.py`
  - verify frame reading, FPS extraction, end-of-stream behavior
  - recreate in new repo: `YES`

- `tests/test_multi_camera_reader.py`
  - verify sequential and round-robin multi-camera reading
  - recreate in new repo: `YES`

- `tests/test_vehicle_detector.py`
  - verify detection filtering, class mapping, bbox conversion, thresholds
  - recreate in new repo: `YES`

- `tests/test_supervision_conversion.py`
  - verify conversion into Supervision Detections
  - recreate in new repo: `YES`

- `tests/test_tracker_factory.py`
  - verify one tracker instance per camera and backend config mapping
  - recreate in new repo: `YES`

- `tests/test_camera_tracker.py`
  - verify native tracker IDs flow through tracking update
  - recreate in new repo: `YES`

- `tests/test_track_lifecycle.py`
  - verify create/update/complete/discard behavior
  - recreate in new repo: `YES`

- `tests/test_class_stabilization.py`
  - verify final-class calculation and UNKNOWN cases
  - recreate in new repo: `YES`

- `tests/test_track_evidence_collector.py`
  - verify evidence selection and artifact writing
  - recreate in new repo: `YES, simplified`

- `tests/test_tracking_orchestrator.py`
  - verify full sequential pipeline report behavior
  - recreate in new repo: `YES`

- `tests/test_tracking_persistence_service.py`
  - verify completed-track persistence rules
  - recreate in new repo: `LATER`

- `tests/test_tracking_worker.py`
  - worker-only threading behavior
  - recreate in new repo: `NO, later only if workers return`

## 18. Experimental complexity to avoid

### DO NOT COPY DIRECTLY

- `workers/` threaded architecture
  - reason: adds concurrency, queues, shutdown handling, and more failure modes before the sequential pipeline is proven

- `cross_camera/`
  - reason: global matching is outside the Milestone 1 single-track local pipeline

- `enrichment/` ANPR, Florence, vehicle colour, body type
  - reason: unrelated to the minimum vehicle detection/tracking rebuild

- `tracking/track_lifecycle.py` custom split/relink sections
  - reason: this file mixes the basic lifecycle with experimental identity logic; retain only the simple lifecycle concept

- `run_class_confidence_experiment.py`
  - reason: threshold sweeps are experiment tooling, not baseline pipeline behavior

- `run_tracking_confidence_experiment.py`
  - reason: large experiment runner with many debug outputs and optional enrichment

- `run_standard_bytetrack_comparison.py`
  - reason: useful as a reference, but should not be copied as production pipeline code

- `run_final_class_parameter_matrix.py`
  - reason: experiment-only parameter sweep

- `run_class_consistency_experiment.py`
  - reason: experiment-only analysis

- `diagnose_track_identity_switch.py`
  - reason: debugging helper for identity problems after the pipeline already exists

- `frontend/` and `api/`
  - reason: not part of the local sequential vehicle-tracking core

- `database/migrations/` full analytics schema set
  - reason: too much database scope for Milestone 1

- `debug_runs/`, generated artifacts, and caches
  - reason: outputs, not source logic

## 19. Old-to-new file mapping

| Old responsibility | Old file(s) | New clean file | Copy code directly? | What to retain |
| --- | --- | --- | --- | --- |
| frame packet model | `ingestion/frame_packet.py` | `src/models.py` | No | packet fields and minimal metadata only |
| camera config loading | `ingestion/camera_config.py` | `src/camera_reader.py` + `config.yaml` parser | No | camera list parsing and path resolution concept |
| video open/read | `ingestion/camera_source.py` | `src/camera_reader.py` | No | OpenCV open/read, FPS/frame count, frame numbering, EOS handling |
| multi-camera sequencing | `ingestion/multi_camera_reader.py` | `src/camera_reader.py` | No | sequential and round-robin reading behavior |
| YOLO detection wrapper | `detection/vehicle_detector.py` | `src/detector_tracker.py` | No | shared YOLO loading, filtering, thresholds, normalized outputs |
| detection data model | `detection/detection_models.py` | `src/models.py` | No | `Detection` structure |
| ByteTrack creation | `tracking/tracker_factory.py` | `src/detector_tracker.py` | No | one tracker per camera, standard backend config |
| YOLO->ByteTrack conversion | `tracking/supervision_conversion.py` | `src/detector_tracker.py` | No | conversion into tracker input |
| per-camera tracking update | `tracking/camera_tracker.py` | `src/detector_tracker.py` | No | native tracker id propagation and per-camera isolation |
| simple lifecycle | `tracking/track_lifecycle.py` | `src/track_manager.py` | No | create/update/lost/complete/discard only |
| final class calculation | `tracking/class_stabilization.py` | `src/track_manager.py` | No | simple class aggregation and UNKNOWN thresholding |
| evidence selection | `evidence/track_evidence_collector.py` | `src/evidence.py` | No | first/last/best-overall crop selection and annotated frame writing |
| JSON/report writing | orchestrator report builders | `src/output_writer.py` | No | completed-track summary serialization |
| top-level sequential pipeline | `orchestration/multicamera_tracking_orchestrator.py` | `src/pipeline.py` | No | sequential stage wiring only |

## 20. Proposed clean new-repo flow

Suggested new structure:

```text
src/
  models.py
  camera_reader.py
  detector_tracker.py
  track_manager.py
  evidence.py
  output_writer.py
  pipeline.py

app.py
config.yaml
requirements.txt
README.md

tests/
  test_camera_reader.py
  test_detector_tracker.py
  test_track_manager.py
  test_pipeline.py
```

Suggested runtime flow:

```text
config.yaml
  ->
camera_reader.py
  ->
detector_tracker.py
  ->
track_manager.py
  ->
evidence.py
  ->
output_writer.py
  ->
report.json
```

## 21. Open questions and unverified assumptions

- `config/cameras.yaml` paths are stale in this checkout; direct runtime validity is `UNVERIFIED`
- `CAM_003` configured file was not found locally
- `config/detection.yaml` primary model path does not exist as written in this checkout; whether the original runs depended on a different checkout layout is `UNVERIFIED`
- Exact installed `ultralytics` and `supervision` versions used in the successful historical runs are `UNVERIFIED`
- Exact “active” SQL migration sequence used for the newest analytics DB state is `UNVERIFIED`; many migrations exist
- The exact small-video frame range used for the best tracking experiments is `UNVERIFIED`

## 22. First rebuild milestone

Milestone 1:

```text
one local video
-> sequential frame reading
-> YOLO vehicle detection
-> standard ByteTrack
-> native tracker IDs
-> simple completed-track summary
-> JSON output
```

No workers, database, API, ANPR, vehicle colour, fragment linking or logical splitting in Milestone 1.

REFERENCE CREATED
File: F:\vinfo\Final_vedio_Ai_system\OLD_REPO_REFERENCE.md
Old repository code modified: NO
