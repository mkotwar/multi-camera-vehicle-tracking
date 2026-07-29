# REFERENCE_BUNDLE_MAP

## Scope

This file maps the runtime bundle at:

- `F:\vinfo\multicamera_vehicle_tracking_clean\multicamera_vehicle_tracking_runtime_bundle.zip`

to the minimum components needed for the first clean rebuild milestone:

`one local video -> frame reading -> YOLO detection -> standard ByteTrack -> simple track summary -> JSON output`

Rules followed for this mapping:

- The ZIP was inspected as a reference only.
- No complete source files were copied into the new repository.
- No existing Python files in the new repository were modified.
- This document names only the smallest useful subset for the first sequential rebuild.

## Bundle Root

Primary runtime package inside the ZIP:

- `multicamera_vehicle_tracking_runtime_bundle/tests/td_case2/multicamera_vehicle_tracking_pipeline/`

Useful reference script entry point for the sequential path:

- `multicamera_vehicle_tracking_runtime_bundle/tests/td_case2/multicamera_vehicle_tracking_pipeline/scripts/validate_multicamera_tracking.py`

Primary sequential orchestration file:

- `multicamera_vehicle_tracking_runtime_bundle/tests/td_case2/multicamera_vehicle_tracking_pipeline/orchestration/multicamera_tracking_orchestrator.py`

## 1. Useful Video Files In The ZIP

Bundle video assets found under:

- `multicamera_vehicle_tracking_runtime_bundle/tests/td_case2/multicamera_vehicle_tracking_pipeline/data/`

Useful for first clean milestone:

| ZIP path | Size | Use now | Notes |
|---|---:|---|---|
| `multicamera_vehicle_tracking_runtime_bundle/tests/td_case2/multicamera_vehicle_tracking_pipeline/data/1test.mp4` | 5,303,076 bytes | Yes | Best first local test video for single-video sequential rebuild. |
| `multicamera_vehicle_tracking_runtime_bundle/tests/td_case2/multicamera_vehicle_tracking_pipeline/data/2test.mp4` | 5,303,076 bytes | Later | Same role as `1test.mp4`, useful after the first single-video pass works. |
| `multicamera_vehicle_tracking_runtime_bundle/tests/td_case2/multicamera_vehicle_tracking_pipeline/data/1ANPR1-D.mp4` | 71,761,507 bytes | No | Larger ANPR-oriented sample; not needed for first YOLO + ByteTrack milestone. |
| `multicamera_vehicle_tracking_runtime_bundle/tests/td_case2/multicamera_vehicle_tracking_pipeline/data/2ANPR1-D.mp4` | 71,761,507 bytes | No | Same as above; later enrichment/ANPR stage. |

Non-video images in the same folder:

- `best_overall.jpg`
- `images.jpg`

These are not required for the first milestone.

Important config mismatch:

- `config/cameras.yaml` and `config/cameras_user_three_videos.yaml` reference `1test.mp4` and `2test.mp4`, but also reference `3test .mp4`.
- `3test .mp4` is not present in the ZIP data folder.
- For the clean rebuild, use one real local video first, preferably `1test.mp4`.

## 2. Active YOLO Detector Files

These are the detector files actively used by the sequential tracking path:

| ZIP path | Responsibility |
|---|---|
| `.../config/detection.yaml` | Declares the active detector model path, fallback model, IOU, image size, and per-class thresholds. |
| `.../detection/detection_config.py` | Loads and validates detector config and environment overrides. |
| `.../detection/detection_models.py` | Defines `VehicleDetection` and `DetectionPacket`. |
| `.../detection/vehicle_detector.py` | Runs YOLO inference and converts predictions into normalized vehicle detections. |

Most important detector class/function:

- `SharedVehicleDetector.detect(...)`
- `SharedVehicleDetector._convert_predictions(...)`
- `load_detection_config(...)`

Detector behavior confirmed from the bundle code:

- YOLO backend import: `ultralytics.YOLO`
- Input: raw frame from `FramePacket.frame`
- Output: `DetectionPacket`
- Output classes normalized to:
  - `3wheeler`
  - `bus`
  - `car`
  - `motorcycle`
  - `truck`

## 3. Active Tracking Files

These are the tracking files actively used by the standard sequential ByteTrack path:

| ZIP path | Responsibility |
|---|---|
| `.../config/tracking.yaml` | Active tracking settings for standard ByteTrack. |
| `.../tracking/tracking_config.py` | Loads and validates tracking config and overrides. |
| `.../tracking/tracker_factory.py` | Creates `supervision.ByteTrack` or alternative backend instances. |
| `.../tracking/supervision_conversion.py` | Converts `DetectionPacket` into `supervision.Detections`. |
| `.../tracking/camera_tracker.py` | Applies per-frame tracker update and emits native tracker IDs as observations. |
| `.../tracking/tracking_models.py` | Defines `TrackObservation` and `LocalVehicleTrack`. |
| `.../tracking/track_lifecycle.py` | Maintains per-track state and finalizes track summaries. |
| `.../tracking/camera_detection_router.py` | Routes each camera's detections into its own tracker instance. |
| `.../orchestration/multicamera_tracking_orchestrator.py` | Sequential glue: reader -> detector -> tracker -> summary/report JSON. |

Most important tracking class/function:

- `TrackerFactory._default_tracker_creator(...)`
- `to_supervision_detections(...)`
- `CameraTracker.update(...)`
- `LocalTrackLifecycle.update(...)`

For the clean milestone, the essential tracking subset is:

- `tracking_config.py`
- `tracker_factory.py`
- `supervision_conversion.py`
- `camera_tracker.py`
- `tracking_models.py`

`track_lifecycle.py` is useful as reference for summary shape, but the clean rebuild can start with a much simpler summary implementation.

## 4. Active Detection And Tracking Configuration Values

### Detection values from `config/detection.yaml`

Active detector config:

- `model_path: models/vehicle_detection/best_old.pt`
- `fallback_model_path: yolov8n.pt`
- `allow_fallback: true`
- `device: auto`
- `iou_threshold: 0.45`
- `image_size: 640`

Allowed classes:

- `3wheeler`
- `car`
- `bus`
- `truck`
- `motorcycle`

Per-class confidence thresholds:

- `3wheeler: 0.65`
- `car: 0.70`
- `bus: 0.75`
- `truck: 0.75`
- `motorcycle: 0.25`

Practical clean-rebuild meaning:

- The detector uses the minimum of those class thresholds as the YOLO inference floor.
- Final accept/reject is done per class after YOLO prediction conversion.

### Tracking values from `config/tracking.yaml`

Active tracking config:

- `behavior_mode: standard_bytetrack`
- `backend: supervision_bytetrack`
- `track_activation_threshold: 0.15`
- `lost_track_buffer: 30`
- `minimum_matching_threshold: 0.80`
- `frame_rate: 20`
- `minimum_consecutive_frames: 1`
- `track_high_thresh: 0.15`
- `track_low_thresh: 0.10`
- `new_track_thresh: 0.30`
- `match_thresh: 0.80`
- `track_buffer: 30`
- `min_confirmed_observations: 3`
- `max_lost_frames: 30`
- `preserve_state_per_camera: true`

Track class settings present in the bundle:

- `track_class.strategy: confidence_weighted_majority`
- `track_class.minimum_observations: 3`
- `track_class.minimum_winner_ratio: 0.50`

Important clean milestone interpretation:

- `behavior_mode: standard_bytetrack` is the correct target.
- Ignore `experimental_custom` behavior in the first clean rewrite.
- Use one tracker per video/camera.
- Use `native_tracker_id` directly as the initial local track ID output.

## 5. Model Paths Referenced By The Bundle

### Active first-milestone detector model references

Referenced by the bundle:

- `models/vehicle_detection/best_old.pt`
- `yolov8n.pt`

Reality check:

- No `.pt` model files were found inside the ZIP.
- These model paths are references only.
- The clean rebuild must provide its own detector weight file path locally.

### Other model references present but not needed now

From config files:

- `config/anpr.yaml`
  - `model_path: null`
  - `model_path_env: PLATE_DETECTOR_MODEL_PATH`
- `config/florence.yaml`
  - `model_path: null`
  - `model_path_env: FLORENCE_MODEL_PATH`

Bundle helper for model resolution:

- `multicamera_vehicle_tracking_runtime_bundle/tests/td_case2/multicamera_vehicle_tracking_pipeline/models/model_path_resolver.py`

Clean rewrite guidance:

- Reuse only the concept: `CLI override -> env override -> config value -> default`.
- Do not bring ANPR or Florence model resolution into milestone 1.

## 6. Minimum Dependencies

### Minimum dependency set for the clean first milestone

This is the real minimum dependency set inferred from the active sequential code path, not from the larger bundle root `requirements.txt`:

- Python `>=3.11`
- `opencv-python-headless`
- `ultralytics`
- `torch`
- `numpy`
- `supervision`
- `pyyaml`

Likely optional for the first milestone:

- `scipy`

Reason:

- `scipy` is only relevant to the alternate `ultralytics_bytetrack` LAP shim path in `tracker_factory.py`.
- It is not required if the clean rebuild uses only `backend: supervision_bytetrack`.

Not minimum for milestone 1 even though present in root `requirements.txt`:

- `fastapi`
- `uvicorn`
- `pydantic`
- `pydantic-settings`
- `loguru`
- `pytest`
- `httpx`
- `python-dotenv`
- `python-multipart`
- `torchvision`
- `transformers`
- `accelerate`
- `qwen-vl-utils`
- `qdrant-client`
- `google-genai`

Important note:

- `supervision` is required by the active standard ByteTrack path, but it is not listed in the copied bundle root `requirements.txt`.
- That requirement is inferred from imports in `tracking/supervision_conversion.py` and `tracking/tracker_factory.py`.

## 7. Files That Should Not Be Copied Initially

Do not copy these areas into the clean repo for milestone 1:

- `api/`
- `frontend/`
- `database/`
- `database/migrations/`
- `cross_camera/`
- `enrichment/`
- ANPR-related code and configs
- Florence/VLM-related code and configs
- `workers/`
- worker orchestration files
- `persistence/`
- experiment and comparison scripts
- preview/debug/reporting helpers not needed for one-video JSON output
- `__pycache__/`
- generated reports
- generated artifacts
- duplicate or stale camera configs

Specific files/folders to avoid initially:

- `.../orchestration/worker_multicamera_tracking_orchestrator.py`
- `.../scripts/validate_worker_multicamera_tracking.py`
- `.../scripts/run_*experiment*.py`
- `.../scripts/run_standard_bytetrack_comparison.py`
- `.../scripts/diagnose_track_identity_switch.py`
- `.../scripts/demo_*`
- `.../scripts/seed_test_data.py`
- `.../evidence/` for the first pass
- `.../persistence/`
- `.../database/`
- `.../cross_camera/`
- `.../workers/`

## 8. Minimal Files And Concepts To Study

### A. Frame packet

- ZIP path:
  - `multicamera_vehicle_tracking_runtime_bundle/tests/td_case2/multicamera_vehicle_tracking_pipeline/ingestion/frame_packet.py`
- Responsibility:
  - Carries one decoded frame plus source metadata through the pipeline.
- Important function/class:
  - `FramePacket`
- Dependencies:
  - `dataclasses`
  - `datetime`
  - `pathlib`
- Concept to rewrite in the new repo:
  - Small immutable-ish data container with:
    - video path
    - frame index
    - fps
    - timestamp seconds
    - image array

### B. Camera/video reader

- ZIP paths:
  - `.../ingestion/camera_config.py`
  - `.../ingestion/camera_source.py`
  - `.../ingestion/multi_camera_reader.py`
- Responsibility:
  - Load local file video config, open `cv2.VideoCapture`, read frames, emit `FramePacket`.
- Important function/class:
  - `load_camera_configs(...)`
  - `CameraSource.open(...)`
  - `CameraSource.read_next(...)`
  - `MultiCameraReader`
- Dependencies:
  - `opencv-python-headless`
  - `pathlib`
  - `pyyaml`
- Concept to rewrite in the new repo:
  - For milestone 1, collapse this to a single-video reader first.
  - Keep:
    - safe file existence check
    - fps/frame-count extraction
    - `frame_number / fps` time calculation
  - Skip:
    - multi-camera round robin
    - start-time policy

### C. Vehicle detector

- ZIP paths:
  - `.../detection/detection_config.py`
  - `.../detection/detection_models.py`
  - `.../detection/vehicle_detector.py`
  - `.../config/detection.yaml`
- Responsibility:
  - Load YOLO model, run inference, normalize labels, filter by class and threshold, output structured detections.
- Important function/class:
  - `SharedVehicleDetector`
  - `SharedVehicleDetector.detect(...)`
  - `SharedVehicleDetector._convert_predictions(...)`
- Dependencies:
  - `ultralytics`
  - `torch`
  - `numpy` indirectly through frame arrays
- Concept to rewrite in the new repo:
  - Keep:
    - single shared YOLO detector instance
    - normalized class mapping
    - bbox clamp/validation
    - per-class threshold filtering
  - Skip:
    - fallback to persistence class mapping module
    - counters and long diagnostics until later

### D. YOLO result conversion

- ZIP path:
  - `.../tracking/supervision_conversion.py`
- Responsibility:
  - Convert internal detection objects into `supervision.Detections`.
- Important function/class:
  - `to_supervision_detections(...)`
- Dependencies:
  - `numpy`
  - `supervision`
- Concept to rewrite in the new repo:
  - Convert `[x1, y1, x2, y2, confidence, class_id]` style data into the exact structure expected by `supervision.ByteTrack`.

### E. Supervision ByteTrack initialization

- ZIP path:
  - `.../tracking/tracker_factory.py`
- Responsibility:
  - Create one ByteTrack instance per camera/video and inject tracking thresholds.
- Important function/class:
  - `TrackerFactory`
  - `TrackerFactory._default_tracker_creator(...)`
- Dependencies:
  - `supervision`
- Concept to rewrite in the new repo:
  - Initialize one `supervision.ByteTrack` with:
    - `track_activation_threshold`
    - `lost_track_buffer`
    - `minimum_matching_threshold`
    - `frame_rate`
    - `minimum_consecutive_frames`

### F. Tracker update

- ZIP paths:
  - `.../tracking/camera_tracker.py`
  - `.../tracking/camera_detection_router.py`
- Responsibility:
  - Feed converted detections into ByteTrack and turn tracker outputs into track observations.
- Important function/class:
  - `CameraTracker.update(...)`
  - `CameraTracker._update_with_supervision_tracker(...)`
  - `CameraDetectionRouter.route(...)`
- Dependencies:
  - `supervision`
  - tracking models
  - lifecycle model/state
- Concept to rewrite in the new repo:
  - For milestone 1:
    - one tracker
    - one update per frame
    - collect track IDs, boxes, class names, confidences
  - Skip:
    - router abstraction for multiple cameras
    - alternate row-tracker path

### G. Native tracker ID output

- ZIP paths:
  - `.../tracking/camera_tracker.py`
  - `.../tracking/tracking_models.py`
  - `.../tracking/track_lifecycle.py`
- Responsibility:
  - Preserve ByteTrack-emitted ID as the native tracking identity and carry it into summary objects.
- Important function/class:
  - `TrackObservation.native_tracker_id`
  - `CameraTracker._update_with_supervision_tracker(...)`
  - `build_track_uuid(...)`
- Dependencies:
  - tracking models
- Concept to rewrite in the new repo:
  - Expose the ByteTrack ID directly in output JSON.
  - Use it as the initial local `track_id` for the clean sequential build.
  - Add UUID-style wrapping only if needed later.

## 9. Recommended Minimal Rewrite Order

Recommended order for the clean repo:

1. Implement a one-video reader based on `FramePacket` + `CameraSource.read_next`.
2. Implement a shared YOLO detector based on `SharedVehicleDetector`.
3. Implement detection-to-supervision conversion.
4. Implement one `supervision.ByteTrack` instance.
5. Emit per-frame observations with native ByteTrack IDs.
6. Build a simple per-track summary:
   - first frame
   - last frame
   - observation count
   - best confidence
   - class name
   - list of boxes or sampled observations
7. Write JSON output.

## 10. What To Reuse Conceptually Versus What To Ignore

Reuse conceptually now:

- `FramePacket`
- local file video reader
- YOLO wrapper
- normalized detection objects
- `supervision.ByteTrack`
- native tracker ID propagation
- simple track summary JSON

Ignore for now:

- APIs
- UI/frontend
- database schema/repository
- Supabase persistence
- evidence package generation
- ANPR
- Florence/VLM
- cross-camera identity matching
- worker threads/queues/messages
- experiment scripts
- debug artifact generation

REFERENCE READY: The runtime bundle has been mapped to the minimum components needed for the clean sequential YOLO and ByteTrack rebuild.
