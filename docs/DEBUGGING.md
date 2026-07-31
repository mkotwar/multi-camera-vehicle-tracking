# Debugging

## Step 5 Outputs

When the lifecycle and evidence stages are active, inspect:

- `tracks.json`
- `observations.csv`
- `track_lifecycle_metrics.json`
- `evidence_index.json`
- `evidence_metrics.json`
- `evidence/`

Quick checks:

- `active_tracks_at_shutdown` should be `0`
- `duplicate_observation_count` should be `0`
- `tracks_completed_by_camera` should match the finalized local tracks
- CSV row counts should line up with per-track `observation_count`
- `UNKNOWN` final classes should include a clear `final_class_reason`
- `tracks_with_evidence` should be greater than `0` for usable completed tracks
- duplicate evidence roles should reuse the same crop and annotated-frame file paths

Lifecycle reminders:

- `TENTATIVE` becomes `ACTIVE` after `lifecycle.minimum_observations`
- missing native IDs are counted once per processed camera frame
- tracks complete when `lost_frames > maximum_lost_frames`
- short tracks are discarded, not completed

## Where To Look First

- `outputs/runs/<run_id>/pipeline.log`
- `outputs/runs/<run_id>/run_metadata.json`
- `outputs/runs/<run_id>/summary.json`
- `outputs/runs/<run_id>/ingestion_metrics.json`
- `outputs/runs/<run_id>/detection_tracking_metrics.json`
- `outputs/runs/<run_id>/raw_frames/`
- `outputs/runs/<run_id>/detected_frames/`
- `outputs/runs/<run_id>/tracked_frames/`
- `outputs/runs/<run_id>/evidence/`
- `outputs/runs/<run_id>/errors/`

## Common Step 3 Issues

- Invalid `config.yaml` structure
- Missing or disabled camera configuration
- Invalid video path or source type
- Duplicate camera IDs
- Queue saturation under slow consumption
- One camera ending earlier than another
- Missing model file
- Configured allowed classes not matching model class names
- YOLO detections appearing but tracker IDs missing
- Invalid log level
- Evidence candidates rejected by invalid or tiny crops
- Evidence files missing because strict write mode raised a pipeline error

## Quick Checks

1. Confirm every enabled `source` exists or is valid for its `source_type`.
2. Confirm `input.max_frames_per_camera` is a positive integer.
3. Confirm `ingestion.worker_count` and `ingestion.frame_queue_size` are positive.
4. Confirm worker assignments were logged.
5. Confirm the run folder was created under the configured output root.
6. Confirm the model path resolved correctly.
7. Confirm `pipeline.log` contains startup, worker assignment, model load, tracker creation, evidence initialization, progress, and completion or failure entries.
