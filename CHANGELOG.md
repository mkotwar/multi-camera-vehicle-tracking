# Changelog

## 2026-07-30

- Added Step 5 evidence collection in `src/evidence.py`.
- Added configurable evidence selection, crop padding, sharpness scoring, best-overall weighting, discarded-track handling, and strict/non-strict write behavior.
- Integrated evidence registration and post-track finalization into `src/pipeline.py` without changing YOLO, ByteTrack, lifecycle, or final-class behavior.
- Added `evidence_index.json`, `evidence_metrics.json`, per-track `evidence.json`, and per-track crop/annotated-frame output folders.
- Linked evidence summaries back into `tracks.json`.
- Added focused evidence tests and expanded pipeline/output coverage for the new run artifacts.

## 2026-07-29

- Audited the clean repository for Step 1 foundation requirements.
- Implemented centralized configuration, structured logging, run output management, camera-reading validation, metadata, summary output, error reporting, documentation, and tests.
- Kept Step 1 free of YOLO, tracking, persistence, workers, API, and frontend code.
- Extended the foundation to Step 2 configurable multi-camera ingestion.
- Added support for shared ingestion workers, bounded frame queues, per-camera/per-worker metrics, sampled raw-frame saving, and a detected-frame placeholder directory.
- Added Step 3 YOLO detection and per-camera Supervision ByteTrack tracking in a single `src/detector_tracker.py` implementation.
- Added detected/tracked frame sampling outputs, detection/tracking metrics, model-weight portability notes, and Step 3 test coverage.
- Added Step 4 local track lifecycle handling, track summaries, and final-class output.
