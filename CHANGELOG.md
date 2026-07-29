# Changelog

## 2026-07-29

- Audited the clean repository for Step 1 foundation requirements.
- Implemented centralized configuration, structured logging, run output management, camera-reading validation, metadata, summary output, error reporting, documentation, and tests.
- Kept Step 1 free of YOLO, tracking, persistence, workers, API, and frontend code.
- Extended the foundation to Step 2 configurable multi-camera ingestion.
- Added support for shared ingestion workers, bounded frame queues, per-camera/per-worker metrics, sampled raw-frame saving, and a detected-frame placeholder directory.
- Added Step 3 YOLO detection and per-camera Supervision ByteTrack tracking in a single `src/detector_tracker.py` implementation.
- Added detected/tracked frame sampling outputs, detection/tracking metrics, model-weight portability notes, and Step 3 test coverage.
