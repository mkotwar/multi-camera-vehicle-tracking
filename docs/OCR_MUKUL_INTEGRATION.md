# OCR_MUKUL Detector/Tracker Integration

The main pipeline now uses the OCR_MUKUL detector/tracker pair by default while keeping the existing:

- multi-camera ingestion
- frame packet handling
- worker orchestration
- evidence collection
- persistence
- output writing
- run summaries and logging

## Detector

The detector backend is `ocr_mukul`.

Integrated source:

- `C:\Mukul K\OCR_MUKUL\anpr_frog_speed.py`

Reused detector behavior:

- `YOLO(VEHICLE_MODEL_PATH)` model loading
- callable inference form `vehicle_model(frame, conf=VEHICLE_CONFIDENCE_THRESHOLD, imgsz=1024, verbose=False)[0]`
- OCR_MUKUL confidence default `0.2`
- OCR_MUKUL image size default `1024`
- OCR_MUKUL class-id allowlist `0..7`

## Tracker

The tracker backend is `ocr_mukul_supervision_bytetrack`.

Integrated source:

- `C:\Mukul K\OCR_MUKUL\anpr_frog_speed.py`

Confirmed tracker implementation:

- `import supervision as sv`
- `tracker = sv.ByteTrack(...)`
- `tracked_detections = tracker.update_with_detections(detections)`

Reused tracker parameters:

- `lost_track_buffer=40`
- `track_activation_threshold=0.3`
- `minimum_matching_threshold=0.6`
- `minimum_consecutive_frames=3`

## Multi-camera behavior

Each camera gets its own independent tracker instance inside the clean project runtime:

- `CAM_001 -> tracker instance`
- `CAM_002 -> tracker instance`
- `CAM_N -> tracker instance`

The detector model is shared by the worker process, but tracker state stays camera-specific.
