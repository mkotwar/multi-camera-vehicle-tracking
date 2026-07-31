# Pipeline Flow

## Step 5 Flow

```text
FramePacket
-> YOLO detection
-> bbox-quality filter
-> Supervision ByteTrack
-> TrackedDetection
-> EvidenceCollector.register_frame(...)
-> TrackManager.update_frame(...)
-> TrackObservation append/update
-> LocalTrack lifecycle transitions
-> completed/discarded local tracks
-> EvidenceCollector.finalize_tracks(...)
-> evidence crops / annotated frames / evidence.json / evidence_index.json / evidence_metrics.json
-> tracks.json / observations.csv / track_lifecycle_metrics.json
```

Notes:

- ByteTrack supplies native per-camera tracker IDs only.
- `TrackManager` does not split, merge, or globally associate tracks.
- Local identity is `camera_id + native_tracker_id`.
- Final class is a simple confidence-weighted majority summary at track completion.
- Evidence is selected after local track completion only.
- Evidence does not feed back into detection, tracking, lifecycle, or final-class logic.
- Only a small number of representative frames are stored per completed local track.

## Step 3 Flow

```text
load config
-> validate config
-> create run output manager
-> configure logging
-> save effective config
-> create metadata
-> build multi-camera ingestion manager
-> load one shared YOLO model
-> assign cameras across shared workers
-> start workers
-> consume FramePacket queue
-> run YOLO on each frame
-> convert accepted detections to supervision.Detections
-> update one ByteTrack instance per camera
-> optionally save sampled raw frames
-> optionally save sampled detected frames
-> optionally save sampled tracked frames
-> accumulate frames by camera and worker
-> accumulate detections and tracked observations
-> wait for all cameras to finish
-> stop workers
-> reset trackers
-> save metadata, summary, ingestion metrics, and detection/tracking metrics
-> finish
```

## Failure Flow

```text
load config
-> create run directory
-> configure logging
-> validate configuration or source open
-> capture exception
-> log failure
-> save structured error JSON
-> save failure summary
-> mark metadata FAILED
```
