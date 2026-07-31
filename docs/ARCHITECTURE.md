# Architecture

## Current Step 5 Architecture

```text
app.py
-> configuration
-> run setup
-> multi-camera ingestion manager
-> 7 shared ingestion workers by default
-> bounded FramePacket queue
-> one shared YOLO model
-> one ByteTrack instance per camera
-> one local TrackManager
-> one post-track EvidenceCollector
-> raw-frame sampling
-> detected-frame sampling
-> tracked-frame sampling
-> local track lifecycle
-> evidence candidate scoring and selection
-> compact crops and annotated evidence frames
-> completed tracks and observations output
-> evidence index and evidence metrics output
-> logging
-> summary, ingestion metrics, detection/tracking metrics, and lifecycle output
```

## Active Components

- `app.py` parses the CLI config path and starts the pipeline.
- `src/pipeline.py` coordinates config loading, validation, ingestion startup, queue consumption, YOLO inference, per-camera tracking, output sampling, metadata, metrics, summary writing, and failure handling.
- `src/camera_reader.py` owns the per-camera `VideoCapture` state and sequential frame numbering.
- `src/ingestion_manager.py` distributes enabled cameras across shared workers and manages the bounded shared frame queue.
- `src/detector_tracker.py` loads the YOLO model once, filters detections, converts them to Supervision detections, and updates one native ByteTrack instance per camera.
- `src/track_manager.py` turns native per-camera tracked detections into local track observations, lifecycle state, and final classes.
- `src/evidence.py` receives tracked detections plus live frames, caches bounded candidate frames, selects representative evidence after track completion, and writes crops and annotated full frames.
- `src/output_writer.py` creates isolated run folders and stores reproducible run artifacts.
- `src/logging_setup.py` configures console and file logging.
- `src/models.py` defines shared ingestion, detection, and tracked-observation dataclasses.
- Local track identity is always `camera_id + native_tracker_id`.
- ByteTrack native IDs are not merged across cameras.
- Final class is a transparent confidence-weighted majority over the preserved raw per-frame classes.
- Evidence is selected only after a local track is completed or discarded.
- Evidence selection does not influence YOLO, ByteTrack, lifecycle state, or final-class calculation.

## Ingestion Notes

- N cameras are configurable through `config.yaml`.
- Worker assignment is deterministic by `camera_index % worker_count`.
- The default worker count is 7, but it is configurable.
- Workers are shared across cameras.
- One `VideoCapture` is owned by exactly one reader and one worker thread.
- Each camera preserves its own frame order independently.
- `raw_frames/` contains frames directly from ingestion.
- `detected_frames/` contains YOLO-only annotations.
- `tracked_frames/` contains native ByteTrack annotations.
- `evidence/` contains representative per-track crops and annotated full frames.
- ByteTrack state is never shared across cameras.
- Evidence frame caching is bounded to frames referenced by active track candidates and is released when the track finalizes.

Warning:

Seven workers do not mean unlimited physical throughput.

For many high-resolution live streams, GPU detection capacity, decoding cost,
network bandwidth and CPU resources must later be measured before selecting
the production worker count.

## Planned Architecture

```text
multiple cameras
-> YOLO detection
-> Supervision ByteTrack
-> lifecycle
-> evidence
-> colour
-> OCR
-> persistence
-> global vehicle identity
```
