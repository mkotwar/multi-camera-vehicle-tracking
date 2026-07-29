# Architecture

## Current Step 3 Architecture

```text
app.py
-> configuration
-> run setup
-> multi-camera ingestion manager
-> 7 shared ingestion workers by default
-> bounded FramePacket queue
-> one shared YOLO model
-> one ByteTrack instance per camera
-> raw-frame sampling
-> detected-frame sampling
-> tracked-frame sampling
-> logging
-> summary, ingestion metrics, and detection/tracking metrics output
```

## Active Components

- `app.py` parses the CLI config path and starts the pipeline.
- `src/pipeline.py` coordinates config loading, validation, ingestion startup, queue consumption, YOLO inference, per-camera tracking, output sampling, metadata, metrics, summary writing, and failure handling.
- `src/camera_reader.py` owns the per-camera `VideoCapture` state and sequential frame numbering.
- `src/ingestion_manager.py` distributes enabled cameras across shared workers and manages the bounded shared frame queue.
- `src/detector_tracker.py` loads the YOLO model once, filters detections, converts them to Supervision detections, and updates one native ByteTrack instance per camera.
- `src/output_writer.py` creates isolated run folders and stores reproducible run artifacts.
- `src/logging_setup.py` configures console and file logging.
- `src/models.py` defines shared ingestion, detection, and tracked-observation dataclasses.

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
- ByteTrack state is never shared across cameras.

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
