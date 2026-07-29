# Pipeline Flow

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
