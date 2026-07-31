# 1test.mp4 Analysis Checklist

Use this checklist after running:

```powershell
.\.venv\Scripts\python.exe app.py --config config.single_1test.yaml
```

## Detection Quality

- Check that real vehicles are detected throughout the video.
- Check for missed vehicles, especially during occlusion, entry, and exit.
- Check for false vehicle detections on background objects.
- Check for wrong vehicle classes.
- Check whether bounding boxes are too small or too large.
- Check for duplicate overlapping detections on the same vehicle.
- Check `bbox_quality_metrics.json` for detections rejected by bbox-quality filtering and verify whether those rejections look correct.

## Tracking Quality

- Check that one physical vehicle keeps the same ID over time.
- Check that two different vehicles do not share one ID.
- Check that IDs do not switch during close overlap or crossing.
- Check that cars and motorcycles remain separated correctly.
- Check that tracks do not fragment unnecessarily.
- Check that a lost vehicle gets a new ID only when that behavior is expected.
- Check class-isolated tracking behavior only if the configured isolation mode changes from `per_camera`.

## Track Summaries

- Check each track’s first and last frames in `tracks.json`.
- Check observation counts for unusually short or long tracks.
- Check final class values for correctness.
- Check any mixed raw classes warnings against `observations.csv`.
- Check for `UNKNOWN` final classes and record the reason when they appear.
- Check completed and discarded track counts in `track_lifecycle_metrics.json`.
- Check duplicate observation count in `track_lifecycle_metrics.json`.
- Check `active_tracks_at_shutdown` and confirm it is reasonable for the end of the video.

## Evidence Quality

- Check that each crop contains the intended vehicle.
- Check that no crop is empty or badly clipped.
- Check that padding around the vehicle looks reasonable.
- Check that first, middle, and last evidence images match the track timeline.
- Check that highest-confidence evidence is actually a strong detection.
- Check that sharpest evidence looks visually clear.
- Check that best-overall evidence is useful for review.
- Check that duplicate evidence images are not being stored unnecessarily.

## Performance

- Check the resolved device in `summary.json`, `run_metadata.json`, and `detection_tracking_metrics.json`.
- Check whether CPU or CUDA was actually used.
- Check total processed frames.
- Check detection timing in `detection_tracking_metrics.json`.
- Check tracking-related counts in `detection_tracking_metrics.json` and `track_lifecycle_metrics.json`.
- Check total runtime in `summary.json`, `run_metadata.json`, and `ingestion_metrics.json`.
- Check `pipeline.log` and `errors/` for pipeline errors or tracebacks.

## Files To Inspect

- `summary.json`: high-level run outcome, processed-frame totals, device info, and output directory.
- `pipeline.log`: ordered runtime log for startup, progress, warnings, and failures.
- `detection_tracking_metrics.json`: model, device, inference timing, detection counts, tracking counts, and tracker setup.
- `bbox_quality_metrics.json`: accepted and rejected detections with rejection reasons.
- `tracks.json`: finalized local track summaries and evidence references.
- `observations.csv`: per-frame tracked observations for each local track.
- `track_lifecycle_metrics.json`: lifecycle counts, final-class stats, and shutdown state.
- `evidence_index.json`: evidence-image index by track and role.
- `evidence_metrics.json`: evidence collection totals and saved-file counts.
- `detected_frames/`: sampled YOLO-annotated frames for detection review.
- `tracked_frames/`: sampled tracking-annotated frames for ID continuity review.
- `evidence/`: per-track crops and annotated evidence images for qualitative inspection.
