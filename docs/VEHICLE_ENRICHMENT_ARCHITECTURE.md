# Vehicle Enrichment Architecture

## Current Detector/Tracker Flow

The clean pipeline keeps the existing OCR_MUKUL detector and per-camera Supervision ByteTrack flow unchanged:

1. `MultiCameraIngestionManager` produces `FramePacket` items.
2. `VehicleDetectorTracker.process_frame(...)` performs detection and tracking.
3. `TrackManager.update_frame(...)` manages per-camera lifecycle state.
4. `EvidenceCollector.register_frame(...)` caches candidate evidence while tracks are active.
5. `EvidenceCollector.finalize_tracks(...)` finalizes evidence only after track completion or flush.
6. `RunOutputManager` writes `tracks.json`, `observations.csv`, evidence files, and metrics.

## Existing Evidence Reuse

Step 2 does not introduce a second evidence collector.

Instead, `src/vehicle_enrichment/evidence_adapter.py` adapts finalized evidence records already produced by `src/evidence.py` into `EnrichmentEvidenceItem` records.

Reuse strategy:

- Prefer existing evidence crop paths when they already exist.
- Fall back to extracting a crop from the finalized evidence image only when a reusable crop is missing.
- Keep the original evidence records authoritative for the track.

## Enrichment Trigger

Vehicle enrichment runs only after the existing evidence finalization call:

1. `track_manager.update_frame(...)`
2. `evidence_collector.finalize_tracks(completed_now)`
3. `vehicle_enrichment_manager.enrich_completed_tracks(completed_now, finalized_evidence_now)`

The same sequence is used during the end-of-stream flush.

This keeps enrichment off the hot per-frame path and prevents detector/tracker behavior changes.

## Package Responsibilities

- `schemas.py`: typed request/result dataclasses and serializer helpers
- `evidence_adapter.py`: convert finalized evidence into enrichment-ready items
- `evidence_quality.py`: deterministic crop-quality scoring
- `attribute_aggregator.py`: shared helper for later attribute voting
- `enrichment_manager.py`: orchestration, selection, fail-open behavior, metrics
- `shared/florence_backend.py`: shared Florence loader, lazy lifecycle, task execution, and runtime metrics
- `body_type/classifier.py`: real Step 3 CAR-only body-type inference plus deterministic label aggregation
- `colour/`, `make_model/`, `plate/`: still disabled placeholders in this step

## Crop Selection Formula

The enrichment quality score is:

```text
score =
    area_weight * normalized_area_score
  + sharpness_weight * normalized_sharpness_score
  + confidence_weight * detection_confidence
  + role_weight * evidence_role_score
  - border_weight * border_penalty
  - clipping_weight * clipping_ratio
  - brightness_weight * brightness_penalty
```

Normalization details:

- `normalized_area_score` is capped to `[0, 1]` relative to the configured minimum crop area.
- `normalized_sharpness_score` is capped to `[0, 1]` relative to `4x` the minimum sharpness.
- `brightness_penalty` is `0` inside `[40, 215]` and rises toward `1` outside that band.
- The final score is clamped to `[0, 1]`.

## Fail-Open Behavior

- Enrichment errors do not fail the main pipeline when `vehicle_enrichment.fail_open: true`.
- Disabled modules return structured `disabled` or `not_run` results.
- Missing or weak evidence yields `no_evidence`, `UNKNOWN`, or `null` rather than guessed values.
- Florence body-type failures resolve to structured `UNKNOWN` predictions rather than crashing tracking.

## Output Structure

New run outputs:

- `vehicle_enrichment.json`
- `vehicle_enrichment_metrics.json`
- `vehicle_enrichment/crops/<TRACK_ID>/...`

`tracks.json` remains backward-compatible. Existing fields are unchanged, and an optional nested `vehicle_enrichment` object is appended only when configured.

## Step 3 Florence Body-Type Flow

- Shared backend:
  - base model: `microsoft/Florence-2-base-ft`
  - adapter path: `C:/Mukul K/OCR_MUKUL/adaptor_florance_baseFT`
  - default body-type run mode: adapter configured, but inactive
  - task prompt: `<VQA>`
- OCR_MUKUL logic reused:
  - BGR OpenCV crop to RGB PIL image
  - prompt assembly as `task_prompt + text_input`
  - processor tensors returned as PyTorch tensors
  - `model.generate(...)`
  - `processor.batch_decode(...)`
  - `processor.post_process_generation(...)`
- Runtime safeguards added around that flow:
  - lazy one-time model load
  - compatibility shims for the Florence remote config and tokenizer under the installed `transformers` build
  - square padding for non-square evidence crops before Florence processing

## Step 3 Body-Type Rules

- Only `CAR` tracks are eligible.
- The top `maximum_crops_per_track` evidence crops are used after evidence-quality ranking.
- Crops below the configured minimum width or height are skipped.
- Florence outputs are normalized to the allowed labels:
  - `SUV`
  - `SEDAN`
  - `HATCHBACK`
  - `MPV`
  - `VAN`
  - `PICKUP`
  - `OTHER`
  - `UNKNOWN`
- If Florence returns free-form text that does not normalize to one of those labels, the crop result is recorded as `UNKNOWN` with reason `unexpected_output`.
- Final per-track aggregation is deterministic:
  - single valid label: accept it
  - strong weighted majority: accept it
  - conflicting high-quality labels: return `UNKNOWN`
  - no valid normalized labels: return `UNKNOWN`
