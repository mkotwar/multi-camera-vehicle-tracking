# Current Active Scope

## Active Runtime Files

- `app.py`
  - Purpose: CLI entry point
  - Called by: direct user execution
  - Important runtime function/class: `main()`
  - Status: ACTIVE

- `src/pipeline.py`
  - Purpose: validated config loading, runtime orchestration, summary/output writing
  - Called by: `app.py`
  - Important runtime function/class: `run_pipeline()`
  - Status: ACTIVE

- `src/ingestion_manager.py`
  - Purpose: multi-camera reader workers and frame queue
  - Called by: `src/pipeline.py`
  - Important runtime function/class: `MultiCameraIngestionManager`
  - Status: ACTIVE

- `src/camera_reader.py`
  - Purpose: per-camera frame reading from video / RTSP / webcam
  - Called by: `src/ingestion_manager.py`
  - Important runtime function/class: `VideoCameraReader`
  - Status: ACTIVE

- `src/detector_tracker.py`
  - Purpose: shared YOLO detection path plus per-camera tracker creation
  - Called by: `src/pipeline.py`
  - Important runtime function/class: `VehicleDetectorTracker`
  - Status: ACTIVE

- `src/track_manager.py`
  - Purpose: camera-local track lifecycle and local track IDs
  - Called by: `src/pipeline.py`
  - Important runtime function/class: `TrackManager`
  - Status: ACTIVE

- `src/evidence.py`
  - Purpose: track/evidence crop collection and selection inputs
  - Called by: `src/pipeline.py`
  - Important runtime function/class: `EvidenceCollector`
  - Status: ACTIVE

- `src/vehicle_enrichment/enrichment_manager.py`
  - Purpose: runtime enrichment routing, colour-only active branch, frozen branch gating
  - Called by: `src/pipeline.py`
  - Important runtime function/class: `VehicleEnrichmentManager`
  - Status: ACTIVE

- `src/vehicle_enrichment/vehicle_attribute_flow.py`
  - Purpose: base Florence colour inference flow
  - Called by: `src/vehicle_enrichment/enrichment_manager.py`
  - Important runtime function/class: `BaseFlorenceVehicleAttributesFlow`
  - Status: ACTIVE

- `src/vehicle_enrichment/colour/classifier.py`
  - Purpose: colour normalization rules and Florence colour prompts
  - Called by: `src/vehicle_enrichment/enrichment_manager.py` and `src/vehicle_enrichment/vehicle_attribute_flow.py`
  - Important runtime function/class: `VehicleColourClassifier`
  - Status: ACTIVE

- `src/vehicle_enrichment/colour/search_aliases.py`
  - Purpose: search label expansion including `RED -> RED + PINK`
  - Called by: colour consumers/tests
  - Important runtime function/class: `expand_colour_search_labels()`
  - Status: ACTIVE

- `src/vehicle_enrichment/evidence_quality.py`
  - Purpose: evidence scoring and crop selection quality logic
  - Called by: `src/vehicle_enrichment/enrichment_manager.py`
  - Important runtime function/class: `EvidenceQualityEvaluator`
  - Status: ACTIVE

- `src/output_writer.py`
  - Purpose: camera-wise outputs, diagnostics, summaries, CSV/JSON artifacts
  - Called by: `src/pipeline.py`
  - Important runtime function/class: `RunOutputManager`
  - Status: ACTIVE

- `config.yaml`
  - Purpose: default active runtime config
  - Called by: `app.py` when `--config` is omitted
  - Important runtime function/class: colour-only frozen-scope configuration
  - Status: ACTIVE

- `config.validation_multicamera_colour.yaml`
  - Purpose: clean two-camera colour-only validation config
  - Called by: `app.py --config ...`
  - Important runtime function/class: colour-only validation config
  - Status: ACTIVE

## Frozen Files / Modules

- `src/vehicle_enrichment/body_type/classifier.py`
  - Purpose: Florence body-type inference
  - Why frozen: out of current production-validation scope
  - How it is disabled: `vehicle_enrichment.vehicle_attributes.body_type.enabled: false` and `vehicle_enrichment.body_type.enabled: false`
  - Status: FROZEN

- `src/vehicle_enrichment/body_type/labels.py`
  - Purpose: body-type labels and prompt defaults
  - Why frozen: body type excluded from current phase
  - How it is disabled: body-type runtime flags disabled
  - Status: FROZEN

- `src/vehicle_enrichment/ocr_mukul/backend.py`
  - Purpose: OCR_MUKUL Florence adapter attribute path
  - Why frozen: OCR adapter path excluded from current phase
  - How it is disabled: `shared_florence.adapter_enabled: false`, `plate.ocr.enabled: false`, `ocr.enabled: false`
  - Status: FROZEN

- `src/vehicle_enrichment/plate/detector.py`
  - Purpose: plate detection
  - Why frozen: plate pipeline excluded from current phase
  - How it is disabled: `vehicle_enrichment.plate.detection_enabled: false`
  - Status: FROZEN

- `src/vehicle_enrichment/plate/quality_validator.py`
  - Purpose: plate quality filtering
  - Why frozen: plate pipeline excluded from current phase
  - How it is disabled: plate detector remains disabled
  - Status: FROZEN

- `src/vehicle_enrichment/plate/ocr_engine.py`
  - Purpose: OCR_MUKUL-based plate OCR
  - Why frozen: OCR excluded from current phase
  - How it is disabled: `vehicle_enrichment.plate.ocr.enabled: false`
  - Status: FROZEN

- `src/vehicle_enrichment/make_model/classifier.py`
  - Purpose: make/model inference
  - Why frozen: make/model excluded from current phase
  - How it is disabled: `vehicle_enrichment.make_model.enabled: false`
  - Status: FROZEN

- `src/global_association.py`
  - Purpose: future cross-camera global association
  - Why frozen: no global tracking in current scope
  - How it is disabled: not imported by active runtime path
  - Status: FROZEN

- `diagnostics/`
  - Purpose: historical benchmark and validation artifacts
  - Why frozen: retained for later development/reference
  - How it is disabled: not part of active runtime path
  - Status: FROZEN

## Not Verified / Unused / Legacy

- `src/vehicle_enrichment/legacy_florence/`
  - Purpose: older Florence flows and adapters
  - Why marked this way: present for comparison/legacy support, not part of default active config
  - How it is disabled: not on the default active runtime path
  - Status: NOT VERIFIED

- `src/vehicle_enrichment/benchmarking/`
  - Purpose: offline benchmarking helpers
  - Why marked this way: utility scripts, not used by default pipeline execution
  - How it is disabled: not imported by `app.py -> src/pipeline.py`
  - Status: NOT VERIFIED

- `config.validation_car_body_type.yaml`
  - Purpose: body-type validation config
  - Why marked this way: specialized validation config outside the frozen active scope
  - How it is disabled: not the default active config
  - Status: NOT VERIFIED

## Active Pipeline Flow

```text
CAM_001 reader worker ─┐
CAM_002 reader worker ─┼→ shared detection path
CAM_003 reader worker ─┘
                              ↓
                             YOLO
                              ↓
                       camera_id router
                      /       |       \
                 CAM_001   CAM_002   CAM_003
                  tracker   tracker   tracker
                      \       |       /
                       track evidence
                              ↓
                     Florence colour
                              ↓
                     camera-wise output
```

Current scope explicitly excludes:

- NO global tracking
- NO cross-camera identity matching
- NO body type
- NO plate detection
- NO plate OCR
- NO make/model
