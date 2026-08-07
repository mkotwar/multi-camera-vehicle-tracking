# Step 1 — Scope Freeze Plan

## Project Goal for Current Phase

Keep the active pipeline focused on:

**Multi-camera input → worker-based ingestion → shared YOLO vehicle detection → per-camera tracking → vehicle colour classification → camera-wise outputs/UI**

No global/cross-camera identity matching is required in this phase.

---

## ACTIVE / KEEP WORKING

These parts must remain active and unchanged unless Codex finds a real wiring issue.

### Core pipeline
- `app.py` — main pipeline entry point / runner (verify actual active entry point)
- `src/pipeline.py` — main orchestration pipeline
- camera ingestion / frame packet modules — keep active
- worker modules for camera reading / detection / tracking — keep active
- shared YOLO detector path — keep active
- per-camera tracker/router/lifecycle path — keep active
- crop/evidence collection — keep active because colour enrichment depends on it

### Colour enrichment
- `src/vehicle_enrichment/colour/classifier.py` — ACTIVE
- `src/vehicle_enrichment/colour/search_aliases.py` — ACTIVE
- `src/vehicle_enrichment/evidence_quality.py` — ACTIVE
- `src/vehicle_enrichment/evidence_adapter.py` — ACTIVE if used by current crop/evidence flow
- `src/vehicle_enrichment/attribute_aggregator.py` — ACTIVE if current colour aggregation uses it
- `src/vehicle_enrichment/enrichment_manager.py` — ACTIVE, but only colour should be enabled
- `src/vehicle_enrichment/vehicle_attribute_flow.py` — ACTIVE, but body-type inference must be disabled in runtime
- `src/vehicle_enrichment/schemas.py` — ACTIVE for colour/result schemas
- `src/output_writer.py` — ACTIVE for camera-wise outputs and diagnostics

### Florence settings
- Base Florence model path remains active for colour:
  - `D:/project/models/Florence-2-base-ft`
- OCR_MUKUL Florence adapter must remain disabled for colour.

### Colour behavior that must remain unchanged
- PINK is a valid normalized vehicle colour.
- RED search expansion must still include RED + PINK.
- Existing crop fallback / small readable crop behaviour for colour remains unchanged.
- Existing track-level colour aggregation remains unchanged.

---

## FROZEN / DISABLED FROM ACTIVE RUNTIME

Do **not delete** these files. Keep the code in the repository, but disable the features in the active config/runtime path.

### Body type
- `src/vehicle_enrichment/body_type/classifier.py` — FROZEN
- `src/vehicle_enrichment/body_type/labels.py` — FROZEN
- body-type prompt/config — FROZEN
- body-type benchmark scripts/diagnostics — KEEP AS REFERENCE, NOT ACTIVE RUNTIME

### Plate / OCR
- plate detector module(s) — FROZEN
- plate quality module(s) — FROZEN
- plate colour classifier — FROZEN
- plate OCR / Florence adapter OCR path — FROZEN
- OCR_MUKUL adapter remains available in repo but must not participate in the current colour-only runtime

### Make / model
- make-model classifier/module(s) — FROZEN

### Global / cross-camera identity
- global vehicle association — FROZEN
- association candidates/decisions — FROZEN
- global vehicle ID generation — FROZEN
- cross-camera ReID / matching logic — FROZEN

The system should **not** attempt to determine whether `CAM_001:TRACK_5` and `CAM_002:TRACK_9` are the same real-world vehicle.

---

## DESIRED ACTIVE OUTPUT PER LOCAL TRACK

Minimum active result:

```text
camera_id
track_id
vehicle_class
colour
first_seen
last_seen
```

Example:

```text
CAM_001
TRACK_12
CAR
WHITE
first_seen=...
last_seen=...
```

No active dependency on:

```text
body_type
plate
ocr
make_model
global_vehicle_id
cross_camera_match
```

---

## CONFIGURATION TARGET

Codex must inspect the actual active config and verify the exact keys. Conceptually, the effective runtime should be equivalent to:

```yaml
vehicle_enrichment:
  enabled: true

  vehicle_attributes:
    enabled: true
    colour:
      enabled: true
    body_type:
      enabled: false

  make_model:
    enabled: false

  plate:
    detection:
      enabled: false
    recognition:
      enabled: false
    colour:
      enabled: false
```

Any global-association / global-tracking feature should also be disabled.

**Important:** do not blindly add these keys if the project's real config schema differs. Codex must inspect the actual config classes/readers first.

---

## STEP 1 ACCEPTANCE CRITERIA

Step 1 is complete only when all of the following are true:

1. Multi-camera input remains supported.
2. Worker-based camera ingestion remains supported.
3. Shared YOLO detection remains active.
4. Tracker state remains separate per camera.
5. Colour enrichment remains active for tracked vehicles.
6. Body type is not called.
7. Plate/OCR is not called.
8. Make/model is not called.
9. Cross-camera/global association is not called.
10. Existing colour/PINK/search behaviour is unchanged.
11. No code is deleted solely because a feature is frozen.
12. A fresh validation run confirms only the intended modules execute.

---

## IMPORTANT NOTE

This file is a **scope-control document based on the current known project architecture**. The repository itself is not attached here, so exact runtime wiring and exact file names must be verified by Codex before making changes. Codex should update this inventory after tracing the actual active code path.
