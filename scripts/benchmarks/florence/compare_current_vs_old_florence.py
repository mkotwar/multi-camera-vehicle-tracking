from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vehicle_enrichment.body_type.classifier import (
    BODY_TYPE_PROMPT_TEXT,
    BODY_TYPE_TASK_PROMPT,
    VehicleBodyTypeClassifier,
)
from src.vehicle_enrichment.colour.classifier import VehicleColourClassifier
from src.vehicle_enrichment.enrichment_manager import normalize_vehicle_enrichment_config
from src.vehicle_enrichment.legacy_florence import OldTdCase2Adapter, inspect_old_reference_project
from src.vehicle_enrichment.schemas import EnrichmentEvidenceItem, TrackEnrichmentRequest
from src.vehicle_enrichment.shared.florence_backend import FlorenceBackend, FlorenceBackendConfig


def _load_run_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_run_config(input_run: Path) -> dict[str, Any]:
    config_path = input_run / "run_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _make_request(result: dict[str, Any], evidence_items: list[EnrichmentEvidenceItem]) -> TrackEnrichmentRequest:
    return TrackEnrichmentRequest(
        local_track_id=str(result["local_track_id"]),
        camera_id=str(result["camera_id"]),
        native_tracker_id=0,
        vehicle_class=str(result.get("vehicle_class", "UNKNOWN")),
        vehicle_class_confidence=result.get("vehicle_class_confidence"),
        track_status=str(result.get("status", "completed")),
        completion_reason=None,
        started_at_seconds=0.0,
        ended_at_seconds=0.0,
        evidence_items=evidence_items,
    )


def _item_from_dict(payload: dict[str, Any]) -> EnrichmentEvidenceItem:
    return EnrichmentEvidenceItem(
        local_track_id=str(payload["local_track_id"]),
        camera_id=str(payload["camera_id"]),
        native_tracker_id=int(payload.get("native_tracker_id", 0) or 0),
        frame_number=int(payload.get("frame_number", 0) or 0),
        timestamp_seconds=float(payload.get("timestamp_seconds", 0.0) or 0.0),
        source_image_path=payload.get("source_image_path"),
        vehicle_crop_path=payload.get("vehicle_crop_path"),
        annotated_frame_path=payload.get("annotated_frame_path"),
        bbox_xyxy=tuple(float(v) for v in payload.get("bbox_xyxy", (0, 0, 0, 0))),
        evidence_role=str(payload.get("evidence_role", "UNKNOWN")),
        detection_confidence=float(payload.get("detection_confidence", 0.0) or 0.0),
        crop_width=int(payload.get("crop_width", 0) or 0),
        crop_height=int(payload.get("crop_height", 0) or 0),
        crop_area=int(payload.get("crop_area", 0) or 0),
        sharpness_score=float(payload.get("sharpness_score", 0.0) or 0.0),
        brightness_score=float(payload.get("brightness_score", 0.0) or 0.0),
        border_penalty=float(payload.get("border_penalty", 0.0) or 0.0),
        clipping_ratio=float(payload.get("clipping_ratio", 0.0) or 0.0),
        quality_score=float(payload.get("quality_score", 0.0) or 0.0),
        original_bbox_xyxy=tuple(float(v) for v in payload.get("original_bbox_xyxy", (0, 0, 0, 0))),
        expanded_crop_bbox_xyxy=tuple(float(v) for v in payload.get("expanded_crop_bbox_xyxy", (0, 0, 0, 0))),
        source_frame_width=int(payload.get("source_frame_width", 0) or 0),
        source_frame_height=int(payload.get("source_frame_height", 0) or 0),
        context_padding_ratio=float(payload.get("context_padding_ratio", 0.0) or 0.0),
        original_crop_width=int(payload.get("original_crop_width", 0) or 0),
        original_crop_height=int(payload.get("original_crop_height", 0) or 0),
        candidate_rank=payload.get("candidate_rank"),
        candidate_retained=bool(payload.get("candidate_retained", True)),
        candidate_rejection_reason=payload.get("candidate_rejection_reason"),
        frame_gap_from_previous_selected=payload.get("frame_gap_from_previous_selected"),
        duplicate_score=payload.get("duplicate_score"),
        resolution_tier=str(payload.get("resolution_tier", "below_minimum")),
        florence_eligible_for_body_type=bool(payload.get("florence_eligible_for_body_type", False)),
        florence_eligible_for_colour=bool(payload.get("florence_eligible_for_colour", False)),
        florence_body_type_skip_reason=payload.get("florence_body_type_skip_reason"),
        florence_colour_skip_reason=payload.get("florence_colour_skip_reason"),
        edge_truncated=bool(payload.get("edge_truncated", False)),
        ranking_score=float(payload.get("ranking_score", 0.0) or 0.0),
        selected_for_body_type=bool(payload.get("selected_for_body_type", False)),
        selected_for_colour=bool(payload.get("selected_for_colour", False)),
        body_type_crop_result=payload.get("body_type_crop_result"),
        colour_crop_result=payload.get("colour_crop_result"),
        rejection_reasons=list(payload.get("rejection_reasons", []) or []),
    )


def _save_visual_panel(
    output_path: Path,
    item: EnrichmentEvidenceItem,
    current_preprocessed: np.ndarray | None,
    old_preprocessed: np.ndarray | None,
) -> None:
    source = None
    if item.source_image_path and Path(item.source_image_path).exists():
        source = cv2.imread(str(item.source_image_path))
    current_crop = cv2.imread(str(item.vehicle_crop_path)) if item.vehicle_crop_path and Path(item.vehicle_crop_path).exists() else None
    old_crop = current_crop
    canvas_images: list[np.ndarray] = []
    for image in [source, current_crop, old_crop, current_preprocessed, old_preprocessed]:
        if image is None or image.size == 0:
            tile = np.full((240, 320, 3), 235, dtype=np.uint8)
        else:
            tile = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
        canvas_images.append(tile)
    panel = np.hstack(canvas_images)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), panel)


def _summarize(rows: list[dict[str, Any]], backend: FlorenceBackend) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    def _avg(key: str) -> float:
        values = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
        return round(statistics.fmean(values), 4) if values else 0.0

    def _rate(key: str, expected: str) -> float:
        if not rows:
            return 0.0
        return round(100.0 * sum(1 for row in rows if str(row.get(key, "")).upper() == expected) / len(rows), 2)

    summary = {
        "number_of_crops_tested": len(rows),
        "average_current_crop_width": _avg("current_original_crop_width"),
        "average_current_crop_height": _avg("current_original_crop_height"),
        "average_old_crop_width": _avg("old_original_crop_width"),
        "average_old_crop_height": _avg("old_original_crop_height"),
        "percentage_where_old_crop_is_larger": round(
            100.0 * sum(
                1
                for row in rows
                if (int(row.get("old_original_crop_width", 0) or 0) * int(row.get("old_original_crop_height", 0) or 0))
                > (int(row.get("current_original_crop_width", 0) or 0) * int(row.get("current_original_crop_height", 0) or 0))
            ) / max(1, len(rows)),
            2,
        ),
        "percentage_where_current_crop_is_larger": round(
            100.0 * sum(
                1
                for row in rows
                if (int(row.get("current_original_crop_width", 0) or 0) * int(row.get("current_original_crop_height", 0) or 0))
                > (int(row.get("old_original_crop_width", 0) or 0) * int(row.get("old_original_crop_height", 0) or 0))
            ) / max(1, len(rows)),
            2,
        ),
        "current_valid_body_type_rate": round(100.0 - _rate("current_body_type_label", "UNKNOWN"), 2),
        "old_valid_body_type_rate": round(100.0 - _rate("old_body_type_label", "UNKNOWN"), 2),
        "current_body_type_unknown_rate": _rate("current_body_type_label", "UNKNOWN"),
        "old_body_type_unknown_rate": _rate("old_body_type_label", "UNKNOWN"),
        "current_valid_colour_rate": round(100.0 - _rate("current_colour_label", "UNKNOWN"), 2),
        "old_valid_colour_rate": round(100.0 - _rate("old_colour_label", "UNKNOWN"), 2),
        "current_colour_unknown_rate": _rate("current_colour_label", "UNKNOWN"),
        "old_colour_unknown_rate": _rate("old_colour_label", "UNKNOWN"),
        "generic_response_count": sum(
            1
            for row in rows
            if str(row.get("old_colour_raw_response", "")).strip().lower() in {"", "vehicle", "answer", "car"}
        ),
        "prompt_echo_count": sum(
            1
            for row in rows
            if str(row.get("old_body_type_raw_response", "")).strip() == str(row.get("old_body_type_prompt", "")).strip()
        ),
        "conflicting_response_count": sum(
            1
            for row in rows
            if str(row.get("current_body_type_label", "")) != str(row.get("old_body_type_label", ""))
            or str(row.get("current_colour_label", "")) != str(row.get("old_colour_label", ""))
        ),
        "average_current_inference_time_ms": _avg("current_inference_time_ms"),
        "average_old_inference_time_ms": _avg("old_inference_time_ms"),
        "florence_load_count": int(backend.metrics.get("florence_load_successes", 0)),
    }
    summary_rows = [{"metric": key, "value": value} for key, value in summary.items()]
    return summary, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current Florence flow against the old td_case2 flow.")
    parser.add_argument("--current-project", default=".")
    parser.add_argument("--old-project", required=True)
    parser.add_argument("--input-run", required=True)
    parser.add_argument("--attribute", choices=["body_type", "colour", "both"], default="both")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_run = Path(args.input_run).resolve()
    output_dir = _ensure_dir(Path(args.output_dir).resolve())
    visual_dir = _ensure_dir(output_dir / "visual_comparisons")

    old_reference = inspect_old_reference_project(args.old_project)
    run_config = _read_run_config(input_run)
    enrichment_config = normalize_vehicle_enrichment_config(run_config.get("vehicle_enrichment", {}))
    backend = FlorenceBackend(FlorenceBackendConfig(**enrichment_config["shared_florence"]))
    legacy = OldTdCase2Adapter(backend)
    body_classifier = VehicleBodyTypeClassifier(enrichment_config["body_type"], backend=backend, logger=__import__("logging").getLogger(__name__))
    colour_classifier = VehicleColourClassifier(enrichment_config["colour"], backend=backend, logger=__import__("logging").getLogger(__name__))

    enrichment_results = _load_run_json(input_run / "vehicle_enrichment.json")
    comparison_rows: list[dict[str, Any]] = []
    hypothesis_notes: list[dict[str, Any]] = []

    for result in enrichment_results:
        evidence_items = [_item_from_dict(item) for item in list(result.get("evidence_used", []) or [])]
        if not evidence_items:
            continue
        legacy_selection = legacy.select_track_evidence(evidence_items, maximum_crops_per_track=3, minimum_frame_gap=3)
        current_request = _make_request(result, evidence_items)
        for item in evidence_items:
            single_request = _make_request(result, [item])
            current_body = body_classifier.classify(single_request)
            current_colour = colour_classifier.classify(single_request)
            legacy_result = legacy.run_caption_inference(str(item.vehicle_crop_path))

            current_body_prediction = current_body.predictions[0] if current_body.predictions else None
            current_colour_prediction = current_colour.predictions[0] if current_colour.predictions else None

            current_preprocessed = None
            if item.vehicle_crop_path and Path(item.vehicle_crop_path).exists():
                crop_image = cv2.imread(str(item.vehicle_crop_path))
                if crop_image is not None and crop_image.size > 0:
                    current_preprocessed = crop_image.copy()
                    if current_body_prediction and current_body_prediction.square_padding_applied:
                        from src.vehicle_enrichment.body_type.classifier import pad_to_square as _pad_square
                        from PIL import Image
                        padded = _pad_square(Image.fromarray(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)), (114, 114, 114))
                        current_preprocessed = cv2.cvtColor(np.array(padded), cv2.COLOR_RGB2BGR)
            old_preprocessed = cv2.imread(str(item.vehicle_crop_path)) if item.vehicle_crop_path and Path(item.vehicle_crop_path).exists() else None

            panel_name = f"{result['camera_id']}_{str(result['local_track_id']).replace(':', '_')}_{int(item.frame_number):06d}.jpg"
            _save_visual_panel(visual_dir / panel_name, item, current_preprocessed, old_preprocessed)

            comparison_rows.append(
                {
                    "camera_id": result["camera_id"],
                    "local_track_id": result["local_track_id"],
                    "frame_index": item.frame_number,
                    "vehicle_class": result.get("vehicle_class", "UNKNOWN"),
                    "source_frame_width": item.source_frame_width,
                    "source_frame_height": item.source_frame_height,
                    "current_original_crop_path": item.vehicle_crop_path,
                    "current_original_crop_width": item.original_crop_width,
                    "current_original_crop_height": item.original_crop_height,
                    "current_context_padding": item.context_padding_ratio,
                    "current_preprocessed_width": current_body_prediction.florence_input_width if current_body_prediction else "",
                    "current_preprocessed_height": current_body_prediction.florence_input_height if current_body_prediction else "",
                    "current_tensor_shape": json.dumps([current_body_prediction.florence_input_height, current_body_prediction.florence_input_width]) if current_body_prediction and current_body_prediction.florence_input_width and current_body_prediction.florence_input_height else "",
                    "old_crop_path": item.vehicle_crop_path,
                    "old_original_crop_width": legacy_result.original_width,
                    "old_original_crop_height": legacy_result.original_height,
                    "old_context_padding": item.context_padding_ratio,
                    "old_preprocessed_width": legacy_result.preprocessed_width,
                    "old_preprocessed_height": legacy_result.preprocessed_height,
                    "old_tensor_shape": json.dumps(legacy_result.tensor_shape),
                    "current_body_type_prompt": BODY_TYPE_PROMPT_TEXT,
                    "old_body_type_prompt": legacy_result.task_prompt,
                    "current_body_type_raw_response": current_body_prediction.raw_response if current_body_prediction else "",
                    "old_body_type_raw_response": legacy_result.raw_response,
                    "current_body_type_label": current_body.label,
                    "old_body_type_label": legacy_result.body_type_label,
                    "current_colour_prompt": colour_classifier.primary_prompt_variant["prompt_text"],
                    "old_colour_prompt": legacy_result.task_prompt,
                    "current_colour_raw_response": current_colour_prediction.raw_response if current_colour_prediction else "",
                    "old_colour_raw_response": legacy_result.raw_response,
                    "current_colour_label": current_colour.label,
                    "old_colour_label": legacy_result.colour_label,
                    "current_inference_time_ms": current_body_prediction.inference_duration_ms if current_body_prediction else (current_colour_prediction.inference_duration_ms if current_colour_prediction else ""),
                    "old_inference_time_ms": legacy_result.inference_time_ms,
                    "manual_body_type": "",
                    "manual_colour": "",
                    "current_body_type_correct": "",
                    "old_body_type_correct": "",
                    "current_colour_correct": "",
                    "old_colour_correct": "",
                    "review_notes": "",
                }
            )

        hypothesis_notes.append(
            {
                "local_track_id": result["local_track_id"],
                "current_selected_body_type_crop_paths": list(result.get("selected_body_type_crop_paths", []) or []),
                "current_selected_colour_crop_paths": list(result.get("selected_colour_crop_paths", []) or []),
                "old_selected_crop_paths": legacy_selection.selected_crop_paths,
            }
        )

    csv_path = output_dir / "comparison.csv"
    if comparison_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
            writer.writeheader()
            writer.writerows(comparison_rows)

    summary, summary_rows = _summarize(comparison_rows, backend)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(summary_rows)

    report_lines = [
        "# Florence Old vs Current Report",
        "",
        f"- Old reference root: `{old_reference['old_project_root']}`",
        f"- Crops tested: `{summary['number_of_crops_tested']}`",
        f"- Florence load count: `{summary['florence_load_count']}`",
        "- Manual review status: incomplete",
        "",
        "## Old reference files",
    ]
    report_lines.extend([f"- `{path}`" for path in old_reference["found_files"]])
    report_lines.extend(
        [
            "",
            "## Hypotheses",
            "- Hypothesis A (larger crop/context): partially supported only when old track selection picks a different crop; old preprocessing itself does not enlarge the selected crop.",
            "- Hypothesis B (full frame instead of crop): not supported for Step 06; Step 06 uses selected vehicle crop paths, not full-scene frames.",
            "- Hypothesis C (later or better frame): partially supported; old Step 05 scoring can choose a different ranked crop than the current path.",
            "- Hypothesis D (different resize before Florence): supported; old path performs no manual square padding before the processor.",
            "- Hypothesis E (no square padding): supported; old path passes the rectangular crop directly.",
            "- Hypothesis F (different task token/prompt): supported; old path uses `<CAPTION>` and parses attributes from the caption.",
            "- Hypothesis G (different parser): supported; old body type and colour come from caption parsing, not direct VQA label normalization.",
            "- Hypothesis H (better evidence vs better inference): insufficient evidence without manual labels.",
            "",
            "## Track-level selection notes",
            *[f"- `{item['local_track_id']}` current={item['current_selected_body_type_crop_paths']} old={item['old_selected_crop_paths']}" for item in hypothesis_notes[:20]],
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    analysis_json = {
        "old_reference": old_reference,
        "execution_path": {
            "step03_detection": "03_yolo_object_crops from original frame with 5% vehicle padding",
            "step05_selection": "scores detections and copies selected crops/full frames",
            "step06_inference": "runs <OCR> and <CAPTION> on selected crop path, parses colour/body_type from caption",
            "streaming_variant": "loads crop path, uses processor(text=prompt, images=PIL RGB), generate, post_process_generation",
        },
    }
    _ensure_dir(Path("outputs"))
    Path("outputs/old_florence_reference_analysis.json").write_text(json.dumps(analysis_json, indent=2), encoding="utf-8")
    Path("outputs/old_florence_reference_analysis.md").write_text(
        "\n".join(
            [
                "# Old Florence Reference Analysis",
                "",
                "## End-to-end path",
                "- Step 03 creates vehicle crops from original frame with 5% padding for vehicle-like classes.",
                "- Step 05 ranks and copies best selected track crops into `05_selected_track_crops`.",
                "- Step 06 reads `selected_crop_path` and runs Florence `<OCR>` and `<CAPTION>` on that crop.",
                "- Old body type and colour are parsed from the caption text, not from dedicated VQA prompts.",
                "",
                "## Key files",
                *[f"- `{path}`" for path in old_reference["found_files"]],
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
