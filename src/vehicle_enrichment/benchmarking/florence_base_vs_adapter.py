from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import statistics
import sys
from typing import Any

import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vehicle_enrichment.body_type.classifier import BODY_TYPE_PROMPT_TEXT, BODY_TYPE_TASK_PROMPT, VehicleBodyTypeClassifier
from src.vehicle_enrichment.colour.classifier import DEFAULT_COLOUR_PROMPT_ID, VehicleColourClassifier
from src.vehicle_enrichment.enrichment_manager import normalize_vehicle_enrichment_config
from src.vehicle_enrichment.legacy_florence import LEGACY_CAPTION_TASK_PROMPT, OldTdCase2Adapter
from src.vehicle_enrichment.schemas import EnrichmentEvidenceItem, TrackEnrichmentRequest
from src.vehicle_enrichment.shared.florence_backend import FlorenceBackend, FlorenceBackendConfig


CONFIG_AUDIT_PATHS = [
    "config.yaml",
    "config.validation_florence_body_type.yaml",
    "config.validation_florence_body_type_colour.yaml",
    "config.validation_florence_body_type_colour.balanced.yaml",
    "config.validation_florence_body_type_colour.strict.yaml",
]

CONFIGURATION_ORDER = [
    "base_current",
    "adapter_current",
    "base_caption",
    "adapter_caption",
]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return yaml.safe_load(text) or {}


def _load_run_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_run_config(input_run: Path) -> dict[str, Any]:
    config_path = input_run / "run_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    return _read_yaml(config_path)


def _sanitize_path(value: Any) -> str | None:
    if value in ("", None):
        return None
    return str(value).replace("\\", "/")


def _configuration_mode(shared_florence: dict[str, Any]) -> str:
    adapter_enabled = bool(shared_florence.get("adapter_enabled"))
    adapter_path = shared_florence.get("adapter_path")
    if adapter_enabled and adapter_path not in ("", None):
        return "base + adapter"
    return "base only"


def generate_configuration_audit(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    configs: list[dict[str, Any]] = []
    for relative_path in CONFIG_AUDIT_PATHS:
        path = repo_root / relative_path
        payload = _read_yaml(path)
        enrichment = payload.get("vehicle_enrichment", {})
        shared_florence = dict(enrichment.get("shared_florence", {}) or {})
        body_type = dict(enrichment.get("body_type", {}) or {})
        colour = dict(enrichment.get("colour", {}) or {})
        configs.append(
            {
                "config_path": relative_path,
                "base_model_id": _sanitize_path(shared_florence.get("base_model_id")),
                "processor_path": _sanitize_path(shared_florence.get("processor_path")),
                "adapter_path": _sanitize_path(shared_florence.get("adapter_path")),
                "florence_mode": _configuration_mode(shared_florence),
                "body_type_prompt_mode": "current_vqa" if bool(body_type.get("enabled")) else "disabled",
                "colour_prompt_mode": str(colour.get("prompt_variant", DEFAULT_COLOUR_PROMPT_ID)) if bool(colour.get("enabled")) else "disabled",
                "device": shared_florence.get("device"),
                "dtype": shared_florence.get("dtype"),
            }
        )

    payload = {"generated_at": None, "configs": configs}
    _ensure_dir(output_dir)
    (output_dir / "florence_model_configuration_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Florence Model Configuration Audit",
        "",
        "| Config | Florence mode | Base model | Processor | Adapter | Body type | Colour | Device | Dtype |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in configs:
        lines.append(
            "| {config_path} | {florence_mode} | `{base_model_id}` | `{processor_path}` | `{adapter_path}` | {body_type_prompt_mode} | {colour_prompt_mode} | {device} | {dtype} |".format(
                **item
            )
        )
    (output_dir / "florence_model_configuration_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def inspect_model_assets(base_model_path: str, processor_path: str, adapter_path: str | None) -> dict[str, Any]:
    base_dir = Path(base_model_path)
    processor_dir = Path(processor_path)
    adapter_dir = Path(adapter_path) if adapter_path else None
    base_files = sorted(path.name for path in base_dir.iterdir()) if base_dir.exists() else []
    processor_files = sorted(path.name for path in processor_dir.iterdir()) if processor_dir.exists() else []
    adapter_files = sorted(path.name for path in adapter_dir.iterdir()) if adapter_dir and adapter_dir.exists() else []
    adapter_config = {}
    if adapter_dir is not None:
        adapter_config_path = adapter_dir / "adapter_config.json"
        if adapter_config_path.exists():
            adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    return {
        "base_model_path": _sanitize_path(base_model_path),
        "processor_path": _sanitize_path(processor_path),
        "adapter_path": _sanitize_path(adapter_path),
        "base_model_files": base_files,
        "processor_files": processor_files,
        "adapter_files": adapter_files,
        "adapter_type": adapter_config.get("peft_type"),
        "adapter_target_modules": list(adapter_config.get("target_modules", []) or []),
        "adapter_base_model_reference": adapter_config.get("base_model_name_or_path"),
        "adapter_rank": adapter_config.get("r"),
        "adapter_alpha": adapter_config.get("lora_alpha"),
        "adapter_dropout": adapter_config.get("lora_dropout"),
        "adapter_task_type": adapter_config.get("task_type"),
    }


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


def load_crop_manifest(input_run: Path, manifest_csv: Path | None = None) -> list[dict[str, Any]]:
    if manifest_csv is not None and manifest_csv.exists():
        with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    enrichment_results = _load_run_json(input_run / "vehicle_enrichment.json")
    rows: list[dict[str, Any]] = []
    for result in enrichment_results:
        for item in list(result.get("evidence_used", []) or []):
            evidence = _item_from_dict(item)
            if not evidence.vehicle_crop_path:
                continue
            rows.append(
                {
                    "camera_id": result["camera_id"],
                    "local_track_id": result["local_track_id"],
                    "frame_index": str(evidence.frame_number),
                    "crop_path": str(evidence.vehicle_crop_path),
                    "vehicle_class": str(result.get("vehicle_class", "UNKNOWN")),
                    "original_crop_width": str(evidence.original_crop_width),
                    "original_crop_height": str(evidence.original_crop_height),
                    "resolution_tier": evidence.resolution_tier,
                    "quality_score": str(evidence.quality_score),
                }
            )
    return rows


def _crop_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["camera_id"]),
        str(row["local_track_id"]),
        str(row["frame_index"]),
        str(row.get("crop_path") or row.get("current_original_crop_path") or row.get("vehicle_crop_path")),
    )


def _manifest_index(input_run: Path) -> dict[tuple[str, str, str, str], tuple[dict[str, Any], EnrichmentEvidenceItem]]:
    enrichment_results = _load_run_json(input_run / "vehicle_enrichment.json")
    indexed: dict[tuple[str, str, str, str], tuple[dict[str, Any], EnrichmentEvidenceItem]] = {}
    for result in enrichment_results:
        for item in list(result.get("evidence_used", []) or []):
            evidence = _item_from_dict(item)
            key = (
                str(result["camera_id"]),
                str(result["local_track_id"]),
                str(evidence.frame_number),
                str(evidence.vehicle_crop_path),
            )
            indexed[key] = (result, evidence)
    return indexed


def _fingerprint_image(image: Any) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()[:16]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _generic_response(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in {"", "vehicle", "car", "answer", "colour", "color"}


def _prompt_echo(raw_value: str, prompt_value: str) -> bool:
    return str(raw_value or "").strip().lower() == str(prompt_value or "").strip().lower()


def _build_backend(
    *,
    base_model: str,
    processor_path: str,
    adapter_path: str | None,
    adapter_enabled: bool,
    template_config: dict[str, Any],
) -> FlorenceBackend:
    shared = dict(template_config)
    shared["base_model_id"] = base_model
    shared["processor_path"] = processor_path
    shared["adapter_path"] = adapter_path
    shared["adapter_enabled"] = adapter_enabled
    backend = FlorenceBackend(FlorenceBackendConfig(**shared))
    backend.load()
    if adapter_enabled and not backend.adapter_active:
        raise RuntimeError("Adapter benchmark requested adapter mode, but the adapter did not load.")
    if not adapter_enabled and backend.adapter_active:
        raise RuntimeError("Base-only benchmark unexpectedly loaded an adapter.")
    return backend


def _config_generation_settings(backend: FlorenceBackend, task_token: str, prompt: str) -> dict[str, Any]:
    return {
        "task_token": task_token,
        "full_prompt": prompt,
        "max_new_tokens": backend.config.max_new_tokens,
        "num_beams": backend.config.num_beams,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "repetition_penalty": None,
        "early_stopping": None,
    }


def _prepare_current_metadata(body_classifier: VehicleBodyTypeClassifier, colour_classifier: VehicleColourClassifier, image: Any) -> dict[str, Any]:
    prepared_image, body_padding = body_classifier._prepare_image_for_florence(image)
    colour_prepared, colour_padding = colour_classifier._prepare_image_for_florence(image)
    if _fingerprint_image(prepared_image) != _fingerprint_image(colour_prepared):
        raise RuntimeError("Body type and colour preprocessing diverged for the same crop.")
    return {
        "prepared_image": prepared_image,
        "square_padding_applied": bool(body_padding["square_padding_applied"]),
        "padded_width": int(body_padding["padded_width"]),
        "padded_height": int(body_padding["padded_height"]),
        "pixel_value_fingerprint": _fingerprint_image(prepared_image),
        "colour_square_padding_applied": bool(colour_padding["square_padding_applied"]),
    }


def _current_row(
    *,
    configuration: str,
    backend: FlorenceBackend,
    body_classifier: VehicleBodyTypeClassifier,
    colour_classifier: VehicleColourClassifier,
    result: dict[str, Any],
    evidence: EnrichmentEvidenceItem,
    attribute: str,
) -> dict[str, Any]:
    image = cv2.imread(str(evidence.vehicle_crop_path))
    if image is None or image.size == 0:
        raise FileNotFoundError(f"Could not read crop image: {evidence.vehicle_crop_path}")
    request = _make_request(result, [evidence])
    preprocess = _prepare_current_metadata(body_classifier, colour_classifier, image)
    body_result = body_classifier.classify(request) if attribute in {"body_type", "both"} else None
    colour_result = colour_classifier.classify(request) if attribute in {"colour", "both"} else None
    body_prediction = body_result.predictions[0] if body_result and body_result.predictions else None
    colour_prediction = colour_result.predictions[0] if colour_result and colour_result.predictions else None
    payload = None
    if body_prediction is not None and isinstance(body_prediction.raw_response, str):
        payload = {"pixel_values_shape": [1, 3, int(preprocess["padded_height"]), int(preprocess["padded_width"])]}
    inference_time_ms = 0.0
    for prediction in [body_prediction, colour_prediction]:
        if prediction and prediction.inference_duration_ms:
            inference_time_ms = max(inference_time_ms, float(prediction.inference_duration_ms))
    body_raw = str(body_prediction.raw_response) if body_prediction and body_prediction.raw_response is not None else ""
    colour_raw = str(colour_prediction.raw_response) if colour_prediction and colour_prediction.raw_response is not None else ""
    generation = {
        "body_type": _config_generation_settings(backend, BODY_TYPE_TASK_PROMPT, BODY_TYPE_PROMPT_TEXT),
        "colour": _config_generation_settings(
            backend,
            str(colour_classifier.primary_prompt_variant["task_prompt"]),
            str(colour_classifier.primary_prompt_variant["prompt_text"]),
        ),
    }
    return {
        "camera_id": result["camera_id"],
        "local_track_id": result["local_track_id"],
        "frame_index": evidence.frame_number,
        "crop_path": str(evidence.vehicle_crop_path),
        "vehicle_class": result.get("vehicle_class", "UNKNOWN"),
        "original_crop_width": evidence.original_crop_width,
        "original_crop_height": evidence.original_crop_height,
        "resolution_tier": evidence.resolution_tier,
        "quality_score": evidence.quality_score,
        "configuration": configuration,
        "adapter_enabled": backend.config.adapter_enabled,
        "adapter_loaded": backend.adapter_active,
        "prompt_mode": "current",
        "task_token": BODY_TYPE_TASK_PROMPT if attribute != "colour" else str(colour_classifier.primary_prompt_variant["task_prompt"]),
        "prompt": BODY_TYPE_PROMPT_TEXT if attribute != "colour" else str(colour_classifier.primary_prompt_variant["prompt_text"]),
        "raw_response": _json({"body_type": body_raw, "colour": colour_raw}),
        "normalized_response": _json({"body_type": body_result.label if body_result else "", "colour": colour_result.label if colour_result else ""}),
        "body_type_label": body_result.label if body_result else "",
        "colour_label": colour_result.label if colour_result else "",
        "parse_status": _json({"body_type": body_prediction.status if body_prediction else "not_run", "colour": colour_prediction.status if colour_prediction else "not_run"}),
        "parse_reason": _json({"body_type": body_prediction.reason if body_prediction else "", "colour": colour_prediction.reason if colour_prediction else ""}),
        "generic_response": _json({"body_type": _generic_response(body_raw), "colour": _generic_response(colour_raw)}),
        "prompt_echo": _json(
            {
                "body_type": _prompt_echo(body_raw, BODY_TYPE_PROMPT_TEXT),
                "colour": _prompt_echo(colour_raw, str(colour_classifier.primary_prompt_variant["prompt_text"])),
            }
        ),
        "inference_time_ms": inference_time_ms,
        "peak_gpu_memory_mb": backend.metrics.get("gpu_memory_allocated_mb"),
        "processor_path": backend.processor_identifier,
        "device": backend.resolved_device,
        "dtype": backend.resolved_dtype,
        "square_padding_applied": preprocess["square_padding_applied"],
        "padded_width": preprocess["padded_width"],
        "padded_height": preprocess["padded_height"],
        "pixel_values_shape": _json(payload["pixel_values_shape"] if payload else None),
        "pixel_value_fingerprint": preprocess["pixel_value_fingerprint"],
        "generation_settings": _json(generation),
        "manual_body_type": "",
        "manual_colour": "",
        "body_type_correct": "",
        "colour_correct": "",
        "review_notes": "",
    }


def _caption_row(
    *,
    configuration: str,
    backend: FlorenceBackend,
    legacy: OldTdCase2Adapter,
    result: dict[str, Any],
    evidence: EnrichmentEvidenceItem,
    attribute: str,
) -> dict[str, Any]:
    legacy_result = legacy.run_caption_inference(str(evidence.vehicle_crop_path))
    crop = cv2.imread(str(evidence.vehicle_crop_path))
    if crop is None or crop.size == 0:
        raise FileNotFoundError(f"Could not read crop image: {evidence.vehicle_crop_path}")
    generation = {
        "caption": _config_generation_settings(backend, LEGACY_CAPTION_TASK_PROMPT, LEGACY_CAPTION_TASK_PROMPT),
    }
    body_label = legacy_result.body_type_label if attribute in {"body_type", "both"} else ""
    colour_label = legacy_result.colour_label if attribute in {"colour", "both"} else ""
    return {
        "camera_id": result["camera_id"],
        "local_track_id": result["local_track_id"],
        "frame_index": evidence.frame_number,
        "crop_path": str(evidence.vehicle_crop_path),
        "vehicle_class": result.get("vehicle_class", "UNKNOWN"),
        "original_crop_width": evidence.original_crop_width,
        "original_crop_height": evidence.original_crop_height,
        "resolution_tier": evidence.resolution_tier,
        "quality_score": evidence.quality_score,
        "configuration": configuration,
        "adapter_enabled": backend.config.adapter_enabled,
        "adapter_loaded": backend.adapter_active,
        "prompt_mode": "caption",
        "task_token": LEGACY_CAPTION_TASK_PROMPT,
        "prompt": LEGACY_CAPTION_TASK_PROMPT,
        "raw_response": _json({"body_type": legacy_result.raw_response, "colour": legacy_result.raw_response}),
        "normalized_response": _json({"body_type": body_label, "colour": colour_label}),
        "body_type_label": body_label,
        "colour_label": colour_label,
        "parse_status": _json({"body_type": "completed" if body_label else "not_run", "colour": "completed" if colour_label else "not_run"}),
        "parse_reason": _json({"body_type": "caption_parse", "colour": "caption_parse"}),
        "generic_response": _json({"body_type": _generic_response(legacy_result.raw_response), "colour": _generic_response(legacy_result.raw_response)}),
        "prompt_echo": _json({"body_type": _prompt_echo(legacy_result.raw_response, LEGACY_CAPTION_TASK_PROMPT), "colour": _prompt_echo(legacy_result.raw_response, LEGACY_CAPTION_TASK_PROMPT)}),
        "inference_time_ms": legacy_result.inference_time_ms,
        "peak_gpu_memory_mb": backend.metrics.get("gpu_memory_allocated_mb"),
        "processor_path": backend.processor_identifier,
        "device": backend.resolved_device,
        "dtype": backend.resolved_dtype,
        "square_padding_applied": legacy_result.manual_square_padding_applied,
        "padded_width": legacy_result.preprocessed_width,
        "padded_height": legacy_result.preprocessed_height,
        "pixel_values_shape": _json(legacy_result.tensor_shape),
        "pixel_value_fingerprint": _fingerprint_image(crop),
        "generation_settings": _json(generation),
        "manual_body_type": "",
        "manual_colour": "",
        "body_type_correct": "",
        "colour_correct": "",
        "review_notes": "",
    }


def build_pivot_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["camera_id"]), str(row["local_track_id"]), str(row["frame_index"]), str(row["crop_path"]))
        grouped.setdefault(
            key,
            {
                "camera_id": row["camera_id"],
                "local_track_id": row["local_track_id"],
                "frame_index": row["frame_index"],
                "crop_path": row["crop_path"],
                "vehicle_class": row["vehicle_class"],
            },
        )
        prefix = str(row["configuration"])
        grouped[key][f"{prefix}_raw"] = row["raw_response"]
        grouped[key][f"{prefix}_body_type"] = row["body_type_label"]
        grouped[key][f"{prefix}_colour"] = row["colour_label"]
    return list(grouped.values())


def _valid_rate(values: list[str]) -> tuple[int, float, int, float]:
    total = len(values)
    unknown = sum(1 for value in values if str(value).upper() in {"", "UNKNOWN"})
    valid = total - unknown
    return valid, round(100.0 * valid / max(1, total), 2), unknown, round(100.0 * unknown / max(1, total), 2)


def summarize_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary: dict[str, Any] = {"configurations": {}, "agreements": {}}
    summary_rows: list[dict[str, Any]] = []
    for configuration in CONFIGURATION_ORDER:
        config_rows = [row for row in rows if row["configuration"] == configuration]
        body_labels = [str(row.get("body_type_label", "")) for row in config_rows]
        colour_labels = [str(row.get("colour_label", "")) for row in config_rows]
        inference_times = [float(row["inference_time_ms"]) for row in config_rows if row.get("inference_time_ms") not in ("", None)]
        generic_count = sum(1 for row in config_rows if "true" in str(row.get("generic_response", "")).lower())
        prompt_echo_count = sum(1 for row in config_rows if "true" in str(row.get("prompt_echo", "")).lower())
        body_valid, body_valid_rate, body_unknown, body_unknown_rate = _valid_rate(body_labels)
        colour_valid, colour_valid_rate, colour_unknown, colour_unknown_rate = _valid_rate(colour_labels)
        summary["configurations"][configuration] = {
            "crops_tested": len(config_rows),
            "body_type_valid_count": body_valid,
            "body_type_valid_rate": body_valid_rate,
            "body_type_unknown_count": body_unknown,
            "body_type_unknown_rate": body_unknown_rate,
            "colour_valid_count": colour_valid,
            "colour_valid_rate": colour_valid_rate,
            "colour_unknown_count": colour_unknown,
            "colour_unknown_rate": colour_unknown_rate,
            "generic_response_count": generic_count,
            "prompt_echo_count": prompt_echo_count,
            "conflicting_response_count": 0,
            "average_inference_time_ms": round(statistics.fmean(inference_times), 4) if inference_times else 0.0,
            "median_inference_time_ms": round(statistics.median(inference_times), 4) if inference_times else 0.0,
            "peak_gpu_memory_mb": max((float(row.get("peak_gpu_memory_mb") or 0.0) for row in config_rows), default=0.0),
            "model_load_count": 1 if config_rows else 0,
            "adapter_load_success": bool(config_rows and config_rows[0]["adapter_loaded"]),
        }
    pivot_rows = build_pivot_rows(rows)
    comparisons = [
        ("current_body_type", "base_current_body_type", "adapter_current_body_type"),
        ("current_colour", "base_current_colour", "adapter_current_colour"),
        ("caption_body_type", "base_caption_body_type", "adapter_caption_body_type"),
        ("caption_colour", "base_caption_colour", "adapter_caption_colour"),
    ]
    for label, left_key, right_key in comparisons:
        pairs = [(str(row.get(left_key, "")), str(row.get(right_key, ""))) for row in pivot_rows if left_key in row or right_key in row]
        total = len(pairs)
        agreement = sum(1 for left, right in pairs if left == right)
        base_unknown_adapter_valid = sum(1 for left, right in pairs if left in {"", "UNKNOWN"} and right not in {"", "UNKNOWN"})
        base_valid_adapter_unknown = sum(1 for left, right in pairs if left not in {"", "UNKNOWN"} and right in {"", "UNKNOWN"})
        both_valid_different = sum(1 for left, right in pairs if left not in {"", "UNKNOWN"} and right not in {"", "UNKNOWN"} and left != right)
        both_unknown = sum(1 for left, right in pairs if left in {"", "UNKNOWN"} and right in {"", "UNKNOWN"})
        summary["agreements"][label] = {
            "total": total,
            "agreement_rate": round(100.0 * agreement / max(1, total), 2),
            "disagreement_rate": round(100.0 * (total - agreement) / max(1, total), 2),
            "base_unknown_adapter_valid": base_unknown_adapter_valid,
            "base_valid_adapter_unknown": base_valid_adapter_unknown,
            "both_valid_but_different": both_valid_different,
            "both_unknown": both_unknown,
        }
    for configuration, metrics in summary["configurations"].items():
        for key, value in metrics.items():
            summary_rows.append({"section": configuration, "metric": key, "value": value})
    for label, metrics in summary["agreements"].items():
        for key, value in metrics.items():
            summary_rows.append({"section": label, "metric": key, "value": value})
    return summary, summary_rows


def build_manual_review_rows(pivot_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    def _pick(name: str, predicate: Any) -> None:
        for row in pivot_rows:
            key = (str(row["camera_id"]), str(row["local_track_id"]), str(row["frame_index"]), str(row["crop_path"]))
            if key in seen:
                continue
            if predicate(row):
                seen.add(key)
                sample = dict(row)
                sample["review_bucket"] = name
                sample["manual_body_type"] = ""
                sample["manual_colour"] = ""
                sample["review_notes"] = ""
                selected.append(sample)
                return

    _pick("base_unknown_adapter_valid", lambda row: row.get("base_current_body_type") in {"", "UNKNOWN"} and row.get("adapter_current_body_type") not in {"", "UNKNOWN"})
    _pick("base_valid_adapter_unknown", lambda row: row.get("base_current_body_type") not in {"", "UNKNOWN"} and row.get("adapter_current_body_type") in {"", "UNKNOWN"})
    _pick("base_adapter_disagree", lambda row: row.get("base_current_body_type") not in {"", ""} and row.get("adapter_current_body_type") not in {"", ""} and row.get("base_current_body_type") != row.get("adapter_current_body_type"))
    _pick("caption_disagree", lambda row: row.get("base_caption_colour") not in {"", ""} and row.get("adapter_caption_colour") not in {"", ""} and row.get("base_caption_colour") != row.get("adapter_caption_colour"))
    _pick("all_agree", lambda row: len({row.get("base_current_body_type"), row.get("adapter_current_body_type"), row.get("base_caption_body_type"), row.get("adapter_caption_body_type")}) == 1)
    _pick("all_unknown", lambda row: all(row.get(key) in {"", "UNKNOWN"} for key in ["base_current_body_type", "adapter_current_body_type", "base_caption_body_type", "adapter_caption_body_type"]))
    return selected


def write_report(output_dir: Path, summary: dict[str, Any], audit: dict[str, Any], model_audit: dict[str, Any], previous_used_adapter: bool) -> None:
    lines = [
        "# Florence Base vs Adapter Report",
        "",
        f"- Previous 22-crop benchmark used adapter: `{previous_used_adapter}`",
        f"- Base model path: `{model_audit['base_model_path']}`",
        f"- Processor path: `{model_audit['processor_path']}`",
        f"- Adapter path: `{model_audit['adapter_path']}`",
        f"- Adapter type: `{model_audit['adapter_type']}`",
        f"- Adapter base reference: `{model_audit['adapter_base_model_reference']}`",
        f"- Adapter task type: `{model_audit['adapter_task_type']}`",
        "",
        "## Active configs",
    ]
    lines.extend([f"- `{item['config_path']}` -> {item['florence_mode']}" for item in audit["configs"]])
    lines.extend(["", "## Configuration summary"])
    for configuration in CONFIGURATION_ORDER:
        metrics = summary["configurations"].get(configuration, {})
        lines.append(
            f"- `{configuration}` crops={metrics.get('crops_tested', 0)} body_valid={metrics.get('body_type_valid_rate', 0)}% colour_valid={metrics.get('colour_valid_rate', 0)}% avg_ms={metrics.get('average_inference_time_ms', 0)}"
        )
    lines.extend(["", "## Agreement summary"])
    for key, metrics in summary["agreements"].items():
        lines.append(
            f"- `{key}` agreement={metrics['agreement_rate']}% disagreement={metrics['disagreement_rate']}% both_unknown={metrics['both_unknown']}"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "- `continue comparison because accuracy is unverified`",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_comparison(
    *,
    input_run: Path,
    output_dir: Path,
    base_model: str,
    processor_path: str,
    adapter_path: str | None,
    attribute: str,
    include_caption_flow: bool,
    manifest_csv: Path | None = None,
) -> dict[str, Any]:
    _ensure_dir(output_dir)
    audit = generate_configuration_audit(REPO_ROOT, output_dir.parent if output_dir.name == "florence_base_vs_adapter" else output_dir)
    model_audit = inspect_model_assets(base_model, processor_path, adapter_path)
    run_config = _read_run_config(input_run)
    enrichment_config = normalize_vehicle_enrichment_config(run_config.get("vehicle_enrichment", {}))
    manifest = load_crop_manifest(input_run, manifest_csv)
    if manifest_csv is None:
        old_manifest = REPO_ROOT / "outputs" / "florence_old_vs_current" / "comparison.csv"
        if old_manifest.exists():
            manifest = load_crop_manifest(input_run, old_manifest)
    indexed = _manifest_index(input_run)

    base_backend = _build_backend(
        base_model=base_model,
        processor_path=processor_path,
        adapter_path=None,
        adapter_enabled=False,
        template_config=dict(enrichment_config["shared_florence"]),
    )
    adapter_backend = _build_backend(
        base_model=base_model,
        processor_path=processor_path,
        adapter_path=adapter_path,
        adapter_enabled=True,
        template_config=dict(enrichment_config["shared_florence"]),
    )
    logger = logging.getLogger(__name__)
    rows: list[dict[str, Any]] = []
    try:
        current_classifiers = {
            "base": (
                VehicleBodyTypeClassifier(enrichment_config["body_type"], backend=base_backend, logger=logger),
                VehicleColourClassifier(enrichment_config["colour"], backend=base_backend, logger=logger),
            ),
            "adapter": (
                VehicleBodyTypeClassifier(enrichment_config["body_type"], backend=adapter_backend, logger=logger),
                VehicleColourClassifier(enrichment_config["colour"], backend=adapter_backend, logger=logger),
            ),
        }
        caption_runners = {
            "base": OldTdCase2Adapter(base_backend, logger=logger),
            "adapter": OldTdCase2Adapter(adapter_backend, logger=logger),
        }
        for manifest_row in manifest:
            key = _crop_key(manifest_row)
            if key not in indexed:
                raise KeyError(f"Crop from manifest not found in run evidence: {key}")
            result, evidence = indexed[key]
            body_classifier, colour_classifier = current_classifiers["base"]
            rows.append(
                _current_row(
                    configuration="base_current",
                    backend=base_backend,
                    body_classifier=body_classifier,
                    colour_classifier=colour_classifier,
                    result=result,
                    evidence=evidence,
                    attribute=attribute,
                )
            )
            body_classifier, colour_classifier = current_classifiers["adapter"]
            rows.append(
                _current_row(
                    configuration="adapter_current",
                    backend=adapter_backend,
                    body_classifier=body_classifier,
                    colour_classifier=colour_classifier,
                    result=result,
                    evidence=evidence,
                    attribute=attribute,
                )
            )
            if include_caption_flow:
                rows.append(
                    _caption_row(
                        configuration="base_caption",
                        backend=base_backend,
                        legacy=caption_runners["base"],
                        result=result,
                        evidence=evidence,
                        attribute=attribute,
                    )
                )
                rows.append(
                    _caption_row(
                        configuration="adapter_caption",
                        backend=adapter_backend,
                        legacy=caption_runners["adapter"],
                        result=result,
                        evidence=evidence,
                        attribute=attribute,
                    )
                )
    finally:
        base_backend.close()
        adapter_backend.close()

    comparison_csv = output_dir / "comparison.csv"
    if rows:
        with comparison_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    pivot_rows = build_pivot_rows(rows)
    with (output_dir / "pivot_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pivot_rows[0].keys()) if pivot_rows else ["camera_id", "local_track_id", "frame_index", "crop_path", "vehicle_class"])
        writer.writeheader()
        writer.writerows(pivot_rows)

    summary, summary_rows = summarize_rows(rows)
    summary["model_audit"] = model_audit
    summary["active_config_audit"] = audit
    previous_run_config = _read_run_config(input_run)
    previous_used_adapter = bool(previous_run_config.get("vehicle_enrichment", {}).get("shared_florence", {}).get("adapter_enabled"))
    summary["previous_22_crop_benchmark_used_adapter"] = previous_used_adapter
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "metric", "value"])
        writer.writeheader()
        writer.writerows(summary_rows)

    manual_review_rows = build_manual_review_rows(pivot_rows)
    with (output_dir / "manual_review.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(manual_review_rows[0].keys()) if manual_review_rows else ["review_bucket", "camera_id", "local_track_id", "frame_index", "crop_path", "manual_body_type", "manual_colour", "review_notes"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manual_review_rows)

    write_report(output_dir, summary, audit, model_audit, previous_used_adapter)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Florence base-only vs adapter-backed inference on identical saved crops.")
    parser.add_argument("--input-run", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--processor-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--attribute", choices=["body_type", "colour", "both"], default="both")
    parser.add_argument("--include-caption-flow", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-csv", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    run_comparison(
        input_run=Path(args.input_run).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        base_model=args.base_model,
        processor_path=args.processor_path,
        adapter_path=args.adapter_path,
        attribute=args.attribute,
        include_caption_flow=bool(args.include_caption_flow),
        manifest_csv=Path(args.manifest_csv).resolve() if args.manifest_csv else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
