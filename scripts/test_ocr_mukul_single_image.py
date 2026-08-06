from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vehicle_enrichment.ocr_mukul.attribute_parser import OCR_MUKUL_UNKNOWN, parse_caption_attributes
from src.vehicle_enrichment.ocr_mukul.caption_generator import OCRMukulCaptionGenerator
from src.vehicle_enrichment.ocr_mukul.image_preprocessor import OCRMukulImagePreprocessor
from src.vehicle_enrichment.schemas import EnrichmentEvidenceItem
from src.vehicle_enrichment.shared.florence_backend import FlorenceBackend, FlorenceBackendConfig
from src.vehicle_enrichment.vehicle_attribute_prompts import assess_response_quality


DIAGNOSTIC_DEFAULTS: dict[str, dict[str, Any]] = {
    "colour": {
        "task_token": "<VQA>",
        "prompt": "What colour is the vehicle?",
        "generation": {
            "max_new_tokens": 16,
            "num_beams": 1,
            "do_sample": False,
            "use_cache": True,
            "early_stopping": False,
        },
    },
    "body_vqa": {
        "task_token": "<VQA>",
        "prompt": "What body type is the vehicle?",
        "generation": {
            "max_new_tokens": 16,
            "num_beams": 1,
            "do_sample": False,
            "use_cache": True,
            "early_stopping": False,
        },
    },
    "body_detailed_caption": {
        "task_token": "<MORE_DETAILED_CAPTION>",
        "prompt": "",
        "generation": {
            "max_new_tokens": 64,
            "num_beams": 3,
            "do_sample": False,
            "use_cache": True,
            "early_stopping": True,
        },
    },
}


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return payload


def extract_ocr_mukul_settings(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    vehicle_enrichment = dict(config.get("vehicle_enrichment") or {})
    shared_florence = dict(vehicle_enrichment.get("shared_florence") or {})
    if not shared_florence:
        raise ValueError("Missing vehicle_enrichment.shared_florence configuration.")
    ocr_mukul = dict(vehicle_enrichment.get("ocr_mukul") or {})
    return shared_florence, ocr_mukul


def extract_task_settings(config: dict[str, Any], task: str) -> tuple[dict[str, Any], dict[str, Any]]:
    vehicle_enrichment = dict(config.get("vehicle_enrichment") or {})
    normalized_task = str(task).strip().lower()
    if normalized_task == "separate-vehicle-attributes":
        task_section = dict(vehicle_enrichment.get("vehicle_attributes") or {})
        florence_section = dict(task_section.get("florence") or vehicle_enrichment.get("shared_florence") or {})
        return florence_section, task_section
    if normalized_task == "vehicle-attributes":
        task_section = dict(vehicle_enrichment.get("vehicle_attributes") or {})
        florence_section = dict(task_section.get("florence") or vehicle_enrichment.get("shared_florence") or {})
        return florence_section, task_section
    if normalized_task == "plate-ocr":
        plate_section = dict(vehicle_enrichment.get("plate", {}) or {})
        ocr_section = dict(plate_section.get("ocr", {}) or {})
        florence_section = dict(ocr_section.get("florence") or vehicle_enrichment.get("shared_florence") or {})
        return florence_section, ocr_section
    return extract_ocr_mukul_settings(config)


def build_backend_config(shared_florence: dict[str, Any]) -> FlorenceBackendConfig:
    return FlorenceBackendConfig(
        enabled=bool(shared_florence.get("enabled", True)),
        backend=str(shared_florence.get("backend", "florence2")),
        base_model_id=str(shared_florence.get("base_model_id") or ""),
        processor_path=str(shared_florence.get("processor_path") or shared_florence.get("base_model_id") or ""),
        adapter_path=str(shared_florence.get("adapter_path") or ""),
        adapter_enabled=bool(shared_florence.get("adapter_enabled", False)),
        device=str(shared_florence.get("device", "auto")),
        dtype=str(shared_florence.get("dtype", "auto")),
        trust_remote_code=bool(shared_florence.get("trust_remote_code", True)),
        attention_implementation=str(shared_florence.get("attention_implementation", "eager")),
        max_new_tokens=int(shared_florence.get("max_new_tokens", 64)),
        num_beams=int(shared_florence.get("num_beams", 1)),
        use_cache=bool(shared_florence.get("use_cache", False)),
        local_files_only=bool(shared_florence.get("local_files_only", True)),
        lazy_load=bool(shared_florence.get("lazy_load", True)),
    )


def resolve_model_mode(model_mode: str) -> tuple[bool, str]:
    normalized = str(model_mode).strip().lower()
    if normalized == "base":
        return False, "base_model"
    if normalized == "adapter":
        return True, "peft_adapter"
    raise ValueError(f"Unsupported model mode: {model_mode}")


def resolve_prompt(task_token: str | None, prompt: str | None, ocr_mukul: dict[str, Any]) -> tuple[str, str]:
    resolved_task_token = str(task_token if task_token is not None else ocr_mukul.get("task_token", "<CAPTION>")).strip()
    resolved_prompt = str(prompt if prompt is not None else ocr_mukul.get("prompt", "") or "")
    if not resolved_task_token:
        raise ValueError("OCR_MUKUL task token must not be empty.")
    return resolved_task_token, resolved_prompt


def build_evidence_item(image_path: Path, image_shape: tuple[int, int, int]) -> EnrichmentEvidenceItem:
    height, width = image_shape[:2]
    return EnrichmentEvidenceItem(
        local_track_id="single-image-test",
        camera_id="single-image-test",
        native_tracker_id=0,
        frame_number=0,
        timestamp_seconds=0.0,
        source_image_path=str(image_path),
        vehicle_crop_path=str(image_path),
        annotated_frame_path=str(image_path),
        bbox_xyxy=(0.0, 0.0, float(width), float(height)),
        evidence_role="MANUAL",
        detection_confidence=1.0,
        crop_width=int(width),
        crop_height=int(height),
        crop_area=int(width * height),
        sharpness_score=0.0,
        brightness_score=0.0,
        border_penalty=0.0,
        clipping_ratio=0.0,
        quality_score=1.0,
        source_frame_width=int(width),
        source_frame_height=int(height),
        context_padding_ratio=0.0,
        original_crop_width=int(width),
        original_crop_height=int(height),
        resolution_tier="manual",
    )


def build_output_directory(base_output_dir: Path) -> Path:
    timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def get_diagnostic_settings(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    diagnostic_section = dict((config.get("vehicle_enrichment") or {}).get("diagnostic_separate_attributes") or {})
    settings: dict[str, dict[str, Any]] = {}
    for key, defaults in DIAGNOSTIC_DEFAULTS.items():
        configured = dict(diagnostic_section.get(key) or {})
        generation = dict(defaults["generation"])
        generation.update(dict(configured.get("generation") or {}))
        settings[key] = {
            "task_token": str(configured.get("task_token", defaults["task_token"])),
            "prompt": str(configured.get("prompt", defaults["prompt"]) or ""),
            "generation": generation,
        }
    return settings


def calculate_repeat_consensus(labels: list[str], *, unknown_label: str = OCR_MUKUL_UNKNOWN) -> dict[str, Any]:
    normalized = [str(label or unknown_label).strip().upper() or unknown_label for label in labels]
    valid_labels = [label for label in normalized if label != unknown_label]
    counts: dict[str, int] = {}
    for label in valid_labels:
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return {
            "valid_count": 0,
            "unknown_count": len(normalized),
            "unique_valid_labels": [],
            "disagreement_count": 0,
            "consensus_label": unknown_label,
            "stable": False,
            "reason": "all_unknown",
        }
    sorted_counts = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    top_label, top_count = sorted_counts[0]
    second_count = sorted_counts[1][1] if len(sorted_counts) > 1 else 0
    disagreement_count = max(0, len(valid_labels) - top_count)
    has_majority = top_count >= 2 and top_count > second_count
    return {
        "valid_count": len(valid_labels),
        "unknown_count": len(normalized) - len(valid_labels),
        "unique_valid_labels": sorted(counts),
        "disagreement_count": disagreement_count,
        "consensus_label": top_label if has_majority else unknown_label,
        "stable": bool(has_majority),
        "reason": "majority_vote" if has_majority else "conflicting_repeated_predictions",
    }


def _save_image(path: Path, image: Any, *, error_message: str) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(error_message)


def _write_text_report(result: dict[str, Any], output_path: Path, command_text: str, repeat: int) -> None:
    lines = [
        f"Command: {command_text}",
        f"Image: {result['image_path']}",
        f"Original dimensions: {result['original_width']}x{result['original_height']}",
        f"Task token: {result['task_token']}",
        f"Prompt: {result['prompt']}",
        f"Effective processor text: {result['effective_processor_text']}",
        f"Repeat count: {repeat}",
        f"Model mode: {result['model_mode']}",
        f"Adapter requested: {result['adapter_requested']}",
        f"Adapter loaded: {result['adapter_loaded']}",
        f"Effective model type: {result['effective_model_type']}",
        f"Model class: {result['model_class']}",
        f"Processor class: {result['processor_class']}",
        f"Device: {result['device']}",
        f"Dtype: {result['dtype']}",
        f"Model load time ms: {result['model_load_time_ms']}",
        f"GPU memory before load MB: {result['gpu_memory_before_load_mb']}",
        f"GPU memory after load MB: {result['gpu_memory_after_load_mb']}",
        f"GPU memory after inference MB: {result['gpu_memory_after_inference_mb']}",
        f"GPU memory allocated MB: {result.get('gpu_memory_allocated_mb')}",
        f"GPU memory reserved MB: {result.get('gpu_memory_reserved_mb')}",
        f"Raw response: {result['raw_generated_text']}",
        f"Post-processed response: {result['post_processed_response']}",
        f"Parsed body type: {result['parsed_body_type']}",
        f"Parsed colour: {result['parsed_colour']}",
        f"Inference time ms: {result['inference_time_ms']}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_comparison_text_report(comparison: dict[str, Any], output_path: Path) -> None:
    lines = [
        f"Same image: {comparison['same_image']}",
        f"Same preprocessing: {comparison['same_preprocessing']}",
        f"Same task token: {comparison['same_task_token']}",
        f"Same prompt: {comparison['same_prompt']}",
        f"Base response: {comparison['base_response']}",
        f"Adapter response: {comparison['adapter_response']}",
        f"Base body type: {comparison['base_body_type']}",
        f"Adapter body type: {comparison['adapter_body_type']}",
        f"Base colour: {comparison['base_colour']}",
        f"Adapter colour: {comparison['adapter_colour']}",
        f"Responses match: {comparison['responses_match']}",
        f"Body types match: {comparison['body_types_match']}",
        f"Colours match: {comparison['colours_match']}",
        f"Base inference time ms: {comparison['base_inference_time_ms']}",
        f"Adapter inference time ms: {comparison['adapter_inference_time_ms']}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_separate_text_report(result: dict[str, Any], output_path: Path, command_text: str) -> None:
    def _section_lines(name: str, section: dict[str, Any], label_key: str) -> list[str]:
        lines = [
            f"{name}:",
            f"  enabled: {section.get('enabled', True)}",
            f"  task token: {section['task_token']}",
            f"  prompt: {section['prompt']}",
            f"  effective processor text: {section['effective_processor_text']}",
            f"  consensus: {section[label_key]}",
            f"  stable: {section['stable']}",
        ]
        for repetition in section["repetitions"]:
            lines.extend(
                [
                    f"  repeat {repetition['index']}:",
                    f"    raw response: {repetition['raw_response']}",
                    f"    post-processed response: {repetition['post_processed_response']}",
                    f"    parsed colour: {repetition['parsed_colour']}",
                    f"    parsed body type: {repetition['parsed_body_type']}",
                    f"    status: {repetition['response_status']}",
                    f"    reason: {repetition['response_reason']}",
                    f"    inference time ms: {repetition['inference_time_ms']}",
                ]
            )
        return lines

    lines = [
        f"Command: {command_text}",
        f"Task: {result['task']}",
        f"Image: {result['image_path']}",
        f"Original dimensions: {result['original_dimensions']['width']}x{result['original_dimensions']['height']}",
        f"Preprocessed dimensions: {result['preprocessed_dimensions']['width']}x{result['preprocessed_dimensions']['height']}",
        f"Model mode: {result['model']['model_mode']}",
        f"Adapter requested: {result['model']['adapter_requested']}",
        f"Adapter loaded: {result['model']['adapter_loaded']}",
        f"Effective model type: {result['model']['effective_model_type']}",
        f"Model class: {result['model']['model_class']}",
        f"Processor class: {result['model']['processor_class']}",
        f"Device: {result['model']['device']}",
        f"Dtype: {result['model']['dtype']}",
        f"Base model load count: {result['model']['load_count']}",
        f"Model load time ms: {result['model']['load_time_ms']}",
        f"GPU memory before load MB: {result['gpu']['before_load_mb']}",
        f"GPU memory after load MB: {result['gpu']['after_load_mb']}",
        f"GPU memory after inference MB: {result['gpu']['after_inference_mb']}",
        f"GPU memory after release MB: {result['gpu']['after_release_mb']}",
    ]
    lines.extend(_section_lines("Colour", result["colour"], "consensus_colour"))
    lines.extend(_section_lines("Body VQA", result["body_vqa"], "consensus_body_type"))
    lines.extend(_section_lines("Body detailed caption", result["body_detailed_caption"], "consensus_body_type"))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_gpu_memory_allocated_mb() -> float:
    try:
        import torch
    except Exception:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    return round(float(torch.cuda.memory_allocated() / (1024 * 1024)), 3)


def release_backend_resources(backend: FlorenceBackend | None) -> dict[str, Any]:
    status = {
        "released": False,
        "gpu_memory_before_release_mb": get_gpu_memory_allocated_mb(),
        "gpu_memory_after_release_mb": 0.0,
    }
    if backend is not None:
        model = getattr(backend, "_model", None)
        if model is not None:
            del model
        backend.close()
    gc.collect()
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    status["gpu_memory_after_release_mb"] = get_gpu_memory_allocated_mb()
    status["released"] = True
    return status


def _run_single_backend_inference(
    *,
    backend: FlorenceBackend,
    prepared_image: Any,
    task_token: str,
    prompt: str,
    generation_settings: dict[str, Any],
) -> tuple[dict[str, Any], Any, str]:
    response = backend.run_task(
        prepared_image,
        task_token,
        prompt,
        adapter_active=False,
        generation_overrides=generation_settings,
    )
    if response["status"] != "completed":
        raise RuntimeError(str(response.get("reason") or "Separate attribute Florence inference failed."))
    payload = dict(response.get("payload") or {})
    raw_generated_text = str(payload.get("generated_text") or "").strip()
    post_processed = payload.get("parsed_answer")
    if isinstance(post_processed, dict):
        response_text = str(post_processed.get(task_token) or raw_generated_text).strip()
    else:
        response_text = str(post_processed or raw_generated_text).strip()
    parsed = parse_caption_attributes(response_text)
    response_status, response_reason = assess_response_quality(
        raw_generated_text,
        parsed,
        attribute_task="colour" if task_token == "<VQA>" and "colour" in prompt.lower() else "body_type",
        prompt=prompt,
    )
    return (
        {
            "raw_response": raw_generated_text,
            "post_processed_response": response_text,
            "parsed_colour": parsed.normalized_colour,
            "parsed_body_type": parsed.normalized_body_type,
            "response_status": response_status,
            "response_reason": response_reason,
            "inference_time_ms": float(payload.get("inference_duration_ms", 0.0) or 0.0),
        },
        parsed,
        str(payload.get("final_processor_text") or f"{task_token}{prompt}"),
    )


def run_separate_vehicle_attribute_test(
    *,
    image_path: Path,
    config_path: Path,
    output_root: Path,
    repeat: int,
    command_text: str,
    body_strategy: str,
    backend: FlorenceBackend | None = None,
) -> tuple[dict[str, Any], Path]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    if repeat < 1:
        raise ValueError("--repeat must be at least 1.")

    config = load_yaml_config(config_path)
    shared_florence, _ = extract_task_settings(config, "separate-vehicle-attributes")
    diagnostic_settings = get_diagnostic_settings(config)
    image = cv2.imread(str(image_path))
    if image is None or image.size == 0:
        raise FileNotFoundError(f"Image could not be read: {image_path}")
    original_height, original_width = image.shape[:2]

    output_dir = build_output_directory(output_root)
    gpu_memory_before_load_mb = get_gpu_memory_allocated_mb()
    backend_instance = backend or FlorenceBackend(
        build_backend_config(shared_florence),
        logger=logging.getLogger("ocr_mukul_single_image.separate_vehicle_attributes"),
        adapter_enabled_override=False,
    )
    backend_instance.load()
    gpu_memory_after_load_mb = get_gpu_memory_allocated_mb()
    if backend_instance.adapter_active:
        raise RuntimeError("Separate vehicle attribute diagnostic must use base Florence without the adapter.")

    preprocessor = OCRMukulImagePreprocessor()
    prepared = preprocessor.prepare(image)

    _save_image(output_dir / "input_image.jpg", image, error_message="Failed to save input_image.jpg")
    _save_image(output_dir / "preprocessed_image.jpg", prepared.image_bgr, error_message="Failed to save preprocessed_image.jpg")

    tasks_to_run: list[tuple[str, str, bool]] = [
        ("colour", "short_vqa", True),
        ("body_vqa", "short_vqa", body_strategy in {"vqa", "both"}),
        ("body_detailed_caption", "detailed_caption", body_strategy in {"detailed-caption", "both"}),
    ]
    csv_rows: list[dict[str, Any]] = []
    section_results: dict[str, dict[str, Any]] = {}

    for section_name, strategy_name, enabled in tasks_to_run:
        settings = diagnostic_settings[section_name]
        repetitions: list[dict[str, Any]] = []
        parsed_labels: list[str] = []
        effective_processor_text = f"{settings['task_token']}{settings['prompt']}"
        if enabled:
            for attempt in range(1, repeat + 1):
                repetition_payload, parsed, effective_processor_text = _run_single_backend_inference(
                    backend=backend_instance,
                    prepared_image=prepared.image_bgr,
                    task_token=settings["task_token"],
                    prompt=settings["prompt"],
                    generation_settings=settings["generation"],
                )
                repetition_payload["index"] = attempt
                repetitions.append(repetition_payload)
                parsed_labels.append(
                    repetition_payload["parsed_colour"] if section_name == "colour" else repetition_payload["parsed_body_type"]
                )
                csv_rows.append(
                    {
                        "image_path": str(image_path),
                        "attribute_task": "colour" if section_name == "colour" else "body_type",
                        "strategy": strategy_name,
                        "repeat_index": attempt,
                        "task_token": settings["task_token"],
                        "prompt": settings["prompt"],
                        "effective_processor_text": effective_processor_text,
                        "raw_response": repetition_payload["raw_response"],
                        "post_processed_response": repetition_payload["post_processed_response"],
                        "parsed_colour": repetition_payload["parsed_colour"],
                        "parsed_body_type": repetition_payload["parsed_body_type"],
                        "response_status": repetition_payload["response_status"],
                        "response_reason": repetition_payload["response_reason"],
                        "inference_time_ms": repetition_payload["inference_time_ms"],
                        "model_mode": "base",
                        "adapter_loaded": False,
                    }
                )
        consensus = calculate_repeat_consensus(parsed_labels)
        section_results[section_name] = {
            "enabled": enabled,
            "task_token": settings["task_token"],
            "prompt": settings["prompt"],
            "effective_processor_text": effective_processor_text,
            "generation_settings": dict(settings["generation"]),
            "repetitions": repetitions,
            "valid_count": consensus["valid_count"],
            "unknown_count": consensus["unknown_count"],
            "unique_valid_labels": consensus["unique_valid_labels"],
            "disagreement_count": consensus["disagreement_count"],
            "stable": consensus["stable"],
            "consensus_colour": consensus["consensus_label"] if section_name == "colour" else OCR_MUKUL_UNKNOWN,
            "consensus_body_type": consensus["consensus_label"] if section_name != "colour" else OCR_MUKUL_UNKNOWN,
            "consensus_reason": consensus["reason"],
        }

    gpu_memory_after_inference_mb = get_gpu_memory_allocated_mb()
    release_status = release_backend_resources(backend_instance)
    result_payload = {
        "task": "separate-vehicle-attributes",
        "image_path": str(image_path),
        "original_dimensions": {"width": int(original_width), "height": int(original_height)},
        "preprocessed_dimensions": {
            "width": int(prepared.preprocessed_width),
            "height": int(prepared.preprocessed_height),
        },
        "model": {
            "model_mode": "base",
            "adapter_requested": False,
            "adapter_loaded": False,
            "effective_model_type": "base_model",
            "model_class": str(backend_instance.metrics.get("florence_model_class") or ""),
            "processor_class": str(backend_instance.metrics.get("florence_processor_class") or ""),
            "device": str(backend_instance.resolved_device),
            "dtype": str(backend_instance.resolved_dtype),
            "load_count": int(backend_instance.metrics.get("florence_load_successes") or 1),
            "load_time_ms": float(backend_instance.metrics.get("florence_load_duration_ms") or 0.0),
        },
        "colour": section_results["colour"],
        "body_vqa": section_results["body_vqa"],
        "body_detailed_caption": section_results["body_detailed_caption"],
        "gpu": {
            "before_load_mb": gpu_memory_before_load_mb,
            "after_load_mb": gpu_memory_after_load_mb,
            "after_inference_mb": gpu_memory_after_inference_mb,
            "after_release_mb": release_status["gpu_memory_after_release_mb"],
        },
    }
    (output_dir / "result.json").write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    _write_separate_text_report(result_payload, output_dir / "result.txt", command_text)
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "attribute_task",
                "strategy",
                "repeat_index",
                "task_token",
                "prompt",
                "effective_processor_text",
                "raw_response",
                "post_processed_response",
                "parsed_colour",
                "parsed_body_type",
                "response_status",
                "response_reason",
                "inference_time_ms",
                "model_mode",
                "adapter_loaded",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    return result_payload, output_dir


def _build_result_payload(
    *,
    model_mode: str,
    image_path: Path,
    original_width: int,
    original_height: int,
    caption_result: Any,
    parsed: Any,
    task_token: str,
    prompt: str,
    backend_instance: FlorenceBackend,
    effective_model_type: str,
    adapter_requested: bool,
    gpu_memory_before_load_mb: float,
    gpu_memory_after_load_mb: float,
    gpu_memory_after_inference_mb: float,
    load_count: int,
    repetitions: list[dict[str, Any]],
) -> dict[str, Any]:
    backend_metrics = backend_instance.metrics
    return {
        "model_mode": model_mode,
        "image_path": str(image_path),
        "original_width": int(original_width),
        "original_height": int(original_height),
        "preprocessed_width": int(caption_result.prepared.preprocessed_width),
        "preprocessed_height": int(caption_result.prepared.preprocessed_height),
        "task_token": task_token,
        "prompt": prompt,
        "effective_processor_text": f"{task_token}{prompt}",
        "raw_generated_text": caption_result.raw_generated_text,
        "post_processed_response": caption_result.post_processed_caption,
        "parsed_body_type": parsed.normalized_body_type,
        "parsed_colour": parsed.normalized_colour,
        "adapter_requested": adapter_requested,
        "adapter_loaded": bool(backend_instance.adapter_active),
        "effective_model_type": effective_model_type,
        "model_class": str(backend_metrics.get("florence_model_class") or ""),
        "processor_class": str(backend_metrics.get("florence_processor_class") or ""),
        "device": str(backend_instance.resolved_device),
        "dtype": str(backend_instance.resolved_dtype),
        "model_load_time_ms": float(backend_metrics.get("florence_load_duration_ms") or 0.0),
        "inference_time_ms": float(caption_result.inference_time_ms),
        "gpu_memory_before_load_mb": gpu_memory_before_load_mb,
        "gpu_memory_after_load_mb": gpu_memory_after_load_mb,
        "gpu_memory_after_inference_mb": gpu_memory_after_inference_mb,
        "gpu_memory_allocated_mb": backend_metrics.get("gpu_memory_allocated_mb"),
        "gpu_memory_reserved_mb": backend_metrics.get("gpu_memory_reserved_mb"),
        "model_identifier": caption_result.model_identifier,
        "processor_identifier": caption_result.processor_identifier,
        "pixel_values_shape": caption_result.pixel_values_shape,
        "model_load_count": load_count,
        "repetitions": repetitions,
    }


def run_mode(
    *,
    model_mode: str,
    image_path: Path,
    config_path: Path,
    task_token: str | None,
    prompt: str | None,
    repeat: int,
    output_dir: Path,
    command_text: str,
    backend: FlorenceBackend | None = None,
    task: str = "generic",
) -> dict[str, Any]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    if repeat < 1:
        raise ValueError("--repeat must be at least 1.")

    config = load_yaml_config(config_path)
    shared_florence, ocr_mukul = extract_task_settings(config, task)
    default_task_token = "<OCR>" if str(task).strip().lower() == "plate-ocr" else "<VQA>"
    resolved_task_token = str(task_token if task_token is not None else ocr_mukul.get("task_token", default_task_token)).strip() or default_task_token
    resolved_prompt = str(prompt if prompt is not None else ocr_mukul.get("prompt", "") or "")
    adapter_requested, effective_model_type = resolve_model_mode(model_mode)
    image = cv2.imread(str(image_path))
    if image is None or image.size == 0:
        raise FileNotFoundError(f"Image could not be read: {image_path}")
    original_height, original_width = image.shape[:2]

    gpu_memory_before_load_mb = get_gpu_memory_allocated_mb()
    backend_instance = backend or FlorenceBackend(
        build_backend_config(shared_florence),
        logger=logging.getLogger(f"ocr_mukul_single_image.{model_mode}"),
        adapter_enabled_override=adapter_requested,
    )
    backend_instance.load()
    gpu_memory_after_load_mb = get_gpu_memory_allocated_mb()
    if adapter_requested and not backend_instance.adapter_active:
        raise RuntimeError("OCR_MUKUL single-image test requires the Florence adapter, but it is not active.")
    if not adapter_requested and backend_instance.adapter_active:
        raise RuntimeError("Base mode unexpectedly loaded the Florence adapter.")

    evidence_item = build_evidence_item(image_path, image.shape)
    caption_generator = OCRMukulCaptionGenerator(
        backend_instance,
        task_token=resolved_task_token,
        prompt=resolved_prompt,
    )

    print(f"[{model_mode}] adapter_requested={adapter_requested}")
    print(f"[{model_mode}] adapter_loaded={backend_instance.adapter_active}")
    print(f"[{model_mode}] effective_model_type={effective_model_type}")
    print(f"[{model_mode}] original image dimensions: {original_width}x{original_height}")
    print(f"[{model_mode}] effective processor text: {resolved_task_token}{resolved_prompt}")

    repetitions: list[dict[str, Any]] = []
    caption_result = None
    parsed = None
    for attempt in range(1, repeat + 1):
        caption_result = caption_generator.generate(evidence_item)
        parsed = parse_caption_attributes(caption_result.post_processed_caption)
        repetitions.append(
            {
                "index": attempt,
                "raw_response": caption_result.raw_generated_text,
                "post_processed_response": caption_result.post_processed_caption,
                "body_type": parsed.normalized_body_type,
                "colour": parsed.normalized_colour,
                "inference_time_ms": float(caption_result.inference_time_ms),
            }
        )
    assert caption_result is not None and parsed is not None
    gpu_memory_after_inference_mb = get_gpu_memory_allocated_mb()
    result_payload = _build_result_payload(
        model_mode=model_mode,
        image_path=image_path,
        original_width=original_width,
        original_height=original_height,
        caption_result=caption_result,
        parsed=parsed,
        task_token=resolved_task_token,
        prompt=resolved_prompt,
        backend_instance=backend_instance,
        effective_model_type=effective_model_type,
        adapter_requested=adapter_requested,
        gpu_memory_before_load_mb=gpu_memory_before_load_mb,
        gpu_memory_after_load_mb=gpu_memory_after_load_mb,
        gpu_memory_after_inference_mb=gpu_memory_after_inference_mb,
        load_count=1,
        repetitions=repetitions,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    _write_text_report(result_payload, output_dir / "result.txt", command_text, repeat)
    if not cv2.imwrite(str(output_dir.parent / "input_image.jpg"), image):
        raise RuntimeError("Failed to save input_image.jpg")
    if not cv2.imwrite(str(output_dir.parent / "preprocessed_image.jpg"), caption_result.prepared.image_bgr):
        raise RuntimeError("Failed to save preprocessed_image.jpg")
    result_payload["_backend"] = backend_instance
    return result_payload


def run_single_image_test(
    *,
    image_path: Path,
    config_path: Path,
    task_token: str | None,
    prompt: str | None,
    output_root: Path,
    repeat: int,
    command_text: str,
    model_mode: str = "adapter",
    task: str = "generic",
    backend: FlorenceBackend | None = None,
) -> tuple[dict[str, Any], Path]:
    output_dir = build_output_directory(output_root)
    normalized_model_mode = str(model_mode).strip().lower()
    if normalized_model_mode in {"base", "adapter"}:
        result_payload = run_mode(
            model_mode=normalized_model_mode,
            image_path=image_path,
            config_path=config_path,
            task_token=task_token,
            prompt=prompt,
            repeat=repeat,
            output_dir=output_dir / normalized_model_mode,
            command_text=command_text,
            task=task,
            backend=backend,
        )
        backend_instance = result_payload.pop("_backend", None)
        release_status = release_backend_resources(backend_instance)
        result_payload["gpu_memory_release_status"] = release_status
        (output_dir / normalized_model_mode / "result.json").write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
        return result_payload, output_dir

    if normalized_model_mode != "compare":
        raise ValueError("--model-mode must be one of: base, adapter, compare")

    base_result = run_mode(
        model_mode="base",
        image_path=image_path,
        config_path=config_path,
        task_token=task_token,
        prompt=prompt,
        repeat=repeat,
        output_dir=output_dir / "base",
        command_text=command_text,
        task=task,
        backend=None,
    )
    base_backend = base_result.pop("_backend", None)
    base_release_status = release_backend_resources(base_backend)

    adapter_result = run_mode(
        model_mode="adapter",
        image_path=image_path,
        config_path=config_path,
        task_token=task_token,
        prompt=prompt,
        repeat=repeat,
        output_dir=output_dir / "adapter",
        command_text=command_text,
        task=task,
        backend=None,
    )
    adapter_backend = adapter_result.pop("_backend", None)
    adapter_release_status = release_backend_resources(adapter_backend)

    comparison = {
        "same_image": base_result["image_path"] == adapter_result["image_path"],
        "same_preprocessing": (
            base_result["preprocessed_width"] == adapter_result["preprocessed_width"]
            and base_result["preprocessed_height"] == adapter_result["preprocessed_height"]
        ),
        "same_task_token": base_result["task_token"] == adapter_result["task_token"],
        "same_prompt": base_result["prompt"] == adapter_result["prompt"],
        "base_response": base_result["raw_generated_text"],
        "adapter_response": adapter_result["raw_generated_text"],
        "base_body_type": base_result["parsed_body_type"],
        "adapter_body_type": adapter_result["parsed_body_type"],
        "base_colour": base_result["parsed_colour"],
        "adapter_colour": adapter_result["parsed_colour"],
        "responses_match": base_result["raw_generated_text"] == adapter_result["raw_generated_text"],
        "body_types_match": base_result["parsed_body_type"] == adapter_result["parsed_body_type"],
        "colours_match": base_result["parsed_colour"] == adapter_result["parsed_colour"],
        "base_inference_time_ms": base_result["inference_time_ms"],
        "adapter_inference_time_ms": adapter_result["inference_time_ms"],
        "base_gpu_release_status": base_release_status,
        "adapter_gpu_release_status": adapter_release_status,
    }
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    _write_comparison_text_report(comparison, output_dir / "comparison.txt")
    (output_dir / "base" / "result.json").write_text(json.dumps(base_result, indent=2), encoding="utf-8")
    (output_dir / "adapter" / "result.json").write_text(json.dumps(adapter_result, indent=2), encoding="utf-8")
    return {
        "model_mode": "compare",
        "base": base_result,
        "adapter": adapter_result,
        "comparison": comparison,
    }, output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OCR_MUKUL Florence on a single image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    parser.add_argument("--model-mode", default="adapter", choices=["base", "adapter", "compare"], help="Run base-only, adapter, or compare mode.")
    parser.add_argument("--task", default="generic", choices=["generic", "vehicle-attributes", "plate-ocr", "separate-vehicle-attributes"], help="Select which task config to use.")
    parser.add_argument("--task-token", help="Override OCR_MUKUL task token.")
    parser.add_argument("--prompt", help="Override OCR_MUKUL prompt.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "single_image_florence_test"), help="Base output directory.")
    parser.add_argument("--repeat", type=int, default=1, help="Run the same loaded model multiple times.")
    parser.add_argument("--body-strategy", default="both", choices=["vqa", "detailed-caption", "both"], help="Body-type strategy for separate-vehicle-attributes mode.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    command_text = subprocess.list2cmdline([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    try:
        if args.task == "separate-vehicle-attributes":
            if args.model_mode != "base":
                raise ValueError("--task separate-vehicle-attributes requires --model-mode base.")
            if args.task_token is not None or args.prompt is not None:
                raise ValueError("--task-token and --prompt are not supported for --task separate-vehicle-attributes.")
            result, output_dir = run_separate_vehicle_attribute_test(
                image_path=Path(args.image),
                config_path=Path(args.config),
                output_root=Path(args.output_dir).parent / "separate_vehicle_attribute_test"
                if Path(args.output_dir).name == "single_image_florence_test"
                else Path(args.output_dir),
                repeat=args.repeat,
                command_text=command_text,
                body_strategy=args.body_strategy,
            )
        else:
            result, output_dir = run_single_image_test(
                image_path=Path(args.image),
                config_path=Path(args.config),
                task_token=args.task_token,
                prompt=args.prompt,
                output_root=Path(args.output_dir),
                repeat=args.repeat,
                command_text=command_text,
                model_mode=args.model_mode,
                task=args.task,
            )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.task == "separate-vehicle-attributes":
        print("Image:", result["image_path"])
        print("Original dimensions:", f"{result['original_dimensions']['width']}x{result['original_dimensions']['height']}")
        print("Preprocessed dimensions:", f"{result['preprocessed_dimensions']['width']}x{result['preprocessed_dimensions']['height']}")
        print("Base model load count:", result["model"]["load_count"])
        print("Adapter loaded:", result["model"]["adapter_loaded"])
        print("Device:", result["model"]["device"])
        print("Dtype:", result["model"]["dtype"])
        print("Task         Strategy            Repeat    Raw response    Colour    Body type    Status    Inference ms")
        for attribute_task, strategy_key, section_key in (
            ("colour", "short_vqa", "colour"),
            ("body_type", "short_vqa", "body_vqa"),
            ("body_type", "detailed_caption", "body_detailed_caption"),
        ):
            section = result[section_key]
            if not section["enabled"]:
                continue
            for repetition in section["repetitions"]:
                print(
                    f"{attribute_task:<12}{strategy_key:<20}{repetition['index']:<10}{repetition['raw_response']:<16}"
                    f"{repetition['parsed_colour']:<10}{repetition['parsed_body_type']:<12}{repetition['response_status']:<10}{repetition['inference_time_ms']}"
                )
        print("Colour consensus:", result["colour"]["consensus_colour"])
        print("Colour stable:", result["colour"]["stable"])
        print("Body VQA consensus:", result["body_vqa"]["consensus_body_type"])
        print("Body VQA stable:", result["body_vqa"]["stable"])
        print("Body caption consensus:", result["body_detailed_caption"]["consensus_body_type"])
        print("Body caption stable:", result["body_detailed_caption"]["stable"])
        print("GPU memory after release:", result["gpu"]["after_release_mb"])
        print("Output folder:", output_dir)
    elif args.model_mode == "compare":
        base = result["base"]
        adapter = result["adapter"]
        print("Mode      Raw response    Body type    Colour    Adapter loaded    Inference ms")
        print(f"base      {base['raw_generated_text']}    {base['parsed_body_type']}    {base['parsed_colour']}    {base['adapter_loaded']}    {base['inference_time_ms']}")
        print(f"adapter   {adapter['raw_generated_text']}    {adapter['parsed_body_type']}    {adapter['parsed_colour']}    {adapter['adapter_loaded']}    {adapter['inference_time_ms']}")
        print("Effective prompt:", base["effective_processor_text"])
        print("Original image dimensions:", f"{base['original_width']}x{base['original_height']}")
        print("Preprocessed image dimensions:", f"{base['preprocessed_width']}x{base['preprocessed_height']}")
        print("Model load count:", base["model_load_count"] + adapter["model_load_count"])
        print("GPU memory release status:", result["comparison"]["base_gpu_release_status"], result["comparison"]["adapter_gpu_release_status"])
    else:
        print("Raw generated text:", result["raw_generated_text"])
        print("Post-processed response:", result["post_processed_response"])
        print("Parsed body type:", result["parsed_body_type"])
        print("Parsed colour:", result["parsed_colour"])
        print("Inference time ms:", result["inference_time_ms"])
        print("Model device:", result["device"])
        print("Model dtype:", result["dtype"])
        print("Adapter loaded status:", result["adapter_loaded"])
        print("GPU memory allocated MB:", result.get("gpu_memory_allocated_mb"))
        print("GPU memory reserved MB:", result.get("gpu_memory_reserved_mb"))
    print("Files created:")
    print(f"- {output_dir}")
    print("Exact command used:", command_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
