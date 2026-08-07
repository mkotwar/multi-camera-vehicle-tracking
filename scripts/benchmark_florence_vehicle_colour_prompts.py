from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
from typing import Any

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vehicle_enrichment.colour.classifier import (
    VehicleColourClassifier,
    get_colour_prompt_variants,
)
from src.vehicle_enrichment.shared import FlorenceBackend, FlorenceBackendConfig


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _build_backend(*, device: str, enabled: bool = True) -> FlorenceBackend:
    config = FlorenceBackendConfig(
        enabled=enabled,
        backend="florence2",
        base_model_id="D:/project/models/Florence-2-base-ft",
        processor_path="D:/project/models/Florence-2-base-ft",
        adapter_path="",
        adapter_enabled=False,
        device=device,
        dtype="auto",
        trust_remote_code=True,
        attention_implementation="eager",
        max_new_tokens=64,
        num_beams=1,
        use_cache=False,
        local_files_only=True,
        lazy_load=True,
    )
    return FlorenceBackend(config, logger=logging.getLogger("colour_prompt_benchmark"))


def _collect_images(input_path: Path, *, max_images: int | None = None) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    images = sorted(path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if max_images is not None:
        return images[: max(0, int(max_images))]
    return images


def _extract_track_id(image_path: Path) -> str:
    for part in image_path.parts[::-1]:
        if part.startswith("CAM_") and "_TRACK_" in part:
            return part.replace("_TRACK_", ":TRACK_")
    return image_path.parent.name


def _run_prompt_once(
    *,
    backend: Any,
    classifier: VehicleColourClassifier,
    image_path: Path,
    prompt_variant: dict[str, str],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "image_path": str(image_path.resolve()),
        "track_id": _extract_track_id(image_path),
        "prompt_id": str(prompt_variant["id"]),
        "task_prompt": str(prompt_variant["task_prompt"]),
        "prompt_text": str(prompt_variant["prompt_text"]),
        "raw_response": None,
        "parsed_label": "UNKNOWN",
        "normalization_reason": "image_load_failed",
        "response_kind": "error",
        "inference_duration_ms": None,
        "status": "error",
        "error": None,
    }
    image = cv2.imread(str(image_path))
    if image is None or image.size == 0:
        record["error"] = "Image could not be decoded."
        return record
    response = backend.run_task(image, record["task_prompt"], record["prompt_text"])
    if response["status"] != "completed":
        record["error"] = str(response.get("reason"))
        record["normalization_reason"] = "backend_error"
        return record
    payload = dict(response.get("payload") or {})
    raw_response = classifier._extract_colour_text(payload)
    parsed_label, normalization_reason = classifier.normalize_label(raw_response)
    record.update(
        {
            "raw_response": raw_response,
            "parsed_label": parsed_label,
            "normalization_reason": normalization_reason,
            "response_kind": classifier.response_kind(raw_response, normalization_reason),
            "inference_duration_ms": float(payload.get("inference_duration_ms", 0.0)),
            "status": "completed",
            "error": None,
        }
    )
    return record


def _build_prompt_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    prompt_groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        prompt_groups.setdefault(str(record["prompt_id"]), []).append(record)

    for prompt_id, prompt_records in sorted(prompt_groups.items()):
        total_images = len(prompt_records)
        successful = [row for row in prompt_records if row["status"] == "completed"]
        valid = [row for row in successful if row["parsed_label"] != "UNKNOWN"]
        unknown = [row for row in successful if row["parsed_label"] == "UNKNOWN"]
        generic = [row for row in successful if row["response_kind"] == "generic_invalid"]
        ambiguous = [row for row in successful if row["normalization_reason"] == "ambiguous_multiple_labels"]
        failed = [row for row in prompt_records if row["status"] != "completed"]
        raw_counts: dict[str, int] = {}
        label_counts: dict[str, int] = {}
        unknown_reason_counts: dict[str, int] = {}
        for row in successful:
            raw_key = VehicleColourClassifier._clean_text(str(row.get("raw_response") or ""))
            if raw_key:
                raw_counts[raw_key] = raw_counts.get(raw_key, 0) + 1
            label = str(row["parsed_label"])
            label_counts[label] = label_counts.get(label, 0) + 1
            if label == "UNKNOWN":
                reason = str(row["normalization_reason"])
                unknown_reason_counts[reason] = unknown_reason_counts.get(reason, 0) + 1
        durations = [float(row["inference_duration_ms"]) for row in successful if row["inference_duration_ms"] is not None]
        summary[prompt_id] = {
            "total_images": total_images,
            "successful_inferences": len(successful),
            "valid_colour_labels": len(valid),
            "unknown_labels": len(unknown),
            "generic_responses": len(generic),
            "ambiguous_responses": len(ambiguous),
            "failed_inferences": len(failed),
            "valid_label_rate": float(len(valid) / total_images) if total_images else 0.0,
            "unknown_rate": float(len(unknown) / total_images) if total_images else 0.0,
            "generic_response_rate": float(len(generic) / total_images) if total_images else 0.0,
            "average_inference_duration_ms": float(sum(durations) / len(durations)) if durations else 0.0,
            "raw_response_frequency": raw_counts,
            "label_counts": label_counts,
            "unknown_reason_counts": unknown_reason_counts,
            "same_track_consistency": _build_track_consistency(prompt_records),
        }
    return summary


def _build_track_consistency(records: list[dict[str, Any]]) -> dict[str, int]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["track_id"]), []).append(record)
    summary = {
        "matching_valid_predictions": 0,
        "conflicting_valid_predictions": 0,
        "one_valid_one_invalid": 0,
        "all_invalid": 0,
    }
    for rows in grouped.values():
        labels = [str(row["parsed_label"]) for row in rows]
        valid = [label for label in labels if label != "UNKNOWN"]
        invalid_count = len(labels) - len(valid)
        if len(valid) >= 2 and len(set(valid)) == 1:
            summary["matching_valid_predictions"] += 1
        elif len(set(valid)) >= 2:
            summary["conflicting_valid_predictions"] += 1
        elif len(valid) == 1 and invalid_count >= 1:
            summary["one_valid_one_invalid"] += 1
        elif not valid:
            summary["all_invalid"] += 1
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def run_benchmark(
    *,
    input_path: Path,
    output_dir: Path,
    device: str = "auto",
    max_images: int | None = None,
    backend: Any | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = _collect_images(input_path, max_images=max_images)
    local_backend = backend or _build_backend(device=device)
    classifier = VehicleColourClassifier(
        {
            "enabled": True,
            "allowed_labels": ["BLACK", "WHITE", "GREY", "SILVER", "RED", "PINK", "BLUE", "GREEN", "YELLOW", "ORANGE", "BROWN", "BEIGE", "PURPLE", "OTHER", "UNKNOWN"],
            "retry_on_invalid_response": True,
            "maximum_prompt_attempts": 2,
        },
        backend=local_backend,
        logger=logging.getLogger("colour_prompt_benchmark"),
    )
    prompt_variants = get_colour_prompt_variants(include_no_task_prefix_variant=True)
    rows: list[dict[str, Any]] = []
    for image_path in images:
        for prompt_variant in prompt_variants:
            try:
                rows.append(
                    _run_prompt_once(
                        backend=local_backend,
                        classifier=classifier,
                        image_path=image_path,
                        prompt_variant=prompt_variant,
                    )
                )
            except Exception as exc:
                rows.append(
                    {
                        "image_path": str(image_path.resolve()),
                        "track_id": _extract_track_id(image_path),
                        "prompt_id": str(prompt_variant["id"]),
                        "task_prompt": str(prompt_variant["task_prompt"]),
                        "prompt_text": str(prompt_variant["prompt_text"]),
                        "raw_response": None,
                        "parsed_label": "UNKNOWN",
                        "normalization_reason": "benchmark_exception",
                        "response_kind": "error",
                        "inference_duration_ms": None,
                        "status": "error",
                        "error": str(exc),
                    }
                )
    summary = _build_prompt_summary(rows)
    csv_path = output_dir / "prompt_results.csv"
    json_path = output_dir / "prompt_results.json"
    summary_path = output_dir / "prompt_summary.json"
    manual_review_path = output_dir / "manual_review.csv"
    comparison_path = output_dir / "prompt_comparison.csv"

    _write_csv(
        csv_path,
        rows,
        [
            "image_path",
            "track_id",
            "prompt_id",
            "task_prompt",
            "prompt_text",
            "raw_response",
            "parsed_label",
            "normalization_reason",
            "response_kind",
            "inference_duration_ms",
            "status",
            "error",
        ],
    )
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manual_review_rows = [
        {
            "image_path": row["image_path"],
            "track_id": row["track_id"],
            "prompt_id": row["prompt_id"],
            "raw_response": row["raw_response"],
            "parsed_label": row["parsed_label"],
            "normalization_reason": row["normalization_reason"],
            "manual_colour": "",
            "is_correct": "",
            "review_notes": "",
        }
        for row in rows
    ]
    _write_csv(
        manual_review_path,
        manual_review_rows,
        [
            "image_path",
            "track_id",
            "prompt_id",
            "raw_response",
            "parsed_label",
            "normalization_reason",
            "manual_colour",
            "is_correct",
            "review_notes",
        ],
    )

    comparison_rows: list[dict[str, Any]] = []
    grouped_by_image: dict[str, dict[str, str]] = {}
    for row in rows:
        grouped_by_image.setdefault(row["image_path"], {})[row["prompt_id"]] = str(row["parsed_label"])
    for image_path, prompt_map in sorted(grouped_by_image.items()):
        comparison_row = {"image_path": image_path}
        for prompt_variant in prompt_variants:
            comparison_row[str(prompt_variant["id"])] = prompt_map.get(str(prompt_variant["id"]), "")
        comparison_rows.append(comparison_row)
    _write_csv(comparison_path, comparison_rows, ["image_path", *[variant["id"] for variant in prompt_variants]])

    return {
        "image_count": len(images),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "summary_path": str(summary_path),
        "manual_review_path": str(manual_review_path),
        "comparison_path": str(comparison_path),
        "summary": summary,
        "backend_metrics": dict(getattr(local_backend, "metrics", {})),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Florence vehicle-colour prompt variants.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Crop image path or directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for CSV/JSON outputs.")
    parser.add_argument("--device", type=str, default="auto", help="Runtime device, e.g. auto/cpu/cuda:0.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional maximum number of images.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    result = run_benchmark(
        input_path=args.input_dir,
        output_dir=args.output_dir,
        device=args.device,
        max_images=args.max_images,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
