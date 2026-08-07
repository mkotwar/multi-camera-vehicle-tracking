from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vehicle_enrichment.enrichment_manager import normalize_vehicle_enrichment_config
from src.vehicle_enrichment.shared import FlorenceBackend, FlorenceBackendConfig


OLD_PROMPT = "What is the body type of this car? Answer with only one of: sedan, hatchback, suv, mpv, van, pickup, coupe, convertible, wagon."
NEW_PROMPT = (
    "Classify only the BODY SHAPE of this car.\n\n"
    "Choose exactly one:\n"
    "sedan\n"
    "hatchback\n"
    "suv\n"
    "mpv\n\n"
    "Do not identify the brand or model.\n"
    'Do not answer "car".\n'
    "Do not explain your answer.\n"
    "Return exactly one label from the list above."
)
TASK_TOKEN = "<VQA>"
BENCHMARK_LABELS = {"SEDAN", "HATCHBACK", "SUV", "MPV"}
BRAND_LIKE_WORDS = {
    "hyundai",
    "honda",
    "audi",
    "toyota",
    "maruti",
    "suzuki",
    "kia",
    "mahindra",
    "tata",
    "ford",
    "bmw",
    "mercedes",
    "skoda",
    "volkswagen",
    "nissan",
    "renault",
}


def _build_backend(config_path: Path, *, device: str = "auto") -> FlorenceBackend:
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    normalized = normalize_vehicle_enrichment_config(raw_config.get("vehicle_enrichment", {}))
    shared_cfg = {key: value for key, value in normalized["shared_florence"].items() if key in FlorenceBackendConfig.__annotations__}
    shared_cfg["adapter_enabled"] = False
    if device:
        shared_cfg["device"] = device
    return FlorenceBackend(FlorenceBackendConfig(**shared_cfg), logger=logging.getLogger("body_type_prompt_comparison"))


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def _normalize_benchmark_label(raw_value: str) -> tuple[str, bool]:
    cleaned = _clean_text(raw_value)
    if not cleaned or cleaned in {"unknown", "unanswerable", "car", "vehicle", "yes", "no", "qa"}:
        return "UNKNOWN", False
    if "sport utility vehicle" in cleaned or "sports utility vehicle" in cleaned or cleaned == "suv" or " suv " in f" {cleaned} ":
        return "SUV", True
    if cleaned == "sedan" or " sedan " in f" {cleaned} " or "saloon" in cleaned:
        return "SEDAN", True
    if cleaned == "hatchback" or "hatch back" in cleaned:
        return "HATCHBACK", True
    if cleaned == "mpv" or "muv" in cleaned or "multi purpose vehicle" in cleaned or "multi-purpose vehicle" in cleaned or "minivan" in cleaned:
        return "MPV", True
    return "UNKNOWN", False


def _response_bucket(raw_value: str) -> str:
    cleaned = _clean_text(raw_value)
    if not cleaned:
        return "<empty>"
    if cleaned in {"sedan", "hatchback", "suv", "mpv", "car", "unanswerable", "yes", "no", "qa"}:
        return cleaned
    if cleaned in BRAND_LIKE_WORDS:
        return f"brand:{cleaned}"
    return f"other:{cleaned}"


def _run_once(backend: Any, image_path: Path, prompt: str) -> tuple[str, str, bool, float]:
    image = cv2.imread(str(image_path))
    if image is None or image.size == 0:
        raise FileNotFoundError(f"Image could not be decoded: {image_path}")
    response = backend.run_task(image, TASK_TOKEN, prompt, adapter_active=False)
    if response["status"] != "completed":
        raise RuntimeError(str(response.get("reason") or "Florence body-type benchmark inference failed"))
    payload = dict(response.get("payload") or {})
    parsed_answer = payload.get("parsed_answer")
    if isinstance(parsed_answer, dict):
        raw_response = str(parsed_answer.get(TASK_TOKEN) or payload.get("generated_text") or "").strip()
    else:
        raw_response = str(parsed_answer or payload.get("generated_text") or "").strip()
    label, valid = _normalize_benchmark_label(raw_response)
    return raw_response, label, valid, float(payload.get("inference_duration_ms", 0.0) or 0.0)


def _load_crop_rows(manual_review_csv: Path) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(manual_review_csv.open("r", encoding="utf-8", newline="")))
    result: list[dict[str, Any]] = []
    for row in rows:
        crop_path = Path(str(row.get("crop_path") or ""))
        if not crop_path.exists():
            continue
        image = cv2.imread(str(crop_path))
        if image is None or image.size == 0:
            continue
        height, width = image.shape[:2]
        result.append(
            {
                "track_id": str(row.get("track_id") or ""),
                "frame_number": int(row.get("frame_number") or -1),
                "crop_path": str(crop_path),
                "crop_width": width,
                "crop_height": height,
            }
        )
    return result


def _build_track_summary(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["track_id"])].append(row)
    per_track: dict[str, Any] = {}
    counters = {
        "tracks_with_unanimous_valid_prediction": 0,
        "tracks_with_mixed_valid_prediction": 0,
        "tracks_with_all_unknown": 0,
        "tracks_with_valid_unknown_mixture": 0,
    }
    for track_id, items in sorted(grouped.items()):
        labels = [str(item[f"{prefix}_prompt_parsed_body_type"]) for item in items]
        valid = [label for label in labels if label in BENCHMARK_LABELS]
        agreement = 0.0
        if labels:
            top_count = max(Counter(labels).values())
            agreement = round((top_count / len(labels)) * 100.0, 2)
        if len(valid) == len(labels) and len(set(valid)) == 1:
            counters["tracks_with_unanimous_valid_prediction"] += 1
        elif len(valid) == 0:
            counters["tracks_with_all_unknown"] += 1
        elif len(valid) == len(labels) and len(set(valid)) > 1:
            counters["tracks_with_mixed_valid_prediction"] += 1
        else:
            counters["tracks_with_valid_unknown_mixture"] += 1
        per_track[track_id] = {
            "predictions": labels,
            "agreement_percent": agreement,
        }
    return {"counts": counters, "per_track": per_track}


def _build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"total_crops": len(rows)}
    for prefix in ("old", "new"):
        valid = [row for row in rows if bool(row[f"{prefix}_prompt_valid"])]
        unknown = [row for row in rows if str(row[f"{prefix}_prompt_parsed_body_type"]) == "UNKNOWN"]
        raw_counts = Counter(_response_bucket(str(row[f"{prefix}_prompt_raw_response"])) for row in rows)
        parsed_counts = Counter(str(row[f"{prefix}_prompt_parsed_body_type"]) for row in rows)
        summary[prefix] = {
            "valid_parsed_crops": len(valid),
            "unknown_crops": len(unknown),
            "valid_parse_rate": round((len(valid) / len(rows)) * 100.0, 2) if rows else 0.0,
            "unknown_rate": round((len(unknown) / len(rows)) * 100.0, 2) if rows else 0.0,
            "raw_response_distribution": dict(raw_counts),
            "parsed_distribution": dict(parsed_counts),
            "track_level": _build_track_summary(rows, prefix),
        }
    return summary


def run_prompt_comparison(
    *,
    manual_review_csv: Path,
    config_path: Path,
    output_dir: Path,
    device: str = "auto",
    backend: Any | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_rows = _load_crop_rows(manual_review_csv)
    local_backend = backend or _build_backend(config_path, device=device)
    comparison_rows: list[dict[str, Any]] = []
    for row in crop_rows:
        crop_path = Path(row["crop_path"])
        old_raw, old_label, old_valid, _old_ms = _run_once(local_backend, crop_path, OLD_PROMPT)
        new_raw, new_label, new_valid, _new_ms = _run_once(local_backend, crop_path, NEW_PROMPT)
        comparison_rows.append(
            {
                **row,
                "old_prompt_raw_response": old_raw,
                "old_prompt_parsed_body_type": old_label,
                "old_prompt_valid": old_valid,
                "new_prompt_raw_response": new_raw,
                "new_prompt_parsed_body_type": new_label,
                "new_prompt_valid": new_valid,
                "ground_truth_body_type": "",
                "old_prompt_correct": "",
                "new_prompt_correct": "",
                "notes": "",
            }
        )
    summary = _build_summary(comparison_rows)
    csv_path = output_dir / "body_type_prompt_comparison.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "track_id",
                "frame_number",
                "crop_path",
                "crop_width",
                "crop_height",
                "old_prompt_raw_response",
                "old_prompt_parsed_body_type",
                "old_prompt_valid",
                "new_prompt_raw_response",
                "new_prompt_parsed_body_type",
                "new_prompt_valid",
                "ground_truth_body_type",
                "old_prompt_correct",
                "new_prompt_correct",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_lines = [
        "# Body-Type Prompt Comparison",
        "",
        f"- Crops tested: `{summary['total_crops']}`",
        f"- Old valid parse rate: `{summary['old']['valid_parse_rate']}%`",
        f"- New valid parse rate: `{summary['new']['valid_parse_rate']}%`",
        f"- Old UNKNOWN rate: `{summary['old']['unknown_rate']}%`",
        f"- New UNKNOWN rate: `{summary['new']['unknown_rate']}%`",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run A/B Florence CAR body-type prompt comparison on a fixed crop set.")
    parser.add_argument(
        "--manual-review-csv",
        default=str(REPO_ROOT / "diagnostics" / "body_type_benchmark" / "body_type_manual_review.csv"),
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "config.validation_car_body_type.yaml"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "diagnostics" / "body_type_prompt_comparison"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    result = run_prompt_comparison(
        manual_review_csv=Path(args.manual_review_csv),
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        device=args.device,
    )
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
