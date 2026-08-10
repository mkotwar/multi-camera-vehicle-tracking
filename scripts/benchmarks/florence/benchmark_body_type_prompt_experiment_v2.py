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


TASK_TOKEN = "<VQA>"
CAPTION_TASK_TOKEN = "<CAPTION>"
BASELINE_PROMPT = "What is the body type of this car? Answer with only one of: sedan, hatchback, suv, mpv, van, pickup, coupe, convertible, wagon."
PROMPT_A = "What type of car is shown in this image?\nAnswer with one word only:\nsedan, hatchback, suv, or mpv."
PROMPT_B = "Look at the overall shape of the car.\nChoose the closest body style:\nsedan\nhatchback\nsuv\nmpv\n\nReturn only the chosen body style."
PROMPT_C = "Which body style best describes this car:\nsedan, hatchback, suv, or mpv?"
METHODS: tuple[tuple[str, str, str | None], ...] = (
    ("baseline", TASK_TOKEN, BASELINE_PROMPT),
    ("prompt_a", TASK_TOKEN, PROMPT_A),
    ("prompt_b", TASK_TOKEN, PROMPT_B),
    ("prompt_c", TASK_TOKEN, PROMPT_C),
    ("caption", CAPTION_TASK_TOKEN, None),
)
VALID_LABELS = {"SEDAN", "HATCHBACK", "SUV", "MPV"}
BRAND_WORDS = {
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
    return FlorenceBackend(FlorenceBackendConfig(**shared_cfg), logger=logging.getLogger("body_type_prompt_experiment_v2"))


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def _parse_body_type_4class(raw_value: str) -> tuple[str, bool]:
    cleaned = _clean_text(raw_value)
    if cleaned in {"", "unknown", "car", "vehicle", "yes", "no", "unanswerable", "qa"}:
        return "UNKNOWN", False
    if cleaned in BRAND_WORDS:
        return "UNKNOWN", False
    if "sport utility vehicle" in cleaned or "sports utility vehicle" in cleaned or cleaned == "suv" or " suv " in f" {cleaned} ":
        return "SUV", True
    if cleaned == "sedan" or " sedan " in f" {cleaned} " or "saloon" in cleaned:
        return "SEDAN", True
    if cleaned == "hatchback" or "hatch back" in cleaned:
        return "HATCHBACK", True
    if cleaned == "mpv" or "muv" in cleaned or "multi purpose vehicle" in cleaned or "multi-purpose vehicle" in cleaned or "multipurpose vehicle" in cleaned:
        return "MPV", True
    return "UNKNOWN", False


def _response_bucket(raw_value: str) -> str:
    cleaned = _clean_text(raw_value)
    if not cleaned:
        return "<empty>"
    if cleaned in {"sedan", "hatchback", "suv", "mpv", "car", "vehicle", "yes", "no", "unanswerable", "qa"}:
        return cleaned
    if cleaned in BRAND_WORDS:
        return f"brand:{cleaned}"
    return f"other:{cleaned}"


def _run_once(backend: Any, image_path: Path, *, task_token: str, prompt: str | None) -> tuple[str, str, bool]:
    image = cv2.imread(str(image_path))
    if image is None or image.size == 0:
        raise FileNotFoundError(f"Image could not be decoded: {image_path}")
    response = backend.run_task(image, task_token, prompt, adapter_active=False)
    if response["status"] != "completed":
        raise RuntimeError(str(response.get("reason") or "Florence inference failed"))
    payload = dict(response.get("payload") or {})
    parsed_answer = payload.get("parsed_answer")
    if isinstance(parsed_answer, dict):
        raw_response = str(parsed_answer.get(task_token) or payload.get("generated_text") or "").strip()
    else:
        raw_response = str(parsed_answer or payload.get("generated_text") or "").strip()
    label, valid = _parse_body_type_4class(raw_response)
    return raw_response, label, valid


def _load_rows(source_csv: Path) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(source_csv.open("r", encoding="utf-8", newline="")))
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


def _build_track_analysis(rows: list[dict[str, Any]], method_name: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["track_id"])].append(row)
    counters = {
        "tracks_with_unanimous_valid_prediction": 0,
        "tracks_with_mixed_valid_predictions": 0,
        "tracks_with_valid_unknown": 0,
        "tracks_with_all_unknown": 0,
    }
    per_track: dict[str, Any] = {}
    for track_id, items in sorted(grouped.items()):
        labels = [str(item[f"{method_name}_parsed_body_type"]) for item in items]
        valid = [label for label in labels if label in VALID_LABELS]
        if len(valid) == len(labels) and len(set(valid)) == 1:
            counters["tracks_with_unanimous_valid_prediction"] += 1
        elif len(valid) == len(labels) and len(set(valid)) > 1:
            counters["tracks_with_mixed_valid_predictions"] += 1
        elif len(valid) == 0:
            counters["tracks_with_all_unknown"] += 1
        else:
            counters["tracks_with_valid_unknown"] += 1
        per_track[track_id] = labels
    return {"counts": counters, "per_track": per_track}


def _build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"total_crops": len(rows)}
    for method_name, _task, _prompt in METHODS:
        valid = [row for row in rows if bool(row[f"{method_name}_valid"])]
        unknown = [row for row in rows if str(row[f"{method_name}_parsed_body_type"]) == "UNKNOWN"]
        raw_counts = Counter(_response_bucket(str(row[f"{method_name}_raw_response"])) for row in rows)
        parsed_counts = Counter(str(row[f"{method_name}_parsed_body_type"]) for row in rows)
        summary[method_name] = {
            "valid_parsed_crops": len(valid),
            "unknown_crops": len(unknown),
            "valid_parse_rate": round((len(valid) / len(rows)) * 100.0, 2) if rows else 0.0,
            "unknown_rate": round((len(unknown) / len(rows)) * 100.0, 2) if rows else 0.0,
            "raw_response_distribution": dict(raw_counts),
            "parsed_distribution": dict(parsed_counts),
            "track_level": _build_track_analysis(rows, method_name),
        }
    return summary


def _report_lines(summary: dict[str, Any]) -> list[str]:
    lines = [
        "# Body-Type Prompt Experiment V2",
        "",
        "| Method | Valid crops | UNKNOWN | Valid parse rate | All-UNKNOWN tracks |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method_name in ["baseline", "prompt_a", "prompt_b", "prompt_c", "caption"]:
        payload = summary[method_name]
        lines.append(
            "| {name} | {valid} | {unknown} | {rate}% | {all_unknown} |".format(
                name=method_name,
                valid=payload["valid_parsed_crops"],
                unknown=payload["unknown_crops"],
                rate=payload["valid_parse_rate"],
                all_unknown=payload["track_level"]["counts"]["tracks_with_all_unknown"],
            )
        )
    return lines


def run_experiment(
    *,
    source_csv: Path,
    config_path: Path,
    output_dir: Path,
    device: str = "auto",
    backend: Any | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(source_csv)
    local_backend = backend or _build_backend(config_path, device=device)
    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        crop_path = Path(row["crop_path"])
        enriched = dict(row)
        for method_name, task_token, prompt in METHODS:
            raw_response, parsed_label, valid = _run_once(local_backend, crop_path, task_token=task_token, prompt=prompt)
            enriched[f"{method_name}_raw_response"] = raw_response
            enriched[f"{method_name}_parsed_body_type"] = parsed_label
            enriched[f"{method_name}_valid"] = valid
        enriched["ground_truth_body_type"] = ""
        enriched["baseline_correct"] = ""
        enriched["prompt_a_correct"] = ""
        enriched["prompt_b_correct"] = ""
        enriched["prompt_c_correct"] = ""
        enriched["caption_correct"] = ""
        enriched["notes"] = ""
        enriched_rows.append(enriched)
    summary = _build_summary(enriched_rows)
    csv_path = output_dir / "body_type_prompt_experiment_v2.csv"
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
                "baseline_raw_response",
                "baseline_parsed_body_type",
                "baseline_valid",
                "prompt_a_raw_response",
                "prompt_a_parsed_body_type",
                "prompt_a_valid",
                "prompt_b_raw_response",
                "prompt_b_parsed_body_type",
                "prompt_b_valid",
                "prompt_c_raw_response",
                "prompt_c_parsed_body_type",
                "prompt_c_valid",
                "caption_raw_response",
                "caption_parsed_body_type",
                "caption_valid",
                "ground_truth_body_type",
                "baseline_correct",
                "prompt_a_correct",
                "prompt_b_correct",
                "prompt_c_correct",
                "caption_correct",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(enriched_rows)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text("\n".join(_report_lines(summary)), encoding="utf-8")
    return {
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "summary": summary,
        "caption_supported": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run controlled body-type prompt experiment v2 on a fixed crop set.")
    parser.add_argument(
        "--source-csv",
        default=str(REPO_ROOT / "diagnostics" / "body_type_prompt_comparison" / "body_type_prompt_comparison.csv"),
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "archive" / "config.validation_car_body_type.yaml"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "diagnostics" / "body_type_prompt_experiment_v2"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    result = run_experiment(
        source_csv=Path(args.source_csv),
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        device=args.device,
    )
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
