from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vehicle_enrichment.colour.classifier import VehicleColourClassifier
from src.vehicle_enrichment.shared.florence_backend import FlorenceBackend, FlorenceBackendConfig


LOGGER = logging.getLogger("rejected_crop_florence_test")

RESULT_COLUMNS = [
    "camera_id",
    "local_track_id",
    "frame_number",
    "crop_path",
    "width",
    "height",
    "inside_capture_zone",
    "normal_evidence_eligible",
    "normal_florence_eligible",
    "eligibility_bypassed",
    "raw_response",
    "parsed_colour",
    "parsed_valid",
    "parse_status",
    "inference_duration_ms",
    "status",
    "error",
]


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return payload


def resolve_colour_diagnostic_settings(config: dict[str, Any]) -> dict[str, Any]:
    vehicle_enrichment = dict(config.get("vehicle_enrichment") or {})
    canonical_florence = dict(vehicle_enrichment.get("florence") or {})
    canonical_enrichment = dict(vehicle_enrichment.get("enrichment") or {})
    canonical_colour = dict(canonical_enrichment.get("colour") or {})
    shared = dict(vehicle_enrichment.get("shared_florence") or {})
    vehicle_attributes = dict(vehicle_enrichment.get("vehicle_attributes") or {})
    colour = dict(vehicle_attributes.get("colour") or {})
    florence_override = dict(vehicle_attributes.get("florence") or {})
    if canonical_florence:
        merged_florence = {
            "enabled": bool(canonical_florence.get("enabled", True)),
            "backend": str(canonical_florence.get("backend", "florence")),
            "base_model_id": str(canonical_florence.get("base_model_id") or canonical_florence.get("model_id") or ""),
            "processor_path": str(canonical_florence.get("processor_path") or canonical_florence.get("base_model_id") or canonical_florence.get("model_id") or ""),
            "adapter_path": "",
            "adapter_enabled": False,
            "device": str(canonical_florence.get("device", "auto")),
            "dtype": str(canonical_florence.get("dtype", "auto")),
            "trust_remote_code": bool(canonical_florence.get("trust_remote_code", True)),
            "attention_implementation": str(canonical_florence.get("attention_implementation", "eager")),
            "max_new_tokens": int(canonical_florence.get("max_new_tokens", 16)),
            "num_beams": int(canonical_florence.get("num_beams", 1)),
            "use_cache": bool(canonical_florence.get("use_cache", True)),
            "local_files_only": bool(canonical_florence.get("local_files_only", True)),
            "lazy_load": bool(canonical_florence.get("lazy_load", True)),
        }
        task_token = str(canonical_colour.get("task_token") or "<VQA>")
        prompt = str(canonical_colour.get("prompt") or "What colour is the vehicle?")
        generation = dict(canonical_colour.get("generation") or {})
    else:
        merged_florence = dict(shared)
        merged_florence.update(florence_override)
        merged_florence["adapter_enabled"] = False
        merged_florence["adapter_path"] = ""
        task_token = str(colour.get("task_token") or vehicle_attributes.get("task_token") or "<VQA>")
        prompt = str(colour.get("prompt") or vehicle_attributes.get("prompt") or "What colour is the vehicle?")
        generation = dict(colour.get("generation") or {})
    if not generation:
        generation = {
            "max_new_tokens": int(merged_florence.get("max_new_tokens", 16)),
            "num_beams": int(merged_florence.get("num_beams", 1)),
            "do_sample": False,
            "use_cache": bool(merged_florence.get("use_cache", True)),
            "early_stopping": False,
        }
    return {
        "shared_florence": merged_florence,
        "task_token": task_token,
        "prompt": prompt,
        "generation": generation,
    }


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


def load_track_crop_manifest(run_dir: Path) -> list[dict[str, Any]]:
    manifest_path = run_dir / "04_track_crops" / "track_crop_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Track crop manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_track_id(track_id: str) -> str:
    normalized = str(track_id or "").strip()
    if not normalized:
        raise ValueError("Track id must not be empty.")
    return normalized


def bucket_for_width(width: int) -> str:
    if width < 80:
        return "<80"
    if width < 100:
        return "80-99"
    if width < 120:
        return "100-119"
    return ">=120"


def bucket_sort_key(bucket: str) -> int:
    return {"<80": 0, "80-99": 1, "100-119": 2, ">=120": 3}.get(bucket, 99)


def filter_track_rows(rows: list[dict[str, Any]], track_id: str) -> list[dict[str, Any]]:
    normalized_track_id = normalize_track_id(track_id)
    filtered = [row for row in rows if str(row.get("local_track_id") or "").strip() == normalized_track_id]
    return sorted(filtered, key=lambda row: parse_int(row.get("frame_number")))


def _track_is_rejected_motorcycle(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and all(
        str(row.get("vehicle_class") or "").strip().lower() == "motorcycle" and not parse_bool(row.get("florence_eligible"))
        for row in rows
    )


def sample_other_motorcycle_rows(rows: list[dict[str, Any]], *, exclude_track_ids: set[str], max_tracks: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        track_id = str(row.get("local_track_id") or "").strip()
        if not track_id or track_id in exclude_track_ids:
            continue
        grouped[track_id].append(row)

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for track_id, group in grouped.items():
        ordered = sorted(group, key=lambda row: parse_int(row.get("frame_number")))
        if not _track_is_rejected_motorcycle(ordered):
            continue
        representative = max(
            ordered,
            key=lambda row: (
                parse_bool(row.get("inside_capture_zone")),
                parse_int(row.get("crop_width")),
                parse_int(row.get("crop_height")),
                parse_int(row.get("frame_number")),
            ),
        )
        candidates[bucket_for_width(parse_int(representative.get("crop_width")))].append(representative)

    selected: list[dict[str, Any]] = []
    used_tracks: set[str] = set()
    for bucket in sorted(candidates, key=bucket_sort_key):
        for row in sorted(
            candidates[bucket],
            key=lambda item: (
                not parse_bool(item.get("inside_capture_zone")),
                -parse_int(item.get("crop_width")),
                -parse_int(item.get("crop_height")),
            ),
        ):
            track_id = str(row.get("local_track_id") or "")
            if track_id in used_tracks:
                continue
            selected.append(row)
            used_tracks.add(track_id)
            if len(selected) >= max_tracks:
                return selected
            break

    remaining = [
        row
        for bucket in sorted(candidates, key=bucket_sort_key)
        for row in candidates[bucket]
        if str(row.get("local_track_id") or "") not in used_tracks
    ]
    for row in sorted(
        remaining,
        key=lambda item: (
            bucket_sort_key(bucket_for_width(parse_int(item.get("crop_width")))),
            -parse_int(item.get("crop_width")),
            -parse_int(item.get("crop_height")),
        ),
    ):
        selected.append(row)
        used_tracks.add(str(row.get("local_track_id") or ""))
        if len(selected) >= max_tracks:
            break
    return selected


def copy_crop_to_diagnostic_folder(crop_path: Path, diagnostics_root: Path, camera_id: str, local_track_id: str) -> Path:
    safe_track_name = str(local_track_id).split(":")[-1]
    destination_dir = diagnostics_root / camera_id / safe_track_name
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / crop_path.name
    shutil.copy2(crop_path, destination_path)
    return destination_path


def run_single_crop_diagnostic(
    row: dict[str, Any],
    *,
    backend: Any,
    task_token: str,
    prompt: str,
    generation_overrides: dict[str, Any],
    diagnostics_root: Path,
) -> dict[str, Any]:
    crop_path = Path(str(row.get("crop_path") or ""))
    result: dict[str, Any] = {
        "camera_id": str(row.get("camera_id") or ""),
        "local_track_id": str(row.get("local_track_id") or ""),
        "frame_number": parse_int(row.get("frame_number")),
        "crop_path": str(crop_path),
        "width": parse_int(row.get("crop_width")),
        "height": parse_int(row.get("crop_height")),
        "inside_capture_zone": parse_bool(row.get("inside_capture_zone")),
        "normal_evidence_eligible": parse_bool(row.get("evidence_eligible")),
        "normal_florence_eligible": parse_bool(row.get("florence_eligible")),
        "eligibility_bypassed": True,
        "raw_response": "",
        "parsed_colour": "UNKNOWN",
        "parsed_valid": False,
        "parse_status": "not_run",
        "inference_duration_ms": None,
        "status": "error",
        "error": "",
    }
    if not crop_path.exists():
        result["error"] = f"Crop image does not exist: {crop_path}"
        return result

    copied_crop = copy_crop_to_diagnostic_folder(
        crop_path,
        diagnostics_root,
        result["camera_id"],
        result["local_track_id"],
    )
    result["diagnostic_crop_copy_path"] = str(copied_crop)

    image = cv2.imread(str(crop_path))
    if image is None or image.size == 0:
        result["error"] = f"Crop image could not be decoded: {crop_path}"
        return result

    response = backend.run_task(
        image,
        task_token,
        prompt,
        adapter_active=False,
        generation_overrides=generation_overrides,
    )
    if str(response.get("status")) != "completed":
        result["error"] = str(response.get("reason") or "Florence inference failed.")
        return result

    payload = dict(response.get("payload") or {})
    raw_response = VehicleColourClassifier._extract_colour_text(payload)
    parsed_colour, parse_status = VehicleColourClassifier.normalize_label(raw_response)
    result["raw_response"] = raw_response
    result["parsed_colour"] = parsed_colour
    result["parsed_valid"] = parsed_colour != "UNKNOWN"
    result["parse_status"] = parse_status
    result["inference_duration_ms"] = float(payload.get("inference_duration_ms", 0.0))
    result["status"] = "completed"
    result["error"] = ""
    return result


def build_bucket_stats(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[bucket_for_width(parse_int(row.get("width")))].append(row)
    stats: list[dict[str, Any]] = []
    for bucket in sorted(grouped, key=bucket_sort_key):
        rows = grouped[bucket]
        tested = len(rows)
        valid = sum(1 for row in rows if bool(row.get("parsed_valid")))
        stats.append(
            {
                "bucket": bucket,
                "tested": tested,
                "valid": valid,
                "valid_percentage": round((100.0 * valid / tested), 2) if tested else 0.0,
            }
        )
    return stats


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in RESULT_COLUMNS})


def write_results_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_rejected_crop_diagnostic(
    *,
    run_dir: Path,
    config_path: Path,
    track_id: str,
    sample_other_motorcycles_count: int,
    backend: Any | None = None,
    command_text: str = "",
) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    settings = resolve_colour_diagnostic_settings(config)
    backend_config = build_backend_config(settings["shared_florence"])
    manifest_rows = load_track_crop_manifest(run_dir)
    primary_rows = filter_track_rows(manifest_rows, track_id)
    if not primary_rows:
        raise ValueError(f"No rows found for track: {track_id}")

    diagnostics_root = run_dir / "diagnostics" / "rejected_crop_florence_test"
    diagnostics_root.mkdir(parents=True, exist_ok=True)

    active_backend = backend or FlorenceBackend(backend_config, logger=LOGGER, adapter_enabled_override=False)
    load_started_at = time.perf_counter()
    active_backend.load()
    model_load_time_ms = (time.perf_counter() - load_started_at) * 1000.0

    all_rows = list(primary_rows)
    sampled_rows = sample_other_motorcycle_rows(
        manifest_rows,
        exclude_track_ids={normalize_track_id(track_id)},
        max_tracks=max(0, sample_other_motorcycles_count),
    )
    all_rows.extend(sampled_rows)

    results: list[dict[str, Any]] = []
    for row in all_rows:
        results.append(
            run_single_crop_diagnostic(
                row,
                backend=active_backend,
                task_token=settings["task_token"],
                prompt=settings["prompt"],
                generation_overrides=settings["generation"],
                diagnostics_root=diagnostics_root,
            )
        )

    bucket_stats = build_bucket_stats(results)
    completed = [row for row in results if row.get("status") == "completed" and row.get("inference_duration_ms") is not None]
    average_inference_time_ms = round(
        sum(float(row["inference_duration_ms"]) for row in completed) / len(completed),
        3,
    ) if completed else 0.0

    output_payload = {
        "command": command_text,
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "model_path": backend_config.base_model_id,
        "processor_path": backend_config.processor_path,
        "adapter_loaded": bool(getattr(active_backend, "adapter_active", False)),
        "task_token": settings["task_token"],
        "prompt": settings["prompt"],
        "generation_overrides": dict(settings["generation"]),
        "model_load_time_ms": round(model_load_time_ms, 3),
        "device": str(getattr(active_backend, "resolved_device", "unknown")),
        "dtype": str(getattr(active_backend, "resolved_dtype", "unknown")),
        "inference_call_count": len(results),
        "average_inference_time_ms": average_inference_time_ms,
        "track_192_results": [row for row in results if row.get("local_track_id") == normalize_track_id(track_id)],
        "other_results": [row for row in results if row.get("local_track_id") != normalize_track_id(track_id)],
        "size_bucket_stats": bucket_stats,
        "results": results,
    }

    csv_path = diagnostics_root / "rejected_crop_florence_results.csv"
    json_path = diagnostics_root / "rejected_crop_florence_results.json"
    write_results_csv(csv_path, results)
    write_results_json(json_path, output_payload)
    output_payload["csv_path"] = str(csv_path)
    output_payload["json_path"] = str(json_path)

    if backend is None and hasattr(active_backend, "close"):
        active_backend.close()
    return output_payload


def print_summary(payload: dict[str, Any], track_id: str) -> None:
    print(f"Eligibility-bypass diagnostic complete for {track_id}")
    print(f"Model path: {payload['model_path']}")
    print(f"Adapter loaded: {payload['adapter_loaded']}")
    print(f"Device: {payload['device']}")
    print(f"Dtype: {payload['dtype']}")
    print(f"Model load time ms: {payload['model_load_time_ms']}")
    print(f"Inference call count: {payload['inference_call_count']}")
    print(f"Average inference time ms: {payload['average_inference_time_ms']}")
    print("")
    print("TRACK_192 results")
    print("frame | size | raw_response | parsed_colour | valid")
    for row in payload["track_192_results"]:
        size = f"{row['width']}x{row['height']}"
        print(f"{row['frame_number']} | {size} | {row['raw_response']} | {row['parsed_colour']} | {str(bool(row['parsed_valid'])).lower()}")
    if payload["other_results"]:
        print("")
        print("Other tested motorcycle crops")
        print("track | frame | size | raw_response | parsed_colour | valid")
        for row in payload["other_results"]:
            size = f"{row['width']}x{row['height']}"
            print(f"{row['local_track_id']} | {row['frame_number']} | {size} | {row['raw_response']} | {row['parsed_colour']} | {str(bool(row['parsed_valid'])).lower()}")
    print("")
    print("Valid prediction percentage by width bucket")
    for bucket in payload["size_bucket_stats"]:
        print(f"{bucket['bucket']}: tested {bucket['tested']} valid {bucket['valid']} valid_percentage {bucket['valid_percentage']}")
    print("")
    print(f"CSV: {payload['csv_path']}")
    print(f"JSON: {payload['json_path']}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an eligibility-bypass Florence diagnostic on rejected saved crops.")
    parser.add_argument("--run-dir", required=True, help="Path to an existing run directory containing 04_track_crops/track_crop_manifest.csv")
    parser.add_argument(
        "--config",
        default="config/archive/config.validation_base_colour_only.yaml",
        help="YAML config to reuse Florence colour settings from.",
    )
    parser.add_argument("--track-id", required=True, help="Track id to test all saved crops for, e.g. CAM_001:TRACK_192")
    parser.add_argument("--sample-other-motorcycles", type=int, default=0, help="Number of additional rejected motorcycle tracks to sample.")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = build_arg_parser().parse_args(argv)
    payload = run_rejected_crop_diagnostic(
        run_dir=Path(args.run_dir),
        config_path=Path(args.config),
        track_id=args.track_id,
        sample_other_motorcycles_count=max(0, int(args.sample_other_motorcycles)),
        command_text=" ".join(sys.argv),
    )
    print_summary(payload, args.track_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
