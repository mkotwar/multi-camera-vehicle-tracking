from __future__ import annotations

import csv
import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np
import supervision as sv
import yaml

from .camera_reader import VideoCameraReader
from .detector_tracker import (
    EDGE_MODE_A,
    EDGE_MODE_B,
    EDGE_MODE_C,
    BBoxQualityProfile,
    VehicleDetectorTracker,
)
from .logging_setup import setup_logging
from .models import Detection, FramePacket
from .output_writer import RunOutputManager
from .pipeline import _load_raw_config, _validate_config


PROFILE_LIBRARY = {
    "car": {
        "CAR_A": {"minimum_width_pixels": 25, "minimum_height_pixels": 25, "minimum_area_ratio": 0.0008, "minimum_aspect_ratio": 0.30, "maximum_aspect_ratio": 4.00},
        "CAR_B": {"minimum_width_pixels": 35, "minimum_height_pixels": 30, "minimum_area_ratio": 0.0015, "minimum_aspect_ratio": 0.40, "maximum_aspect_ratio": 3.50},
        "CAR_C": {"minimum_width_pixels": 45, "minimum_height_pixels": 35, "minimum_area_ratio": 0.0025, "minimum_aspect_ratio": 0.50, "maximum_aspect_ratio": 3.20},
        "CAR_D": {"minimum_width_pixels": 55, "minimum_height_pixels": 40, "minimum_area_ratio": 0.0035, "minimum_aspect_ratio": 0.60, "maximum_aspect_ratio": 3.00},
    },
    "motorcycle": {
        "MOTORCYCLE_A": {"minimum_width_pixels": 12, "minimum_height_pixels": 15, "minimum_area_ratio": 0.0002, "minimum_aspect_ratio": 0.15, "maximum_aspect_ratio": 3.50},
        "MOTORCYCLE_B": {"minimum_width_pixels": 18, "minimum_height_pixels": 20, "minimum_area_ratio": 0.0004, "minimum_aspect_ratio": 0.18, "maximum_aspect_ratio": 3.20},
        "MOTORCYCLE_C": {"minimum_width_pixels": 24, "minimum_height_pixels": 25, "minimum_area_ratio": 0.0007, "minimum_aspect_ratio": 0.20, "maximum_aspect_ratio": 3.00},
        "MOTORCYCLE_D": {"minimum_width_pixels": 30, "minimum_height_pixels": 30, "minimum_area_ratio": 0.0010, "minimum_aspect_ratio": 0.25, "maximum_aspect_ratio": 2.80},
    },
    "3wheeler": {
        "3WHEELER_A": {"minimum_width_pixels": 20, "minimum_height_pixels": 25, "minimum_area_ratio": 0.0005, "minimum_aspect_ratio": 0.20, "maximum_aspect_ratio": 3.20},
        "3WHEELER_B": {"minimum_width_pixels": 30, "minimum_height_pixels": 35, "minimum_area_ratio": 0.0010, "minimum_aspect_ratio": 0.25, "maximum_aspect_ratio": 3.00},
        "3WHEELER_C": {"minimum_width_pixels": 40, "minimum_height_pixels": 45, "minimum_area_ratio": 0.0020, "minimum_aspect_ratio": 0.30, "maximum_aspect_ratio": 2.80},
        "3WHEELER_D": {"minimum_width_pixels": 50, "minimum_height_pixels": 55, "minimum_area_ratio": 0.0030, "minimum_aspect_ratio": 0.35, "maximum_aspect_ratio": 2.60},
    },
    "truck": {
        "TRUCK_A": {"minimum_width_pixels": 30, "minimum_height_pixels": 30, "minimum_area_ratio": 0.0010, "minimum_aspect_ratio": 0.25, "maximum_aspect_ratio": 4.50},
        "TRUCK_B": {"minimum_width_pixels": 40, "minimum_height_pixels": 40, "minimum_area_ratio": 0.0020, "minimum_aspect_ratio": 0.35, "maximum_aspect_ratio": 4.00},
        "TRUCK_C": {"minimum_width_pixels": 50, "minimum_height_pixels": 45, "minimum_area_ratio": 0.0035, "minimum_aspect_ratio": 0.40, "maximum_aspect_ratio": 3.80},
        "TRUCK_D": {"minimum_width_pixels": 65, "minimum_height_pixels": 55, "minimum_area_ratio": 0.0050, "minimum_aspect_ratio": 0.50, "maximum_aspect_ratio": 3.50},
    },
    "bus": {
        "BUS_A": {"minimum_width_pixels": 30, "minimum_height_pixels": 35, "minimum_area_ratio": 0.0015, "minimum_aspect_ratio": 0.25, "maximum_aspect_ratio": 4.00},
        "BUS_B": {"minimum_width_pixels": 45, "minimum_height_pixels": 45, "minimum_area_ratio": 0.0030, "minimum_aspect_ratio": 0.35, "maximum_aspect_ratio": 3.60},
        "BUS_C": {"minimum_width_pixels": 60, "minimum_height_pixels": 55, "minimum_area_ratio": 0.0050, "minimum_aspect_ratio": 0.40, "maximum_aspect_ratio": 3.30},
        "BUS_D": {"minimum_width_pixels": 75, "minimum_height_pixels": 65, "minimum_area_ratio": 0.0075, "minimum_aspect_ratio": 0.50, "maximum_aspect_ratio": 3.00},
    },
}

COMMON_PROFILE_LIMITS = {
    "maximum_area_ratio": 0.95,
    "edge_mode": EDGE_MODE_A,
    "edge_margin_pixels": 8,
}

PERMISSIVE_PROFILE = {
    "minimum_width_pixels": 0,
    "minimum_height_pixels": 0,
    "minimum_area_ratio": 0.0,
    "maximum_area_ratio": 0.999999,
    "minimum_aspect_ratio": 0.01,
    "maximum_aspect_ratio": 100.0,
    "edge_mode": EDGE_MODE_A,
    "edge_margin_pixels": 8,
}

KNOWN_CASE_DEFINITIONS = [
    ("full_visible_3wheeler", "3wheeler", "largest_non_edge"),
    ("partial_exiting_3wheeler", "3wheeler", "largest_edge"),
    ("small_valid_motorcycle", "motorcycle", "smallest_non_edge"),
    ("false_small_car_candidate", "car", "smallest_any"),
    ("normal_medium_car", "car", "median_non_edge"),
    ("valid_large_bus", "bus", "largest_non_edge"),
    ("valid_truck", "truck", "largest_non_edge"),
    ("tiny_partial_bus_or_truck", "bus_or_truck", "smallest_edge"),
]


def run_bbox_tuning(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).expanduser().resolve()
    raw_config = _load_raw_config(config_file)
    validated_config = _validate_config(raw_config, config_file)
    output_root = Path(validated_config["output"]["root_directory"]).resolve()
    output_manager = RunOutputManager(output_root)
    logger = setup_logging(output_manager.run_directory, log_level=validated_config["project"]["log_level"])
    tuning_directory = output_manager.future_output_path("bbox_tuning")
    review_directory = tuning_directory / "review_samples"
    final_validation_directory = tuning_directory / "final_validation"
    tuning_directory.mkdir(parents=True, exist_ok=True)
    review_directory.mkdir(parents=True, exist_ok=True)
    final_validation_directory.mkdir(parents=True, exist_ok=True)

    experiment_config = deepcopy(validated_config)
    experiment_config["detection"]["bbox_quality"] = {
        "enabled": False,
        "default": dict(PERMISSIVE_PROFILE),
        "classes": {},
    }
    minimum_requested_frames = max(int(validated_config["input"]["max_frames_per_camera"]), 600)

    logger.info("BBox tuning started run_id=%s frame_target=%s", output_manager.run_id, minimum_requested_frames)
    detector = VehicleDetectorTracker(experiment_config, logger)
    dataset_rows, video_sources = _capture_raw_dataset(
        config=validated_config,
        detector=detector,
        logger=logger,
        frame_target=minimum_requested_frames,
    )
    raw_dataset_path = tuning_directory / "raw_bbox_dataset.csv"
    _write_raw_dataset_csv(raw_dataset_path, dataset_rows)
    raw_summary = _build_raw_summary(dataset_rows, validated_config, minimum_requested_frames)
    (tuning_directory / "raw_bbox_summary.json").write_text(json.dumps(raw_summary, indent=2), encoding="utf-8")

    review_index = _generate_review_samples(dataset_rows, review_directory, video_sources)
    (tuning_directory / "review_samples.md").write_text(review_index, encoding="utf-8")

    known_cases = _build_known_cases(dataset_rows, tuning_directory / "known_cases", video_sources)
    _write_json(tuning_directory / "known_validation_cases.json", known_cases)

    baseline_metrics = _evaluate_profile_run(dataset_rows, validated_config, {}, "__RAW_BASELINE__")
    class_profile_results = _evaluate_all_class_profiles(dataset_rows, validated_config, known_cases)
    _write_json(tuning_directory / "class_profile_results.json", class_profile_results)
    (tuning_directory / "class_profile_results.md").write_text(_render_class_profile_markdown(class_profile_results), encoding="utf-8")

    selected_profiles = _select_profiles(class_profile_results)
    edge_results = _evaluate_edge_modes(dataset_rows, validated_config, selected_profiles, known_cases)
    selected_edge_mode = _select_edge_mode(edge_results)
    selected_profiles_yaml = {
        "detection": {
            "bbox_quality": {
                "enabled": True,
                "default": dict(PERMISSIVE_PROFILE),
                "classes": {
                    class_name: {**profile_payload, "maximum_area_ratio": 0.95, "edge_mode": selected_edge_mode, "edge_margin_pixels": 8}
                    for class_name, profile_payload in selected_profiles.items()
                },
            }
        }
    }
    (tuning_directory / "selected_profiles.yaml").write_text(yaml.safe_dump(selected_profiles_yaml, sort_keys=False), encoding="utf-8")

    final_combined = _evaluate_profile_run(
        dataset_rows,
        validated_config,
        {
            class_name: {**profile_payload, "maximum_area_ratio": 0.95, "edge_mode": selected_edge_mode, "edge_margin_pixels": 8}
            for class_name, profile_payload in selected_profiles.items()
        },
        "__FINAL_COMBINED__",
    )
    evidence_index = _generate_final_validation_evidence(
        dataset_rows,
        selected_profiles_yaml["detection"]["bbox_quality"]["classes"],
        video_sources,
        final_validation_directory,
    )
    final_comparison = {
        "baseline": baseline_metrics,
        "selected_profiles": selected_profiles,
        "edge_mode_results": edge_results,
        "selected_edge_mode": selected_edge_mode,
        "final_combined": final_combined,
        "known_cases": known_cases,
        "evidence_index": evidence_index,
    }
    _write_json(tuning_directory / "final_comparison.json", final_comparison)
    (tuning_directory / "final_comparison.md").write_text(
        _render_final_comparison_markdown(raw_summary, class_profile_results, selected_profiles, selected_edge_mode, baseline_metrics, final_combined, known_cases, evidence_index),
        encoding="utf-8",
    )

    result = {
        "run_id": output_manager.run_id,
        "run_directory": str(output_manager.run_directory),
        "bbox_tuning_directory": str(tuning_directory),
        "selected_profiles": selected_profiles,
        "selected_edge_mode": selected_edge_mode,
        "raw_summary": raw_summary,
        "class_profile_results": class_profile_results,
        "final_comparison": final_comparison,
    }
    _write_json(tuning_directory / "tuning_result.json", result)
    logger.info("BBox tuning completed run_id=%s output=%s", output_manager.run_id, tuning_directory)
    return result


def _capture_raw_dataset(
    *,
    config: dict[str, Any],
    detector: VehicleDetectorTracker,
    logger,
    frame_target: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    dataset_rows: list[dict[str, Any]] = []
    video_sources: dict[str, str] = {}
    for camera in config["input"]["cameras"]:
        if not camera["enabled"]:
            continue
        reader = VideoCameraReader(
            camera_id=camera["camera_id"],
            source_type=camera["source_type"],
            source=camera["source"],
            target_read_fps=config["ingestion"]["target_read_fps"],
        )
        video_sources[camera["camera_id"]] = str(camera["source"])
        logger.info("Capturing raw bbox dataset camera=%s source=%s", camera["camera_id"], camera["source"])
        try:
            while True:
                packet = reader.read_next_frame(worker_id=0)
                if packet is None or packet.frame_number >= frame_target:
                    break
                detections = detector.infer_yolo_detections(packet)
                accepted, diagnostics = detector.filter_detections(packet, detections)
                del accepted
                for diagnostic in diagnostics:
                    row = asdict(diagnostic)
                    row["source_path"] = str(camera["source"])
                    row["source_fps"] = packet.source_fps
                    dataset_rows.append(row)
                if packet.frame_number > 0 and packet.frame_number % 100 == 0:
                    logger.info("Raw bbox capture camera=%s frame=%s detections=%s", packet.camera_id, packet.frame_number, len(diagnostics))
        finally:
            reader.close()
    return dataset_rows, video_sources


def _write_raw_dataset_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "camera_id",
        "frame_number",
        "timestamp_seconds",
        "class_name",
        "normalized_class_name",
        "confidence",
        "bbox_xyxy",
        "bbox_width",
        "bbox_height",
        "bbox_area",
        "frame_width",
        "frame_height",
        "width_ratio",
        "height_ratio",
        "area_ratio",
        "aspect_ratio",
        "touches_left_edge",
        "touches_right_edge",
        "touches_top_edge",
        "touches_bottom_edge",
        "touches_edge",
        "accepted_by_bbox_quality",
        "rejection_reason",
        "source_path",
        "source_fps",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["bbox_xyxy"] = ",".join(f"{float(value):.4f}" for value in row["bbox_xyxy"])
            writer.writerow(payload)


def _build_raw_summary(rows: list[dict[str, Any]], config: dict[str, Any], frame_target: int) -> dict[str, Any]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_camera: dict[str, int] = defaultdict(int)
    for row in rows:
        by_class[row["normalized_class_name"]].append(row)
        by_camera[row["camera_id"]] += 1
    summary_classes: dict[str, Any] = {}
    for class_name, class_rows in by_class.items():
        area_values = [row["area_ratio"] for row in class_rows]
        width_values = [row["bbox_width"] for row in class_rows]
        height_values = [row["bbox_height"] for row in class_rows]
        aspect_values = [row["aspect_ratio"] for row in class_rows]
        confidence_values = [row["confidence"] for row in class_rows]
        summary_classes[class_name] = {
            "raw_detections": len(class_rows),
            "edge_touching": len([row for row in class_rows if row["touches_edge"]]),
            "area_ratio_percentiles": _percentiles(area_values),
            "bbox_width_percentiles": _percentiles(width_values),
            "bbox_height_percentiles": _percentiles(height_values),
            "aspect_ratio_percentiles": _percentiles(aspect_values),
            "confidence_percentiles": _percentiles(confidence_values),
        }
    return {
        "configured_cameras": [camera["camera_id"] for camera in config["input"]["cameras"] if camera["enabled"]],
        "frame_target_per_camera": frame_target,
        "total_raw_detections": len(rows),
        "raw_detections_by_camera": dict(by_camera),
        "raw_detections_by_class": summary_classes,
    }


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
    }


def _generate_review_samples(rows: list[dict[str, Any]], review_directory: Path, video_sources: dict[str, str]) -> str:
    lines = ["# Review Samples", ""]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["normalized_class_name"]].append(row)
    for class_name, class_rows in sorted(grouped.items()):
        class_directory = review_directory / class_name
        class_directory.mkdir(parents=True, exist_ok=True)
        selections = _select_review_rows(class_rows)
        lines.append(f"## {class_name}")
        lines.append("")
        for sample_name, row in selections.items():
            image_path = class_directory / f"{sample_name}.jpg"
            _save_annotated_detection_image(row, video_sources[row["camera_id"]], image_path, f"{sample_name}")
            relative = image_path.relative_to(review_directory.parent)
            lines.append(f"- `{sample_name}`: `{relative}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def _select_review_rows(class_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows_by_area = sorted(class_rows, key=lambda item: item["area_ratio"])
    rows_by_aspect = sorted(class_rows, key=lambda item: item["aspect_ratio"])
    rows_by_conf = sorted(class_rows, key=lambda item: item["confidence"])
    rows_edge = [row for row in rows_by_area if row["touches_edge"]]
    return {
        "smallest_area": rows_by_area[0],
        "p10_area": _nearest_percentile_row(rows_by_area, "area_ratio", 10),
        "p25_area": _nearest_percentile_row(rows_by_area, "area_ratio", 25),
        "median_area": _nearest_percentile_row(rows_by_area, "area_ratio", 50),
        "large_area": _nearest_percentile_row(rows_by_area, "area_ratio", 90),
        "edge_touching": rows_edge[0] if rows_edge else rows_by_area[0],
        "lowest_aspect": rows_by_aspect[0],
        "highest_aspect": rows_by_aspect[-1],
        "low_confidence": rows_by_conf[0],
        "high_confidence": rows_by_conf[-1],
    }


def _nearest_percentile_row(rows: list[dict[str, Any]], key: str, percentile: float) -> dict[str, Any]:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    target = float(np.percentile(values, percentile))
    return min(rows, key=lambda row: abs(row[key] - target))


def _build_known_cases(rows: list[dict[str, Any]], output_directory: Path, video_sources: dict[str, str]) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["normalized_class_name"]].append(row)
    known_cases: dict[str, Any] = {}
    for case_name, class_name, strategy in KNOWN_CASE_DEFINITIONS:
        if class_name == "bus_or_truck":
            pool = grouped.get("bus", []) + grouped.get("truck", [])
        else:
            pool = grouped.get(class_name, [])
        if not pool:
            continue
        row = _pick_known_case(pool, strategy)
        image_path = output_directory / f"{case_name}.jpg"
        _save_annotated_detection_image(row, video_sources[row["camera_id"]], image_path, case_name)
        known_cases[case_name] = {
            "expected_keep": case_name not in {"false_small_car_candidate", "tiny_partial_bus_or_truck"},
            "reason": strategy,
            "image_path": str(image_path),
            **row,
        }
    return known_cases


def _pick_known_case(pool: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    non_edge = [row for row in pool if not row["touches_edge"]]
    edge = [row for row in pool if row["touches_edge"]]
    if strategy == "largest_non_edge":
        target_pool = non_edge or pool
        return max(target_pool, key=lambda row: row["bbox_area"])
    if strategy == "largest_edge":
        target_pool = edge or pool
        return max(target_pool, key=lambda row: row["bbox_area"])
    if strategy == "smallest_non_edge":
        target_pool = non_edge or pool
        return min(target_pool, key=lambda row: row["bbox_area"])
    if strategy == "smallest_any":
        return min(pool, key=lambda row: row["bbox_area"])
    if strategy == "median_non_edge":
        target_pool = sorted(non_edge or pool, key=lambda row: row["bbox_area"])
        return target_pool[len(target_pool) // 2]
    if strategy == "smallest_edge":
        target_pool = edge or pool
        return min(target_pool, key=lambda row: row["bbox_area"])
    return pool[0]


def _evaluate_all_class_profiles(rows: list[dict[str, Any]], config: dict[str, Any], known_cases: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for class_name, profiles in PROFILE_LIBRARY.items():
        class_results: list[dict[str, Any]] = []
        for profile_name, profile_payload in profiles.items():
            targeted_profile = {**profile_payload, **COMMON_PROFILE_LIMITS}
            metrics = _evaluate_profile_run(rows, config, {class_name: targeted_profile}, profile_name)
            class_rows = [row for row in rows if row["normalized_class_name"] == class_name]
            accepted_rows = [row for row in metrics["accepted_rows"] if row["normalized_class_name"] == class_name]
            known_valid_lost = 0
            false_removed = 0
            for case in known_cases.values():
                if case["normalized_class_name"] != class_name:
                    continue
                match = _find_matching_row(metrics["accepted_rows"], case)
                if case["expected_keep"]:
                    known_valid_lost += 0 if match is not None else 1
                else:
                    false_removed += 1 if match is None else 0
            class_results.append(
                {
                    "class": class_name,
                    "profile": profile_name,
                    "profile_payload": targeted_profile,
                    "raw_detections": len(class_rows),
                    "accepted_detections": len(accepted_rows),
                    "rejected_detections": len(class_rows) - len(accepted_rows),
                    "acceptance_rate": float(len(accepted_rows) / len(class_rows)) if class_rows else 0.0,
                    "tracked_observations": metrics["tracked_observations_by_class"].get(class_name, 0),
                    "unique_native_tracks": metrics["unique_tracks_by_class"].get(class_name, 0),
                    "short_tracks": metrics["short_tracks_by_class"].get(class_name, 0),
                    "average_track_length": metrics["average_track_length_by_class"].get(class_name, 0.0),
                    "minimum_accepted_bbox_width": min((row["bbox_width"] for row in accepted_rows), default=0.0),
                    "minimum_accepted_bbox_height": min((row["bbox_height"] for row in accepted_rows), default=0.0),
                    "minimum_accepted_area_ratio": min((row["area_ratio"] for row in accepted_rows), default=0.0),
                    "false_small_detections_removed": false_removed,
                    "known_valid_detections_lost": known_valid_lost,
                    "edge_touching_detections_retained": len([row for row in accepted_rows if row["touches_edge"]]),
                }
            )
        results[class_name] = class_results
    return results


def _evaluate_edge_modes(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    selected_profiles: dict[str, dict[str, Any]],
    known_cases: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for edge_mode in (EDGE_MODE_A, EDGE_MODE_B, EDGE_MODE_C):
        class_profiles = {
            class_name: {**profile_payload, "maximum_area_ratio": 0.95, "edge_mode": edge_mode, "edge_margin_pixels": 8}
            for class_name, profile_payload in selected_profiles.items()
        }
        metrics = _evaluate_profile_run(rows, config, class_profiles, f"EDGE_{edge_mode}")
        kept_valid = 0
        removed_false = 0
        for case in known_cases.values():
            match = _find_matching_row(metrics["accepted_rows"], case)
            if case["expected_keep"]:
                kept_valid += 1 if match is not None else 0
            else:
                removed_false += 1 if match is None else 0
        results.append(
            {
                "edge_mode": edge_mode,
                "kept_valid_known_cases": kept_valid,
                "removed_false_known_cases": removed_false,
                "edge_touching_retained": len([row for row in metrics["accepted_rows"] if row["touches_edge"]]),
                "tracked_observations_total": sum(metrics["tracked_observations_by_class"].values()),
            }
        )
    return results


def _select_profiles(class_profile_results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for class_name, results in class_profile_results.items():
        ordered = sorted(results, key=lambda item: item["profile"])
        chosen = None
        for result in ordered:
            if result["false_small_detections_removed"] >= 1 and result["known_valid_detections_lost"] == 0:
                chosen = result
                break
        if chosen is None:
            chosen = min(
                ordered,
                key=lambda item: (
                    item["known_valid_detections_lost"],
                    -item["false_small_detections_removed"],
                    -(item["acceptance_rate"]),
                ),
            )
        selected[class_name] = dict(chosen["profile_payload"])
    return selected


def _select_edge_mode(edge_results: list[dict[str, Any]]) -> str:
    ordered = sorted(
        edge_results,
        key=lambda item: (
            -item["removed_false_known_cases"],
            -item["kept_valid_known_cases"],
            -item["edge_touching_retained"],
            -item["tracked_observations_total"],
        ),
    )
    return ordered[0]["edge_mode"]


def _evaluate_profile_run(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    class_profiles: dict[str, dict[str, Any]],
    run_label: str,
) -> dict[str, Any]:
    evaluation_config = deepcopy(config)
    evaluation_config["detection"]["bbox_quality"] = {
        "enabled": True,
        "default": dict(PERMISSIVE_PROFILE),
        "classes": {class_name: dict(profile) for class_name, profile in class_profiles.items()},
    }
    detector = VehicleDetectorTracker(evaluation_config, logging_proxy())
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    tracked_observations_by_class: dict[str, int] = defaultdict(int)
    unique_tracks_by_class: dict[str, set[tuple[str, int]]] = defaultdict(set)
    track_counts_by_key: dict[tuple[str, int], int] = defaultdict(int)
    track_class_by_key: dict[tuple[str, int], str] = {}
    grouped = _group_rows_by_camera_and_frame(rows)
    for camera_id, frames in grouped.items():
        detector.reset_camera(camera_id)
        source_fps = float(next(iter(frames.values()))[0]["source_fps"]) if frames else 30.0
        for frame_number in sorted(frames):
            frame_rows = frames[frame_number]
            packet = _synthetic_packet(frame_rows[0], source_fps)
            detections = [_row_to_detection(row) for row in frame_rows]
            accepted, diagnostics = detector.filter_detections(packet, detections)
            accepted_rows.extend(_merge_rows_with_diagnostics(frame_rows, diagnostics, accepted_only=True))
            rejected_rows.extend(_merge_rows_with_diagnostics(frame_rows, diagnostics, accepted_only=False))
            tracked = detector.track_detections(packet, accepted)
            for item in tracked:
                normalized = item.raw_class_name
                tracked_observations_by_class[normalized] += 1
                key = (item.camera_id, item.tracker_id)
                unique_tracks_by_class[normalized].add(key)
                track_counts_by_key[key] += 1
                track_class_by_key[key] = normalized
    track_lengths_by_class: dict[str, list[int]] = defaultdict(list)
    short_tracks_by_class: dict[str, int] = defaultdict(int)
    for key, count in track_counts_by_key.items():
        class_name = track_class_by_key[key]
        track_lengths_by_class[class_name].append(count)
        if count < 3:
            short_tracks_by_class[class_name] += 1
    return {
        "run_label": run_label,
        "accepted_rows": accepted_rows,
        "rejected_rows": rejected_rows,
        "tracked_observations_by_class": {key: int(value) for key, value in tracked_observations_by_class.items()},
        "unique_tracks_by_class": {key: len(value) for key, value in unique_tracks_by_class.items()},
        "short_tracks_by_class": {key: int(value) for key, value in short_tracks_by_class.items()},
        "average_track_length_by_class": {
            key: float(mean(values)) if values else 0.0 for key, values in track_lengths_by_class.items()
        },
    }


def _group_rows_by_camera_and_frame(rows: list[dict[str, Any]]) -> dict[str, dict[int, list[dict[str, Any]]]]:
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["camera_id"]][int(row["frame_number"])].append(row)
    return grouped


def _synthetic_packet(row: dict[str, Any], source_fps: float) -> FramePacket:
    frame = np.zeros((int(row["frame_height"]), int(row["frame_width"]), 3), dtype=np.uint8)
    return FramePacket(
        camera_id=row["camera_id"],
        frame_number=int(row["frame_number"]),
        timestamp_seconds=float(row["timestamp_seconds"]),
        source_fps=float(source_fps),
        frame=frame,
        worker_id=0,
        captured_at="",
        source_type="video",
    )


def _row_to_detection(row: dict[str, Any]) -> Detection:
    x1, y1, x2, y2 = row["bbox_xyxy"]
    return Detection(
        bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
        confidence=float(row["confidence"]),
        class_id=0,
        class_name=row["normalized_class_name"],
    )


def _merge_rows_with_diagnostics(
    frame_rows: list[dict[str, Any]],
    diagnostics: list[Any],
    *,
    accepted_only: bool,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for row, diagnostic in zip(frame_rows, diagnostics):
        if bool(diagnostic.accepted_by_bbox_quality) != accepted_only:
            continue
        payload = dict(row)
        payload.update(asdict(diagnostic))
        merged.append(payload)
    return merged


def _find_matching_row(rows: list[dict[str, Any]], candidate: dict[str, Any]) -> dict[str, Any] | None:
    for row in rows:
        if row["camera_id"] == candidate["camera_id"] and int(row["frame_number"]) == int(candidate["frame_number"]):
            if tuple(float(value) for value in row["bbox_xyxy"]) == tuple(float(value) for value in candidate["bbox_xyxy"]):
                return row
    return None


def _generate_final_validation_evidence(
    rows: list[dict[str, Any]],
    selected_profiles: dict[str, dict[str, Any]],
    video_sources: dict[str, str],
    output_directory: Path,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["normalized_class_name"]].append(row)
    for class_name, profile_payload in selected_profiles.items():
        class_dir = output_directory / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        class_rows = grouped.get(class_name, [])
        if not class_rows:
            continue
        accepted_rows = [row for row in class_rows if _passes_profile(row, profile_payload)]
        rejected_rows = [row for row in class_rows if not _passes_profile(row, profile_payload)]
        edge_rows = [row for row in class_rows if row["touches_edge"]]
        saved_paths: dict[str, list[str]] = {"accepted_smallest": [], "rejected_largest": [], "edge_touching": []}
        for index, row in enumerate(sorted(accepted_rows, key=lambda item: item["bbox_area"])[:5]):
            path = class_dir / f"accepted_smallest_{index+1}.jpg"
            _save_annotated_detection_image(row, video_sources[row["camera_id"]], path, "ACCEPTED")
            saved_paths["accepted_smallest"].append(str(path))
        for index, row in enumerate(sorted(rejected_rows, key=lambda item: item["bbox_area"], reverse=True)[:5]):
            path = class_dir / f"rejected_largest_{index+1}.jpg"
            _save_annotated_detection_image(row, video_sources[row["camera_id"]], path, "REJECTED")
            saved_paths["rejected_largest"].append(str(path))
        for index, row in enumerate(sorted(edge_rows, key=lambda item: item["bbox_area"], reverse=True)[:5]):
            path = class_dir / f"edge_touching_{index+1}.jpg"
            _save_annotated_detection_image(row, video_sources[row["camera_id"]], path, "EDGE")
            saved_paths["edge_touching"].append(str(path))
        evidence[class_name] = saved_paths
    return evidence


def _passes_profile(row: dict[str, Any], profile_payload: dict[str, Any]) -> bool:
    profile = BBoxQualityProfile(
        minimum_width_pixels=float(profile_payload["minimum_width_pixels"]),
        minimum_height_pixels=float(profile_payload["minimum_height_pixels"]),
        minimum_area_ratio=float(profile_payload["minimum_area_ratio"]),
        maximum_area_ratio=float(profile_payload.get("maximum_area_ratio", 0.95)),
        minimum_aspect_ratio=float(profile_payload["minimum_aspect_ratio"]),
        maximum_aspect_ratio=float(profile_payload["maximum_aspect_ratio"]),
        edge_margin_pixels=float(profile_payload.get("edge_margin_pixels", 8)),
        edge_mode=str(profile_payload.get("edge_mode", EDGE_MODE_A)),
    )
    if row["bbox_width"] < profile.minimum_width_pixels:
        return False
    if row["bbox_height"] < profile.minimum_height_pixels:
        return False
    if row["area_ratio"] < profile.minimum_area_ratio or row["area_ratio"] > profile.maximum_area_ratio:
        return False
    if row["aspect_ratio"] < profile.minimum_aspect_ratio or row["aspect_ratio"] > profile.maximum_aspect_ratio:
        return False
    if profile.edge_mode == EDGE_MODE_B and row["touches_edge"] and row["area_ratio"] < profile.minimum_area_ratio:
        return False
    if profile.edge_mode == EDGE_MODE_C and row["touches_edge"] and (
        row["bbox_width"] < profile.minimum_width_pixels or row["bbox_height"] < profile.minimum_height_pixels
    ):
        return False
    return True


def _save_annotated_detection_image(row: dict[str, Any], source_path: str, output_path: Path, prefix: str) -> None:
    frame = _read_video_frame(source_path, int(row["frame_number"]))
    if frame is None:
        return
    x1, y1, x2, y2 = [int(round(value)) for value in row["bbox_xyxy"]]
    label_lines = [
        f"{prefix} {row['normalized_class_name']} conf={row['confidence']:.2f}",
        f"w={row['bbox_width']:.1f} h={row['bbox_height']:.1f} area={row['area_ratio']:.4f}",
        f"aspect={row['aspect_ratio']:.2f} edge={_edge_flags(row)}",
        f"{row['camera_id']} frame={row['frame_number']}",
    ]
    annotated = frame.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 255), 2)
    y = max(18, y1 - 42)
    for line in label_lines:
        cv2.putText(annotated, line, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
        y += 16
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def _read_video_frame(source_path: str, frame_number: int) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(source_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()


def _edge_flags(row: dict[str, Any]) -> str:
    flags = []
    if row.get("touches_left_edge"):
        flags.append("L")
    if row.get("touches_right_edge"):
        flags.append("R")
    if row.get("touches_top_edge"):
        flags.append("T")
    if row.get("touches_bottom_edge"):
        flags.append("B")
    return "".join(flags) or "NONE"


def _render_class_profile_markdown(results: dict[str, Any]) -> str:
    lines = ["# Class Profile Results", ""]
    for class_name, class_results in sorted(results.items()):
        lines.append(f"## {class_name}")
        lines.append("")
        lines.append("| Class | Profile | Raw | Accepted | Rejected | Valid lost | False removed | Tracked observations | Unique tracks |")
        lines.append("| ----- | ------- | --: | -------: | -------: | ---------: | ------------: | -------------------: | ------------: |")
        for item in class_results:
            lines.append(
                f"| {item['class']} | {item['profile']} | {item['raw_detections']} | {item['accepted_detections']} | {item['rejected_detections']} | {item['known_valid_detections_lost']} | {item['false_small_detections_removed']} | {item['tracked_observations']} | {item['unique_native_tracks']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _render_final_comparison_markdown(
    raw_summary: dict[str, Any],
    class_profile_results: dict[str, Any],
    selected_profiles: dict[str, dict[str, Any]],
    selected_edge_mode: str,
    baseline_metrics: dict[str, Any],
    final_combined: dict[str, Any],
    known_cases: dict[str, Any],
    evidence_index: dict[str, Any],
) -> str:
    lines = ["# Final BBox Tuning Comparison", ""]
    lines.append("## Raw Detection Review Counts")
    lines.append("")
    for class_name, payload in sorted(raw_summary["raw_detections_by_class"].items()):
        lines.append(f"- `{class_name}`: {payload['raw_detections']} raw detections reviewed")
    lines.append("")
    lines.append("## Recommended Initial Values")
    lines.append("")
    for class_name, payload in sorted(selected_profiles.items()):
        lines.append(f"- `{class_name}`: width>={payload['minimum_width_pixels']}, height>={payload['minimum_height_pixels']}, area>={payload['minimum_area_ratio']}, aspect={payload['minimum_aspect_ratio']}..{payload['maximum_aspect_ratio']}")
    lines.append(f"- `edge_mode`: {selected_edge_mode}")
    lines.append("")
    lines.append("## Known Cases")
    lines.append("")
    for case_name, payload in sorted(known_cases.items()):
        lines.append(
            f"- `{case_name}`: class={payload['normalized_class_name']} keep={payload['expected_keep']} frame={payload['frame_number']} area_ratio={payload['area_ratio']:.4f} aspect={payload['aspect_ratio']:.2f} edge={_edge_flags(payload)} image=`{payload['image_path']}`"
        )
    lines.append("")
    lines.append("## Before And After")
    lines.append("")
    lines.append(f"- Baseline tracked observations: {sum(baseline_metrics['tracked_observations_by_class'].values())}")
    lines.append(f"- Final tracked observations: {sum(final_combined['tracked_observations_by_class'].values())}")
    lines.append(f"- Baseline unique tracks: {sum(baseline_metrics['unique_tracks_by_class'].values())}")
    lines.append(f"- Final unique tracks: {sum(final_combined['unique_tracks_by_class'].values())}")
    lines.append("")
    lines.append("## Evidence")
    lines.append("")
    for class_name, payload in sorted(evidence_index.items()):
        lines.append(f"- `{class_name}` accepted smallest: {', '.join(payload['accepted_smallest'])}")
        lines.append(f"- `{class_name}` rejected largest: {', '.join(payload['rejected_largest'])}")
        lines.append(f"- `{class_name}` edge touching: {', '.join(payload['edge_touching'])}")
    lines.append("")
    lines.append("## Remaining Limitations")
    lines.append("")
    lines.append("- These are recommended initial values for the tested cameras.")
    lines.append("- Known-case selection is seeded from detection statistics and still expects human review of the saved evidence.")
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class _NullLogger:
    def debug(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

    def info(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

    def warning(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

    def error(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None


def logging_proxy():
    return _NullLogger()
