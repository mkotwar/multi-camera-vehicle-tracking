from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bytetrack_association_experiment import (  # noqa: E402
    TARGET_IDS,
    TARGET_LOCAL_IDS,
    _build_tracks,
    _comparison_row,
    _iou,
    _read_csv,
    _target_median_bbox,
    _threshold_key,
    _write_csv,
    inspect_threshold_semantics,
    read_frozen_detections,
    replay_threshold_variant,
    validate_baseline_replay,
)


RUN_DIR = Path("outputs/runs/20260813_182311")
INPUT_DIR = RUN_DIR / "bytetrack_association_test"
OUTPUT_DIR = RUN_DIR / "bytetrack_association_permissive_test"
THRESHOLDS = [0.60, 0.65, 0.70, 0.75]


def main() -> None:
    run_path = RUN_DIR
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load((run_path / "run_config.yaml").read_text(encoding="utf-8"))
    original_tracks = json.loads((run_path / "tracks.json").read_text(encoding="utf-8"))
    original_observations = _read_csv(run_path / "observations.csv")
    target_bbox = _target_median_bbox(original_observations)
    frozen_path = INPUT_DIR / "frozen_detections.jsonl"
    frames = read_frozen_detections(frozen_path)

    variants: dict[str, dict[str, Any]] = {}
    comparison: list[dict[str, Any]] = []
    baseline_validation: dict[str, Any] | None = None
    for threshold in THRESHOLDS:
        key = _threshold_key(threshold)
        variant_dir = output_dir / key
        variant_dir.mkdir(parents=True, exist_ok=True)
        result = replay_threshold_variant(
            config=config,
            frames=frames,
            threshold=threshold,
            output_dir=variant_dir,
            target_bbox=target_bbox,
        )
        enrich_variant(variant_dir, result, target_bbox)
        variants[key] = result
        comparison.append(_comparison_row(key, result) | {
            "completed_tracks": result["completed_tracks"],
            "mean_observations_per_track": result["mean_observations_per_track"],
            "median_observations_per_track": result["median_observations_per_track"],
            "confirmed_wrong_continuations": result["confirmed_wrong_continuations"],
            "suspected_wrong_continuations": result["suspected_wrong_continuations"],
        })
        if threshold == 0.60:
            baseline_validation = validate_baseline_replay(original_tracks, original_observations, result)
            (output_dir / "replay_validation.json").write_text(json.dumps(baseline_validation, indent=2), encoding="utf-8")
            if not baseline_validation["valid"]:
                break

    transition_rows = build_transition_rows(variants)
    _write_csv(output_dir / "comparison.csv", comparison)
    _write_csv(output_dir / "transition_comparison.csv", transition_rows)
    (output_dir / "transition_comparison.json").write_text(json.dumps(transition_rows, indent=2), encoding="utf-8")

    summary = {
        "frozen_detection_source": str(frozen_path),
        "threshold_semantics": inspect_threshold_semantics() | {
            "linear_assignment_comparison": "cost_matrix[cost_matrix > thresh] = thresh + 1e-4; indices_to_matches(..., thresh) keeps assignments whose cost <= thresh",
            "fused_distance_formula": "fuse_cost = 1 - ((1 - iou_distance) * detection_score) = 1 - (IoU * detection_score)",
            "confirmed_order": "0.75 > 0.70 > 0.65 > 0.60, so 0.75 is most permissive for first-pass high-confidence association.",
        },
        "baseline_validation": baseline_validation,
        "variants": variants,
        "comparison": comparison,
        "transition_comparison": transition_rows,
        "decision": decide(comparison, transition_rows, baseline_validation),
    }
    (output_dir / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "decision": summary["decision"]}, indent=2))


def enrich_variant(variant_dir: Path, result: dict[str, Any], target_bbox: list[float]) -> None:
    observations = _read_csv(variant_dir / "tracks.csv")
    _ = observations
    timeline = _read_csv(variant_dir / "truck_timeline.csv")
    result["wrong_continuation_rows"] = wrong_continuations(timeline, target_bbox)
    result["confirmed_wrong_continuations"] = sum(1 for row in result["wrong_continuation_rows"] if row["classification"] == "CONFIRMED")
    result["suspected_wrong_continuations"] = sum(1 for row in result["wrong_continuation_rows"] if row["classification"] == "SUSPECTED")
    result["transition_diagnostics"] = transition_diagnostics(variant_dir, result)
    _write_csv(variant_dir / "wrong_continuations.csv", result["wrong_continuation_rows"])


def wrong_continuations(timeline: list[dict[str, Any]], target_bbox: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_id: dict[int, list[dict[str, Any]]] = {}
    for row in timeline:
        by_id.setdefault(int(row["native_tracker_id"]), []).append(row)
    for native_id, items in by_id.items():
        items.sort(key=lambda row: int(row["frame_number"]))
        for left, right in zip(items, items[1:]):
            gap = int(right["frame_number"]) - int(left["frame_number"])
            if gap <= 1:
                continue
            left_box = [float(left[k]) for k in ("x1", "y1", "x2", "y2")]
            right_box = [float(right[k]) for k in ("x1", "y1", "x2", "y2")]
            continuity_iou = _iou(left_box, right_box)
            target_iou = float(right.get("target_iou") or _iou(right_box, target_bbox))
            if continuity_iou < 0.12 or target_iou < 0.08:
                rows.append(
                    {
                        "native_tracker_id": native_id,
                        "from_frame": int(left["frame_number"]),
                        "to_frame": int(right["frame_number"]),
                        "gap_frames": gap,
                        "continuity_iou": continuity_iou,
                        "target_iou_after_gap": target_iou,
                        "class_after_gap": right.get("class_name"),
                        "classification": "CONFIRMED" if target_iou < 0.08 and str(right.get("class_name")) != "truck" else "SUSPECTED",
                    }
                )
    return rows


def transition_diagnostics(variant_dir: Path, result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    diagnostics = _read_csv(variant_dir / "association_diagnostics.csv")
    tracks = result["truck_tracks"]
    out: dict[str, list[dict[str, Any]]] = {}
    for index, (old_id, new_id) in enumerate(zip(TARGET_IDS[:-1], TARGET_IDS[1:])):
        key = f"{old_id}_to_{new_id}"
        if index >= len(tracks):
            out[key] = []
            continue
        old_native = int(tracks[index]["native_tracker_id"])
        next_first = int(tracks[index + 1]["first_frame"]) if index + 1 < len(tracks) else int(tracks[index]["last_frame"])
        near = [
            normalize_diag(row)
            for row in diagnostics
            if int(row["watched_native_id"]) == old_native and next_first - 5 <= int(row["frame_number"]) <= next_first + 5
        ]
        out[key] = near
    (variant_dir / "transition_diagnostics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def normalize_diag(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_number": int(row["frame_number"]),
        "old_native_id": int(row["watched_native_id"]),
        "old_track_state": row["track_state_before"],
        "predicted_bbox": row["predicted_bbox"],
        "candidate_detection_bbox": row["best_high_detection_bbox"],
        "iou": float(row["best_high_detection_iou"]) if row["best_high_detection_iou"] else None,
        "matching_cost_distance": float(row["best_fused_distance"]) if row["best_fused_distance"] else None,
        "matching_threshold": float(row["matching_threshold"]),
        "candidate_confidence": float(row["best_high_detection_score"]) if row["best_high_detection_score"] else None,
        "association_accepted": str(row["first_pass_match_accepted"]).lower() == "true",
        "resulting_native_id": int(row["watched_native_id"]) if str(row["first_pass_match_accepted"]).lower() == "true" else None,
    }


def build_transition_rows(variants: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for old_id, new_id in zip(TARGET_IDS[:-1], TARGET_IDS[1:]):
        transition = f"{old_id} -> {new_id}"
        row = {"transition": transition}
        key = f"{old_id}_to_{new_id}"
        for threshold in THRESHOLDS:
            variant_key = _threshold_key(threshold)
            row[f"{threshold:.2f}"] = variants.get(variant_key, {}).get("transition_recovery", {}).get(key, "NOT_RUN")
        rows.append(row)
    return rows


def decide(
    comparison: list[dict[str, Any]],
    transition_rows: list[dict[str, Any]],
    baseline_validation: dict[str, Any] | None,
) -> str:
    _ = transition_rows
    if not baseline_validation or not baseline_validation.get("valid"):
        return "STOP: baseline replay did not reproduce the known stationary-truck behavior; 0.65/0.70/0.75 were not run."
    if not comparison or comparison[0]["stationary_truck_fragments"] != 6:
        return "STOP: baseline replay did not reproduce the known stationary-truck behavior."
    safe = [
        row
        for row in comparison
        if int(row["confirmed_wrong_continuations"]) == 0 and int(row["suspected_wrong_continuations"]) == 0
    ]
    baseline = comparison[0]
    improving = [row for row in safe if int(row["stationary_truck_fragments"]) < int(baseline["stationary_truck_fragments"])]
    if not improving:
        return "NO SAFE THRESHOLD IMPROVEMENT: MATCHING THRESHOLD TUNING IS NOT SUFFICIENT."
    best = min(improving, key=lambda row: (int(row["stationary_truck_fragments"]), float(str(row["threshold"]).split("_")[-1])))
    return f"{best['threshold']} is promising on this replay; validate on more videos."


if __name__ == "__main__":
    main()
