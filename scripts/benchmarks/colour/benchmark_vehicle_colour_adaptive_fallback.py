from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_vehicle_colour_single_crop import (
    DEFAULT_SEARCH_ROOTS,
    extract_ranked_predictions,
    find_latest_suitable_run,
    normalize_colour,
)


DEFAULT_OUTPUT_DIR = Path("diagnostics/colour_adaptive_fallback_benchmark")
VALID_COLOURS = {
    "BLACK",
    "WHITE",
    "RED",
    "PINK",
    "BLUE",
    "GREEN",
    "YELLOW",
    "BROWN",
    "GREY",
    "SILVER",
    "ORANGE",
    "BEIGE",
    "PURPLE",
    "OTHER",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark adaptive colour fallback against existing 1-crop and 3-crop outputs.")
    parser.add_argument("--run-dir", default="", help="Existing run directory. If omitted, auto-detect the latest suitable run.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for benchmark outputs.")
    parser.add_argument("--min-tracks", type=int, default=50, help="Minimum track count for auto-detected runs.")
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_valid_colour(label: str) -> bool:
    return normalize_colour(label) in VALID_COLOURS


def _adaptive_result(predictions: list[Any]) -> tuple[str, int, int | None, bool]:
    if not predictions:
        return "UNKNOWN", 0, None, False
    calls_used = 0
    for prediction in predictions:
        calls_used += 1
        label = normalize_colour(prediction.parsed_colour)
        if _is_valid_colour(label):
            return label, calls_used, prediction.rank, True
    return "UNKNOWN", calls_used, predictions[-1].rank if predictions else None, False


def analyze_run(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    track_summary_rows = _read_json(run_dir / "vehicle_colour_track_summary.json")
    enrichment_rows = _read_json(run_dir / "vehicle_enrichment.json")
    summary = _read_json(run_dir / "summary.json")
    enrichment_metrics = _read_json(run_dir / "vehicle_enrichment_metrics.json")
    by_track = {
        str(row.get("local_track_id") or ""): row
        for row in enrichment_rows
        if str(row.get("local_track_id") or "").strip()
    }

    benchmark_rows: list[dict[str, Any]] = []
    disagreement_patterns: Counter[str] = Counter()
    adaptive_stop_counts = Counter()
    total_adaptive_calls = 0
    total_one_crop_calls = 0
    total_three_crop_calls = 0
    total_adaptive_time_ms = 0.0
    total_one_crop_time_ms = 0.0
    total_three_crop_time_ms = 0.0

    for track_row in track_summary_rows:
        track_id = str(track_row.get("local_track_id") or "")
        enrichment_row = by_track.get(track_id)
        if enrichment_row is None:
            continue
        ranked = extract_ranked_predictions(track_row, enrichment_row)
        if not ranked:
            continue
        crop1 = ranked[0] if len(ranked) >= 1 else None
        crop2 = ranked[1] if len(ranked) >= 2 else None
        crop3 = ranked[2] if len(ranked) >= 3 else None
        final_1crop = normalize_colour(crop1.parsed_colour if crop1 is not None else "UNKNOWN")
        final_3crop = normalize_colour(track_row.get("final_vehicle_colour"))
        final_adaptive, adaptive_calls_used, adaptive_stop_rank, adaptive_valid = _adaptive_result(ranked)
        same_as_3crop = final_adaptive == final_3crop
        if not same_as_3crop:
            disagreement_patterns[f"{final_adaptive} vs {final_3crop}"] += 1
        adaptive_stop_counts[str(adaptive_stop_rank or 0)] += 1 if adaptive_stop_rank is not None else 0
        if not adaptive_valid:
            adaptive_stop_counts["unresolved"] += 1
        total_adaptive_calls += adaptive_calls_used
        total_one_crop_calls += 1
        total_three_crop_calls += int(track_row.get("colour_inference_count", len(ranked)) or len(ranked))
        total_adaptive_time_ms += sum(item.inference_time_ms for item in ranked[:adaptive_calls_used])
        total_one_crop_time_ms += crop1.inference_time_ms if crop1 is not None else 0.0
        total_three_crop_time_ms += sum(item.inference_time_ms for item in ranked)

        benchmark_rows.append(
            {
                "camera_id": track_row.get("camera_id"),
                "track_id": track_id,
                "vehicle_class": track_row.get("vehicle_class"),
                "crop_1_path": crop1.crop_path if crop1 else "",
                "crop_1_raw": crop1.raw_response if crop1 else "",
                "crop_1_parsed": crop1.parsed_colour if crop1 else "",
                "crop_2_path": crop2.crop_path if crop2 else "",
                "crop_2_raw": crop2.raw_response if crop2 else "",
                "crop_2_parsed": crop2.parsed_colour if crop2 else "",
                "crop_3_path": crop3.crop_path if crop3 else "",
                "crop_3_raw": crop3.raw_response if crop3 else "",
                "crop_3_parsed": crop3.parsed_colour if crop3 else "",
                "final_colour_1crop": final_1crop,
                "final_colour_adaptive": final_adaptive,
                "final_colour_3crop": final_3crop,
                "adaptive_calls_used": adaptive_calls_used,
                "adaptive_stopped_at_crop": adaptive_stop_rank if adaptive_stop_rank is not None else "",
                "adaptive_valid": adaptive_valid,
                "same_as_3crop": same_as_3crop,
                "ground_truth_colour": "",
                "adaptive_correct": "",
                "three_crop_correct": "",
                "notes": "" if same_as_3crop else f"adaptive disagreement: {final_adaptive} vs {final_3crop}",
            }
        )

    tracks_tested = len(benchmark_rows)
    one_unknown = sum(1 for row in benchmark_rows if normalize_colour(row["final_colour_1crop"]) == "UNKNOWN")
    adaptive_unknown = sum(1 for row in benchmark_rows if normalize_colour(row["final_colour_adaptive"]) == "UNKNOWN")
    three_unknown = sum(1 for row in benchmark_rows if normalize_colour(row["final_colour_3crop"]) == "UNKNOWN")
    adaptive_same = sum(1 for row in benchmark_rows if row["same_as_3crop"])
    adaptive_diff = tracks_tested - adaptive_same
    adaptive_agreement = round((adaptive_same / tracks_tested) * 100.0, 3) if tracks_tested else 0.0
    adaptive_call_reduction = total_three_crop_calls - total_adaptive_calls
    adaptive_percent_reduction = round((adaptive_call_reduction / total_three_crop_calls) * 100.0, 3) if total_three_crop_calls else 0.0

    manual_review_rows = [
        {
            "camera_id": row["camera_id"],
            "track_id": row["track_id"],
            "vehicle_class": row["vehicle_class"],
            "crop_1_path": row["crop_1_path"],
            "crop_2_path": row["crop_2_path"],
            "crop_3_path": row["crop_3_path"],
            "crop_1_parsed": row["crop_1_parsed"],
            "crop_2_parsed": row["crop_2_parsed"],
            "crop_3_parsed": row["crop_3_parsed"],
            "adaptive_final": row["final_colour_adaptive"],
            "three_crop_final": row["final_colour_3crop"],
            "ground_truth_colour": "",
            "notes": row["notes"] or f"adaptive_calls={row['adaptive_calls_used']}",
        }
        for row in benchmark_rows
        if (not row["same_as_3crop"]) or int(row["adaptive_calls_used"]) >= 2
    ]

    summary_payload = {
        "run_id": summary.get("run_id"),
        "run_directory": str(run_dir),
        "tracks_tested": tracks_tested,
        "three_crop_calls": total_three_crop_calls,
        "one_crop_calls": total_one_crop_calls,
        "adaptive_calls": total_adaptive_calls,
        "adaptive_call_reduction": adaptive_call_reduction,
        "adaptive_percent_reduction": adaptive_percent_reduction,
        "resolved_at_crop_1": adaptive_stop_counts.get("1", 0),
        "resolved_at_crop_2": adaptive_stop_counts.get("2", 0),
        "resolved_at_crop_3": adaptive_stop_counts.get("3", 0),
        "unresolved": adaptive_unknown,
        "one_crop_unknown": one_unknown,
        "adaptive_unknown": adaptive_unknown,
        "three_crop_unknown": three_unknown,
        "adaptive_vs_3crop_agreement": adaptive_agreement,
        "adaptive_vs_3crop_same": adaptive_same,
        "adaptive_vs_3crop_different": adaptive_diff,
        "three_crop_elapsed_time_sec": round(total_three_crop_time_ms / 1000.0, 6),
        "one_crop_elapsed_time_sec": round(total_one_crop_time_ms / 1000.0, 6),
        "adaptive_elapsed_time_sec": round(total_adaptive_time_ms / 1000.0, 6),
        "three_crop_tracks_per_sec": round((tracks_tested / (total_three_crop_time_ms / 1000.0)), 6) if total_three_crop_time_ms > 0 else 0.0,
        "one_crop_tracks_per_sec": round((tracks_tested / (total_one_crop_time_ms / 1000.0)), 6) if total_one_crop_time_ms > 0 else 0.0,
        "adaptive_tracks_per_sec": round((tracks_tested / (total_adaptive_time_ms / 1000.0)), 6) if total_adaptive_time_ms > 0 else 0.0,
        "three_crop_calls_per_sec": round((total_three_crop_calls / (total_three_crop_time_ms / 1000.0)), 6) if total_three_crop_time_ms > 0 else 0.0,
        "one_crop_calls_per_sec": round((total_one_crop_calls / (total_one_crop_time_ms / 1000.0)), 6) if total_one_crop_time_ms > 0 else 0.0,
        "adaptive_calls_per_sec": round((total_adaptive_calls / (total_adaptive_time_ms / 1000.0)), 6) if total_adaptive_time_ms > 0 else 0.0,
        "estimated_queue_pressure_reduction_percent": adaptive_percent_reduction,
        "florence_average_inference_time_ms": enrichment_metrics.get("average_colour_inference_time_ms"),
        "known_hard_case_results": [
            {
                "track_id": row["track_id"],
                "crop_1_prediction": row["crop_1_parsed"],
                "crop_2_prediction": row["crop_2_parsed"],
                "crop_3_prediction": row["crop_3_parsed"],
                "adaptive_stopped_at_crop": row["adaptive_stopped_at_crop"],
                "adaptive_final": row["final_colour_adaptive"],
            }
            for row in benchmark_rows
            if str(row["track_id"]).endswith(":TRACK_16")
        ],
        "disagreement_patterns": dict(disagreement_patterns.most_common()),
    }
    return benchmark_rows, summary_payload, manual_review_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_report(summary: dict[str, Any]) -> str:
    hard_case_lines = [
        f"- `{item['track_id']}` crop1=`{item['crop_1_prediction']}` crop2=`{item['crop_2_prediction']}` crop3=`{item['crop_3_prediction']}` stop=`{item['adaptive_stopped_at_crop']}` final=`{item['adaptive_final']}`"
        for item in summary.get("known_hard_case_results", [])
    ] or ["- No TRACK_16 cases found."]
    disagreement_lines = [
        f"- `{pattern}` = {count}"
        for pattern, count in (summary.get("disagreement_patterns", {}) or {}).items()
    ] or ["- No disagreements."]
    return (
        "# Colour Adaptive Fallback Benchmark\n\n"
        f"- Run ID: `{summary['run_id']}`\n"
        f"- Tracks tested: `{summary['tracks_tested']}`\n"
        f"- 3-crop calls: `{summary['three_crop_calls']}`\n"
        f"- 1-crop calls: `{summary['one_crop_calls']}`\n"
        f"- Adaptive calls: `{summary['adaptive_calls']}`\n"
        f"- Adaptive call reduction: `{summary['adaptive_call_reduction']}` (`{summary['adaptive_percent_reduction']}%`)\n"
        f"- Resolved at crop 1/2/3/unresolved: `{summary['resolved_at_crop_1']}` / `{summary['resolved_at_crop_2']}` / `{summary['resolved_at_crop_3']}` / `{summary['unresolved']}`\n"
        f"- 1-crop UNKNOWN: `{summary['one_crop_unknown']}`\n"
        f"- Adaptive UNKNOWN: `{summary['adaptive_unknown']}`\n"
        f"- 3-crop UNKNOWN: `{summary['three_crop_unknown']}`\n"
        f"- Adaptive vs 3-crop agreement: `{summary['adaptive_vs_3crop_agreement']}%`\n"
        f"- 3-crop elapsed time: `{summary['three_crop_elapsed_time_sec']}` seconds\n"
        f"- 1-crop elapsed time: `{summary['one_crop_elapsed_time_sec']}` seconds\n"
        f"- Adaptive elapsed time: `{summary['adaptive_elapsed_time_sec']}` seconds\n\n"
        "## Known Hard Case\n\n"
        + "\n".join(hard_case_lines)
        + "\n\n## Disagreement Patterns\n\n"
        + "\n".join(disagreement_lines)
    )


def run_benchmark(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    _ensure_dir(output_dir)
    rows, summary, manual_review_rows = analyze_run(run_dir)
    csv_path = output_dir / "colour_adaptive_fallback.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    manual_review_path = output_dir / "manual_review.csv"

    _write_csv(
        csv_path,
        rows,
        [
            "camera_id",
            "track_id",
            "vehicle_class",
            "crop_1_path",
            "crop_1_raw",
            "crop_1_parsed",
            "crop_2_path",
            "crop_2_raw",
            "crop_2_parsed",
            "crop_3_path",
            "crop_3_raw",
            "crop_3_parsed",
            "final_colour_1crop",
            "final_colour_adaptive",
            "final_colour_3crop",
            "adaptive_calls_used",
            "adaptive_stopped_at_crop",
            "adaptive_valid",
            "same_as_3crop",
            "ground_truth_colour",
            "adaptive_correct",
            "three_crop_correct",
            "notes",
        ],
    )
    _write_csv(
        manual_review_path,
        manual_review_rows,
        [
            "camera_id",
            "track_id",
            "vehicle_class",
            "crop_1_path",
            "crop_2_path",
            "crop_3_path",
            "crop_1_parsed",
            "crop_2_parsed",
            "crop_3_parsed",
            "adaptive_final",
            "three_crop_final",
            "ground_truth_colour",
            "notes",
        ],
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(_build_report(summary), encoding="utf-8")
    return {
        "summary": summary,
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "manual_review_path": str(manual_review_path),
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else find_latest_suitable_run(search_roots=DEFAULT_SEARCH_ROOTS, min_tracks=args.min_tracks)
    output_dir = Path(args.output_dir).expanduser().resolve()
    result = run_benchmark(run_dir, output_dir)
    summary = result["summary"]
    print(f"Run directory: {run_dir}")
    print(f"Tracks tested: {summary['tracks_tested']}")
    print(f"Adaptive calls: {summary['adaptive_calls']}")
    print(f"Adaptive agreement vs 3-crop: {summary['adaptive_vs_3crop_agreement']}%")
    print(f"Report: {result['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
