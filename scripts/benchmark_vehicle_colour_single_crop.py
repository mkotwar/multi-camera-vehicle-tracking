from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("diagnostics/colour_single_crop_benchmark")
DEFAULT_SEARCH_ROOTS = [
    Path("diagnostics/scalability_benchmark/configs/outputs/runs"),
    Path("outputs/runs"),
]


@dataclass
class CropPrediction:
    rank: int
    crop_path: str
    crop_width: int | None
    crop_height: int | None
    quality_score: float | None
    raw_response: str
    parsed_colour: str
    inference_time_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark 1-crop vs current up-to-3-crop vehicle colour results from an existing run.")
    parser.add_argument("--run-dir", default="", help="Existing run directory to benchmark. If omitted, auto-detect the latest suitable run.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for benchmark CSV/JSON/Markdown outputs.")
    parser.add_argument("--min-tracks", type=int, default=50, help="Minimum track count for auto-detected runs.")
    parser.add_argument("--agreement-sample-size", type=int, default=25, help="Optional number of agreement tracks to include in manual review CSV.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for optional agreement sampling.")
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_latest_suitable_run(*, search_roots: list[Path], min_tracks: int) -> Path:
    candidates: list[Path] = []
    for root in search_roots:
        resolved = root.expanduser().resolve()
        if not resolved.exists():
            continue
        for candidate in resolved.iterdir():
            if not candidate.is_dir():
                continue
            required = [
                candidate / "vehicle_enrichment.json",
                candidate / "vehicle_colour_results.json",
                candidate / "vehicle_colour_track_summary.json",
                candidate / "summary.json",
            ]
            if not all(path.exists() for path in required):
                continue
            try:
                track_summary = _read_json(candidate / "vehicle_colour_track_summary.json")
            except Exception:
                continue
            if not isinstance(track_summary, list) or len(track_summary) < min_tracks:
                continue
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(f"No suitable run found in {[str(root) for root in search_roots]}")
    candidates.sort(key=lambda path: path.name, reverse=True)
    return candidates[0]


def normalize_colour(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text else "UNKNOWN"


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_enrichment_index(enrichment_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in enrichment_rows:
        local_track_id = str(row.get("local_track_id") or "").strip()
        if local_track_id:
            indexed[local_track_id] = row
    return indexed


def extract_ranked_predictions(track_summary_row: dict[str, Any], enrichment_row: dict[str, Any]) -> list[CropPrediction]:
    crop_level_rows = list(enrichment_row.get("crop_level_colours", []) or [])
    by_path = {str(item.get("crop_path") or ""): item for item in crop_level_rows if str(item.get("crop_path") or "")}
    evidence_rows = list(enrichment_row.get("evidence_used", []) or [])
    evidence_by_path = {str(item.get("vehicle_crop_path") or ""): item for item in evidence_rows if str(item.get("vehicle_crop_path") or "")}
    selected_paths = list(enrichment_row.get("selected_colour_crop_paths", []) or [])
    ranked: list[CropPrediction] = []
    for rank, crop_path in enumerate(selected_paths[:3], start=1):
        item = by_path.get(str(crop_path), {})
        evidence_item = evidence_by_path.get(str(crop_path), {})
        parsed_colour = normalize_colour(item.get("normalized_colour"))
        if parsed_colour in {"", "NONE"}:
            parsed_colour = "UNKNOWN"
        ranked.append(
            CropPrediction(
                rank=rank,
                crop_path=str(crop_path),
                crop_width=_safe_int(evidence_item.get("original_crop_width") or evidence_item.get("crop_width") or item.get("original_crop_width")),
                crop_height=_safe_int(evidence_item.get("original_crop_height") or evidence_item.get("crop_height") or item.get("original_crop_height")),
                quality_score=_safe_float(evidence_item.get("quality_score")),
                raw_response=str(item.get("raw_response") or ""),
                parsed_colour=parsed_colour if parsed_colour else "UNKNOWN",
                inference_time_ms=float(item.get("inference_time_ms", 0.0) or 0.0),
            )
        )
    return ranked


def analyze_run(run_dir: Path, *, agreement_sample_size: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    track_summary_rows = _read_json(run_dir / "vehicle_colour_track_summary.json")
    colour_result_rows = _read_json(run_dir / "vehicle_colour_results.json")
    enrichment_rows = _read_json(run_dir / "vehicle_enrichment.json")
    run_summary = _read_json(run_dir / "summary.json")
    enrichment_metrics = _read_json(run_dir / "vehicle_enrichment_metrics.json")
    enrichment_index = _build_enrichment_index(enrichment_rows)
    per_track_rows: list[dict[str, Any]] = []
    disagreement_patterns: Counter[str] = Counter()

    total_current_calls = 0
    total_single_calls = 0
    total_current_time_ms = 0.0
    total_single_time_ms = 0.0

    for summary_row in track_summary_rows:
        local_track_id = str(summary_row.get("local_track_id") or "")
        enrichment_row = enrichment_index.get(local_track_id)
        if enrichment_row is None:
            continue
        ranked_predictions = extract_ranked_predictions(summary_row, enrichment_row)
        if not ranked_predictions:
            continue
        crop_lookup = {prediction.rank: prediction for prediction in ranked_predictions}
        crop1 = crop_lookup.get(1)
        crop2 = crop_lookup.get(2)
        crop3 = crop_lookup.get(3)
        final_colour_1crop = normalize_colour(crop1.parsed_colour if crop1 is not None else "UNKNOWN")
        final_colour_3crop = normalize_colour(summary_row.get("final_vehicle_colour"))
        same_result = final_colour_1crop == final_colour_3crop
        one_crop_unknown = final_colour_1crop == "UNKNOWN"
        three_crop_unknown = final_colour_3crop == "UNKNOWN"
        if not same_result:
            disagreement_patterns[f"{final_colour_1crop} vs {final_colour_3crop}"] += 1

        current_calls = int(summary_row.get("colour_inference_count", len(ranked_predictions)) or len(ranked_predictions))
        total_current_calls += current_calls
        total_single_calls += 1
        total_current_time_ms += sum(prediction.inference_time_ms for prediction in ranked_predictions)
        total_single_time_ms += crop1.inference_time_ms if crop1 is not None else 0.0

        per_track_rows.append(
            {
                "camera_id": summary_row.get("camera_id"),
                "track_id": local_track_id,
                "vehicle_class": summary_row.get("vehicle_class"),
                "best_crop_path": crop1.crop_path if crop1 else "",
                "best_crop_width": crop1.crop_width if crop1 else "",
                "best_crop_height": crop1.crop_height if crop1 else "",
                "best_crop_quality_score": crop1.quality_score if crop1 and crop1.quality_score is not None else "",
                "crop_1_raw_response": crop1.raw_response if crop1 else "",
                "crop_1_parsed_colour": crop1.parsed_colour if crop1 else "UNKNOWN",
                "crop_2_raw_response": crop2.raw_response if crop2 else "",
                "crop_2_parsed_colour": crop2.parsed_colour if crop2 else "",
                "crop_3_raw_response": crop3.raw_response if crop3 else "",
                "crop_3_parsed_colour": crop3.parsed_colour if crop3 else "",
                "final_colour_1crop": final_colour_1crop,
                "final_colour_3crop": final_colour_3crop,
                "same_result": same_result,
                "one_crop_unknown": one_crop_unknown,
                "three_crop_unknown": three_crop_unknown,
                "ground_truth_colour": "",
                "one_crop_correct": "",
                "three_crop_correct": "",
                "notes": "" if same_result else f"disagreement: {final_colour_1crop} vs {final_colour_3crop}",
                "current_colour_inference_count": current_calls,
                "single_crop_inference_count": 1,
                "current_colour_inference_time_ms": round(sum(prediction.inference_time_ms for prediction in ranked_predictions), 6),
                "single_crop_inference_time_ms": round(crop1.inference_time_ms if crop1 else 0.0, 6),
            }
        )

    total_tracks = len(per_track_rows)
    same_count = sum(1 for row in per_track_rows if row["same_result"])
    different_count = total_tracks - same_count
    one_unknown = sum(1 for row in per_track_rows if row["one_crop_unknown"])
    three_unknown = sum(1 for row in per_track_rows if row["three_crop_unknown"])
    one_unknown_three_valid = sum(1 for row in per_track_rows if row["one_crop_unknown"] and not row["three_crop_unknown"])
    one_valid_three_unknown = sum(1 for row in per_track_rows if (not row["one_crop_unknown"]) and row["three_crop_unknown"])
    both_unknown = sum(1 for row in per_track_rows if row["one_crop_unknown"] and row["three_crop_unknown"])
    agreement_rate = round((same_count / total_tracks) * 100.0, 3) if total_tracks else 0.0
    call_reduction = total_current_calls - total_single_calls
    percent_reduction = round((call_reduction / total_current_calls) * 100.0, 3) if total_current_calls else 0.0
    current_elapsed_sec = round(total_current_time_ms / 1000.0, 6)
    single_elapsed_sec = round(total_single_time_ms / 1000.0, 6)
    current_tracks_per_sec = round(total_tracks / current_elapsed_sec, 6) if current_elapsed_sec > 0 else 0.0
    single_tracks_per_sec = round(total_tracks / single_elapsed_sec, 6) if single_elapsed_sec > 0 else 0.0

    random.seed(seed)
    manual_review_rows = [row for row in per_track_rows if (not row["same_result"]) or (row["one_crop_unknown"] and not row["three_crop_unknown"])]
    agreement_pool = [row for row in per_track_rows if row["same_result"]]
    if agreement_sample_size > 0 and agreement_pool:
        sampled = random.sample(agreement_pool, k=min(agreement_sample_size, len(agreement_pool)))
        manual_review_rows.extend(sampled)

    manual_review_csv_rows = [
        {
            "camera_id": row["camera_id"],
            "track_id": row["track_id"],
            "best_crop_path": row["best_crop_path"],
            "one_crop_prediction": row["final_colour_1crop"],
            "three_crop_prediction": row["final_colour_3crop"],
            "ground_truth_colour": "",
            "one_crop_correct": "",
            "three_crop_correct": "",
            "notes": row["notes"],
        }
        for row in manual_review_rows
    ]

    summary_payload = {
        "run_id": run_summary.get("run_id"),
        "run_directory": str(run_dir),
        "tracks_tested": total_tracks,
        "current_3crop_florence_calls": total_current_calls,
        "single_crop_florence_calls": total_single_calls,
        "call_reduction": call_reduction,
        "percent_reduction": percent_reduction,
        "one_crop_valid": total_tracks - one_unknown,
        "one_crop_unknown": one_unknown,
        "three_crop_valid": total_tracks - three_unknown,
        "three_crop_unknown": three_unknown,
        "same_final_colour": same_count,
        "different_final_colour": different_count,
        "agreement_rate": agreement_rate,
        "one_crop_unknown_three_crop_valid": one_unknown_three_valid,
        "one_crop_valid_three_crop_unknown": one_valid_three_unknown,
        "both_unknown": both_unknown,
        "three_crop_elapsed_time_sec": current_elapsed_sec,
        "one_crop_elapsed_time_sec": single_elapsed_sec,
        "three_crop_tracks_per_sec": current_tracks_per_sec,
        "one_crop_tracks_per_sec": single_tracks_per_sec,
        "estimated_florence_compute_reduction_percent": percent_reduction,
        "expected_colour_queue_pressure_reduction_percent": percent_reduction,
        "disagreement_patterns": dict(disagreement_patterns.most_common()),
        "florence_average_inference_time_ms": enrichment_metrics.get("average_colour_inference_time_ms"),
    }
    return per_track_rows, summary_payload, manual_review_csv_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    disagreements = [row for row in rows if not row["same_result"]]
    disagreement_lines = []
    for row in disagreements[:50]:
        disagreement_lines.append(
            f"- `{row['track_id']}` {row['vehicle_class']} 1-crop=`{row['final_colour_1crop']}` 3-crop=`{row['final_colour_3crop']}` best=`{row['best_crop_path']}`"
        )
    if not disagreement_lines:
        disagreement_lines.append("- No disagreements found.")

    patterns = summary.get("disagreement_patterns", {})
    pattern_lines = [f"- `{pattern}` = {count}" for pattern, count in patterns.items()] or ["- No disagreement patterns."]

    if summary["agreement_rate"] >= 98.0 and summary["one_crop_unknown_three_crop_valid"] <= max(1, int(summary["tracks_tested"] * 0.01)):
        recommendation = "A. always 1 crop looks promising"
    else:
        recommendation = "B. 1 best crop + fallback only when needed looks safer"

    return (
        "# Colour Single-Crop Benchmark\n\n"
        f"- Run ID: `{summary['run_id']}`\n"
        f"- Tracks tested: `{summary['tracks_tested']}`\n"
        f"- Current 3-crop Florence calls: `{summary['current_3crop_florence_calls']}`\n"
        f"- Single-crop Florence calls: `{summary['single_crop_florence_calls']}`\n"
        f"- Call reduction: `{summary['call_reduction']}` (`{summary['percent_reduction']}%`)\n"
        f"- Agreement rate: `{summary['agreement_rate']}%`\n"
        f"- 1-crop UNKNOWN / 3-crop valid: `{summary['one_crop_unknown_three_crop_valid']}`\n"
        f"- 1-crop valid / 3-crop UNKNOWN: `{summary['one_crop_valid_three_crop_unknown']}`\n"
        f"- 3-crop elapsed time: `{summary['three_crop_elapsed_time_sec']}` seconds\n"
        f"- 1-crop elapsed time: `{summary['one_crop_elapsed_time_sec']}` seconds\n"
        f"- 3-crop tracks/sec: `{summary['three_crop_tracks_per_sec']}`\n"
        f"- 1-crop tracks/sec: `{summary['one_crop_tracks_per_sec']}`\n"
        f"- Recommendation: `{recommendation}`\n\n"
        "## Disagreement Patterns\n\n"
        + "\n".join(pattern_lines)
        + "\n\n## Disagreement Tracks\n\n"
        + "\n".join(disagreement_lines)
    )


def run_benchmark(*, run_dir: Path, output_dir: Path, agreement_sample_size: int, seed: int) -> dict[str, Any]:
    _ensure_dir(output_dir)
    rows, summary, manual_review_rows = analyze_run(run_dir, agreement_sample_size=agreement_sample_size, seed=seed)
    csv_path = output_dir / "colour_1crop_vs_3crop.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    manual_review_path = output_dir / "colour_1crop_manual_review.csv"

    write_csv(
        csv_path,
        rows,
        [
            "camera_id",
            "track_id",
            "vehicle_class",
            "best_crop_path",
            "best_crop_width",
            "best_crop_height",
            "best_crop_quality_score",
            "crop_1_raw_response",
            "crop_1_parsed_colour",
            "crop_2_raw_response",
            "crop_2_parsed_colour",
            "crop_3_raw_response",
            "crop_3_parsed_colour",
            "final_colour_1crop",
            "final_colour_3crop",
            "same_result",
            "one_crop_unknown",
            "three_crop_unknown",
            "ground_truth_colour",
            "one_crop_correct",
            "three_crop_correct",
            "notes",
            "current_colour_inference_count",
            "single_crop_inference_count",
            "current_colour_inference_time_ms",
            "single_crop_inference_time_ms",
        ],
    )
    write_csv(
        manual_review_path,
        manual_review_rows,
        [
            "camera_id",
            "track_id",
            "best_crop_path",
            "one_crop_prediction",
            "three_crop_prediction",
            "ground_truth_colour",
            "one_crop_correct",
            "three_crop_correct",
            "notes",
        ],
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(build_report(summary, rows), encoding="utf-8")
    return {
        "csv_path": str(csv_path),
        "manual_review_path": str(manual_review_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "summary": summary,
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else find_latest_suitable_run(search_roots=DEFAULT_SEARCH_ROOTS, min_tracks=args.min_tracks)
    output_dir = Path(args.output_dir).expanduser().resolve()
    result = run_benchmark(run_dir=run_dir, output_dir=output_dir, agreement_sample_size=args.agreement_sample_size, seed=args.seed)
    summary = result["summary"]
    print(f"Run directory: {run_dir}")
    print(f"Tracks tested: {summary['tracks_tested']}")
    print(f"Agreement rate: {summary['agreement_rate']}%")
    print(f"Call reduction: {summary['call_reduction']} ({summary['percent_reduction']}%)")
    print(f"Report: {result['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
