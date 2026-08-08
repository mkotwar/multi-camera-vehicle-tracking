from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_vehicle_colour_adaptive_fallback import _adaptive_result, analyze_run, run_benchmark
from scripts.benchmark_vehicle_colour_single_crop import CropPrediction


def _prediction(rank: int, label: str, ms: float = 10.0) -> CropPrediction:
    return CropPrediction(
        rank=rank,
        crop_path=f"crop_{rank}.jpg",
        crop_width=100,
        crop_height=80,
        quality_score=0.5,
        raw_response=label.lower(),
        parsed_colour=label,
        inference_time_ms=ms,
    )


def test_adaptive_result_stops_after_first_valid_prediction() -> None:
    result, calls, stop_rank, valid = _adaptive_result([_prediction(1, "BLACK"), _prediction(2, "GREEN")])
    assert result == "BLACK"
    assert calls == 1
    assert stop_rank == 1
    assert valid is True


def test_adaptive_result_falls_back_until_valid_prediction() -> None:
    result, calls, stop_rank, valid = _adaptive_result([_prediction(1, "UNKNOWN"), _prediction(2, "GREEN"), _prediction(3, "BLUE")])
    assert result == "GREEN"
    assert calls == 2
    assert stop_rank == 2
    assert valid is True


def test_adaptive_result_uses_third_crop_when_needed() -> None:
    result, calls, stop_rank, valid = _adaptive_result([_prediction(1, "UNKNOWN"), _prediction(2, "UNKNOWN"), _prediction(3, "PINK")])
    assert result == "PINK"
    assert calls == 3
    assert stop_rank == 3
    assert valid is True


def test_adaptive_result_returns_unknown_when_all_predictions_unknown() -> None:
    result, calls, stop_rank, valid = _adaptive_result([_prediction(1, "UNKNOWN"), _prediction(2, "UNKNOWN"), _prediction(3, "UNKNOWN")])
    assert result == "UNKNOWN"
    assert calls == 3
    assert stop_rank == 3
    assert valid is False


def test_analyze_run_recovers_single_crop_unknown_with_fallback(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260808_130000"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({"run_id": "20260808_130000"}), encoding="utf-8")
    (run_dir / "vehicle_enrichment_metrics.json").write_text(json.dumps({"average_colour_inference_time_ms": 100.0}), encoding="utf-8")
    (run_dir / "vehicle_colour_track_summary.json").write_text(
        json.dumps(
            [
                {
                    "camera_id": "CAM_001",
                    "local_track_id": "CAM_001:TRACK_16",
                    "vehicle_class": "3WHEELER",
                    "colour_inference_count": 3,
                    "final_vehicle_colour": "GREEN",
                }
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "vehicle_enrichment.json").write_text(
        json.dumps(
            [
                {
                    "local_track_id": "CAM_001:TRACK_16",
                    "selected_colour_crop_paths": ["crop1.jpg", "crop2.jpg", "crop3.jpg"],
                    "crop_level_colours": [
                        {"crop_path": "crop1.jpg", "normalized_colour": "UNKNOWN", "raw_response": "", "inference_time_ms": 12.0},
                        {"crop_path": "crop2.jpg", "normalized_colour": "GREEN", "raw_response": "green", "inference_time_ms": 13.0},
                        {"crop_path": "crop3.jpg", "normalized_colour": "GREEN", "raw_response": "green", "inference_time_ms": 14.0},
                    ],
                    "evidence_used": [
                        {"vehicle_crop_path": "crop1.jpg", "original_crop_width": 100, "original_crop_height": 80, "quality_score": 0.7},
                        {"vehicle_crop_path": "crop2.jpg", "original_crop_width": 100, "original_crop_height": 80, "quality_score": 0.6},
                        {"vehicle_crop_path": "crop3.jpg", "original_crop_width": 100, "original_crop_height": 80, "quality_score": 0.5},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    rows, summary, manual_review = analyze_run(run_dir)
    assert len(rows) == 1
    assert rows[0]["final_colour_1crop"] == "UNKNOWN"
    assert rows[0]["final_colour_adaptive"] == "GREEN"
    assert rows[0]["adaptive_calls_used"] == 2
    assert summary["adaptive_calls"] == 2
    assert summary["resolved_at_crop_2"] == 1
    assert summary["adaptive_unknown"] == 0
    assert summary["adaptive_vs_3crop_agreement"] == 100.0
    assert len(manual_review) == 1


def test_run_benchmark_writes_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260808_130001"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({"run_id": "20260808_130001"}), encoding="utf-8")
    (run_dir / "vehicle_enrichment_metrics.json").write_text(json.dumps({"average_colour_inference_time_ms": 100.0}), encoding="utf-8")
    (run_dir / "vehicle_colour_track_summary.json").write_text(
        json.dumps(
            [
                {
                    "camera_id": "CAM_001",
                    "local_track_id": "CAM_001:TRACK_1",
                    "vehicle_class": "CAR",
                    "colour_inference_count": 1,
                    "final_vehicle_colour": "WHITE",
                }
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "vehicle_enrichment.json").write_text(
        json.dumps(
            [
                {
                    "local_track_id": "CAM_001:TRACK_1",
                    "selected_colour_crop_paths": ["crop1.jpg"],
                    "crop_level_colours": [
                        {"crop_path": "crop1.jpg", "normalized_colour": "WHITE", "raw_response": "white", "inference_time_ms": 12.0}
                    ],
                    "evidence_used": [
                        {"vehicle_crop_path": "crop1.jpg", "original_crop_width": 100, "original_crop_height": 80, "quality_score": 0.7}
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    result = run_benchmark(run_dir, output_dir)
    assert Path(result["csv_path"]).exists()
    assert Path(result["summary_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert Path(result["manual_review_path"]).exists()
