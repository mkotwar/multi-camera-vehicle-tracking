from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_vehicle_colour_single_crop import (
    analyze_run,
    extract_ranked_predictions,
    find_latest_suitable_run,
    normalize_colour,
    run_benchmark,
)


def test_normalize_colour_defaults_to_unknown() -> None:
    assert normalize_colour("") == "UNKNOWN"
    assert normalize_colour(None) == "UNKNOWN"
    assert normalize_colour("white") == "WHITE"


def test_extract_ranked_predictions_uses_selected_order_and_evidence_dimensions() -> None:
    enrichment_row = {
        "selected_colour_crop_paths": [
            "crop_b.jpg",
            "crop_a.jpg",
        ],
        "crop_level_colours": [
            {
                "crop_path": "crop_a.jpg",
                "normalized_colour": "WHITE",
                "raw_response": "white",
                "inference_time_ms": 11.0,
            },
            {
                "crop_path": "crop_b.jpg",
                "normalized_colour": "BLACK",
                "raw_response": "black",
                "inference_time_ms": 9.0,
            },
        ],
        "evidence_used": [
            {
                "vehicle_crop_path": "crop_b.jpg",
                "original_crop_width": 222,
                "original_crop_height": 111,
                "quality_score": 0.91,
            },
            {
                "vehicle_crop_path": "crop_a.jpg",
                "original_crop_width": 123,
                "original_crop_height": 99,
                "quality_score": 0.81,
            },
        ],
    }
    ranked = extract_ranked_predictions({}, enrichment_row)
    assert [item.crop_path for item in ranked] == ["crop_b.jpg", "crop_a.jpg"]
    assert ranked[0].parsed_colour == "BLACK"
    assert ranked[0].crop_width == 222
    assert ranked[0].crop_height == 111
    assert ranked[0].quality_score == 0.91


def test_analyze_run_computes_agreement_and_call_reduction(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260808_120000"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({"run_id": "20260808_120000"}), encoding="utf-8")
    (run_dir / "vehicle_enrichment_metrics.json").write_text(
        json.dumps({"average_colour_inference_time_ms": 100.0}),
        encoding="utf-8",
    )
    (run_dir / "vehicle_colour_results.json").write_text(json.dumps([]), encoding="utf-8")
    (run_dir / "vehicle_colour_track_summary.json").write_text(
        json.dumps(
            [
                {
                    "camera_id": "CAM_001",
                    "local_track_id": "CAM_001:TRACK_1",
                    "vehicle_class": "CAR",
                    "selected_crop_count": 3,
                    "colour_inference_count": 3,
                    "final_vehicle_colour": "WHITE",
                },
                {
                    "camera_id": "CAM_001",
                    "local_track_id": "CAM_001:TRACK_2",
                    "vehicle_class": "CAR",
                    "selected_crop_count": 2,
                    "colour_inference_count": 2,
                    "final_vehicle_colour": "BLUE",
                },
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "vehicle_enrichment.json").write_text(
        json.dumps(
            [
                {
                    "local_track_id": "CAM_001:TRACK_1",
                    "selected_colour_crop_paths": ["crop_1b.jpg", "crop_1a.jpg", "crop_1c.jpg"],
                    "crop_level_colours": [
                        {"crop_path": "crop_1a.jpg", "normalized_colour": "WHITE", "raw_response": "white", "inference_time_ms": 10.0},
                        {"crop_path": "crop_1b.jpg", "normalized_colour": "WHITE", "raw_response": "white", "inference_time_ms": 12.0},
                        {"crop_path": "crop_1c.jpg", "normalized_colour": "WHITE", "raw_response": "white", "inference_time_ms": 14.0},
                    ],
                    "evidence_used": [
                        {"vehicle_crop_path": "crop_1b.jpg", "original_crop_width": 200, "original_crop_height": 150, "quality_score": 0.9},
                        {"vehicle_crop_path": "crop_1a.jpg", "original_crop_width": 180, "original_crop_height": 140, "quality_score": 0.8},
                        {"vehicle_crop_path": "crop_1c.jpg", "original_crop_width": 170, "original_crop_height": 130, "quality_score": 0.7},
                    ],
                },
                {
                    "local_track_id": "CAM_001:TRACK_2",
                    "selected_colour_crop_paths": ["crop_2a.jpg", "crop_2b.jpg"],
                    "crop_level_colours": [
                        {"crop_path": "crop_2a.jpg", "normalized_colour": "UNKNOWN", "raw_response": "", "inference_time_ms": 20.0},
                        {"crop_path": "crop_2b.jpg", "normalized_colour": "BLUE", "raw_response": "blue", "inference_time_ms": 22.0},
                    ],
                    "evidence_used": [
                        {"vehicle_crop_path": "crop_2a.jpg", "original_crop_width": 160, "original_crop_height": 120, "quality_score": 0.6},
                        {"vehicle_crop_path": "crop_2b.jpg", "original_crop_width": 150, "original_crop_height": 110, "quality_score": 0.5},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    rows, summary, manual_review = analyze_run(run_dir, agreement_sample_size=0, seed=7)
    assert len(rows) == 2
    assert summary["tracks_tested"] == 2
    assert summary["current_3crop_florence_calls"] == 5
    assert summary["single_crop_florence_calls"] == 2
    assert summary["call_reduction"] == 3
    assert summary["percent_reduction"] == 60.0
    assert summary["same_final_colour"] == 1
    assert summary["different_final_colour"] == 1
    assert summary["agreement_rate"] == 50.0
    assert summary["one_crop_unknown_three_crop_valid"] == 1
    assert len(manual_review) == 1


def test_find_latest_suitable_run_prefers_newest_with_enough_tracks(tmp_path: Path) -> None:
    older = tmp_path / "20260807_100000"
    newer = tmp_path / "20260808_100000"
    for path, count in ((older, 10), (newer, 60)):
        path.mkdir()
        (path / "vehicle_enrichment.json").write_text("[]", encoding="utf-8")
        (path / "vehicle_colour_results.json").write_text("[]", encoding="utf-8")
        (path / "summary.json").write_text("{}", encoding="utf-8")
        (path / "vehicle_colour_track_summary.json").write_text(json.dumps([{"local_track_id": f"T{i}"} for i in range(count)]), encoding="utf-8")
    chosen = find_latest_suitable_run(search_roots=[tmp_path], min_tracks=50)
    assert chosen == newer


def test_run_benchmark_writes_expected_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260808_120001"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({"run_id": "20260808_120001"}), encoding="utf-8")
    (run_dir / "vehicle_enrichment_metrics.json").write_text(json.dumps({"average_colour_inference_time_ms": 100.0}), encoding="utf-8")
    (run_dir / "vehicle_colour_results.json").write_text("[]", encoding="utf-8")
    (run_dir / "vehicle_colour_track_summary.json").write_text(
        json.dumps(
            [
                {
                    "camera_id": "CAM_001",
                    "local_track_id": "CAM_001:TRACK_1",
                    "vehicle_class": "CAR",
                    "selected_crop_count": 1,
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
                    "selected_colour_crop_paths": ["crop.jpg"],
                    "crop_level_colours": [{"crop_path": "crop.jpg", "normalized_colour": "WHITE", "raw_response": "white", "inference_time_ms": 12.0}],
                    "evidence_used": [{"vehicle_crop_path": "crop.jpg", "original_crop_width": 120, "original_crop_height": 90, "quality_score": 0.7}],
                }
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "benchmark"
    result = run_benchmark(run_dir=run_dir, output_dir=output_dir, agreement_sample_size=0, seed=7)
    assert Path(result["csv_path"]).exists()
    assert Path(result["manual_review_path"]).exists()
    assert Path(result["summary_path"]).exists()
    assert Path(result["report_path"]).exists()
