from __future__ import annotations

from scripts.run_yolo_imgsz_benchmark import build_performance_config, choose_recommendations
from src.yolo_imgsz_benchmark import BenchmarkDetection, classify_size_group, compute_iou, summarize_parity


def _detection(
    *,
    camera_id: str = "CAM_001",
    frame_number: int = 1,
    frame_path: str = "frame.jpg",
    class_name: str = "car",
    confidence: float = 0.9,
    bbox_xyxy: tuple[float, float, float, float] = (10.0, 10.0, 30.0, 30.0),
    frame_width: int = 100,
    frame_height: int = 100,
    imgsz: int = 1024,
) -> BenchmarkDetection:
    return BenchmarkDetection(
        camera_id=camera_id,
        frame_number=frame_number,
        frame_path=frame_path,
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy=bbox_xyxy,
        frame_width=frame_width,
        frame_height=frame_height,
        imgsz=imgsz,
    )


def test_compute_iou_returns_expected_overlap() -> None:
    assert compute_iou((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)) == 1.0
    assert compute_iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)) == 0.0
    assert compute_iou((0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 15.0, 15.0)) == 25.0 / 175.0


def test_classify_size_group_uses_documented_thresholds() -> None:
    assert classify_size_group(0.010, small_threshold=0.015, medium_threshold=0.060) == "small"
    assert classify_size_group(0.020, small_threshold=0.015, medium_threshold=0.060) == "medium"
    assert classify_size_group(0.100, small_threshold=0.015, medium_threshold=0.060) == "large"


def test_summarize_parity_counts_matches_missing_and_additional() -> None:
    baseline = [
        _detection(frame_number=1, class_name="car", bbox_xyxy=(10.0, 10.0, 30.0, 30.0)),
        _detection(frame_number=1, class_name="truck", bbox_xyxy=(40.0, 40.0, 80.0, 80.0), confidence=0.8),
    ]
    candidate = [
        _detection(frame_number=1, class_name="car", bbox_xyxy=(11.0, 11.0, 31.0, 31.0), imgsz=896),
        _detection(frame_number=1, class_name="bus", bbox_xyxy=(70.0, 70.0, 90.0, 90.0), imgsz=896),
    ]
    parity = summarize_parity(baseline, candidate, iou_threshold=0.5, small_threshold=0.015, medium_threshold=0.06)
    assert parity["total_baseline_detections"] == 2
    assert parity["matched_detections"] == 1
    assert parity["missing_detections"] == 1
    assert parity["additional_detections"] == 1
    assert parity["per_class"]["car"]["matched"] == 1
    assert parity["per_class"]["truck"]["matched"] == 0


def test_summarize_parity_counts_same_box_different_class_as_mismatch() -> None:
    baseline = [_detection(class_name="car", bbox_xyxy=(10.0, 10.0, 40.0, 40.0))]
    candidate = [_detection(class_name="truck", bbox_xyxy=(10.0, 10.0, 40.0, 40.0), imgsz=768)]
    parity = summarize_parity(baseline, candidate, iou_threshold=0.5)
    assert parity["matched_detections"] == 0
    assert parity["class_mismatch_count"] == 1


def test_build_performance_config_keeps_batch_size_one_and_sets_imgsz() -> None:
    base_config = {
        "input": {"cameras": [{"camera_id": "CAM_001", "source_type": "video", "source": "sample.mp4", "enabled": True}]},
        "ingestion": {},
        "detection": {"image_size": 1024},
        "vehicle_enrichment": {"enabled": True, "async_colour": {"enabled": True}},
    }
    config = build_performance_config(base_config, camera_count=4, frame_limit=30, imgsz=896, mode="detector_only")
    assert config["detection"]["image_size"] == 896
    assert config["detection"]["batch"]["enabled"] is False
    assert config["detection"]["batch"]["max_size"] == 1
    assert config["input"]["max_frames_per_camera"] == 30
    assert config["vehicle_enrichment"]["enabled"] is False


def test_choose_recommendations_prefers_quality_preserving_smaller_size() -> None:
    detector_rows = [
        {"camera_count": 12, "imgsz": 1024, "pipeline_frames_per_second": 5.0, "p95_total_detection_latency_ms": 100.0, "cuda_peak_allocated_mb": 400.0},
        {"camera_count": 12, "imgsz": 896, "pipeline_frames_per_second": 5.8, "p95_total_detection_latency_ms": 95.0, "cuda_peak_allocated_mb": 420.0},
        {"camera_count": 12, "imgsz": 768, "pipeline_frames_per_second": 6.2, "p95_total_detection_latency_ms": 90.0, "cuda_peak_allocated_mb": 430.0},
    ]
    parity_by_size = {
        896: {"match_rate": 0.99, "size_groups": {"small": {"match_rate": 0.98}}},
        768: {"match_rate": 0.95, "size_groups": {"small": {"match_rate": 0.90}}},
    }
    recommendations = choose_recommendations(detector_rows, parity_by_size)
    assert recommendations["best_performance_size"] == 768
    assert recommendations["recommended_production_size"] == 896
