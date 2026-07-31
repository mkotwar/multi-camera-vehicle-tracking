from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from scripts.run_yolo_detection_diagnostic import (
    PROFILE_CLEAN,
    PROFILE_OCR_MUKUL,
    PROFILE_REFERENCE,
    compute_sha256,
    run_diagnostic,
    select_frame_numbers,
)
from src.detector_tracker import VehicleDetectorTracker


class FakeTensor:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class FakeBoxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = FakeTensor(xyxy)
        self.cls = FakeTensor(cls)
        self.conf = FakeTensor(conf)


class FakeResult:
    def __init__(self, xyxy, cls, conf):
        self.boxes = FakeBoxes(xyxy, cls, conf)


class SequencedFakeModel:
    def __init__(self, path: str, results: list[FakeResult]):
        self.path = path
        self.names = {0: "car", 1: "truck", 2: "bus", 3: "motorcycle", 4: "3 wheeler", 5: "person", 6: "van", 7: "tractor"}
        self._results = list(results)
        self.predict_calls: list[dict] = []
        self.call_calls: list[dict] = []

    def _next_result(self):
        return self._results.pop(0) if self._results else FakeResult([], [], [])

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return [self._next_result()]

    def __call__(self, source, **kwargs):
        self.call_calls.append({"source": source, **kwargs})
        return [self._next_result()]


def _create_test_video(path: Path, *, fps: float = 10.0, frame_count: int = 3, width: int = 96, height: int = 64) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for index in range(frame_count):
        frame = np.full((height, width, 3), 30 + index * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _write_config(path: Path, *, video_path: Path, model_path: Path) -> None:
    payload = {
        "project": {"name": "diag_test", "environment": "test", "log_level": "INFO"},
        "input": {
            "cameras": [{"camera_id": "CAM_001", "source_type": "video", "source": str(video_path), "enabled": True}],
            "max_frames_per_camera": 3,
        },
        "ingestion": {
            "worker_count": 1,
            "target_read_fps": 10.0,
            "frame_queue_size": 10,
            "queue_put_timeout_seconds": 0.1,
            "queue_get_timeout_seconds": 0.1,
            "stop_on_camera_error": False,
            "round_robin": True,
            "raw_frames": {"enabled": True, "save_every_n_frames": 1, "max_saved_frames_per_camera": 3, "image_format": "jpg", "jpeg_quality": 90},
        },
        "detection": {
            "model_path": str(model_path),
            "device": "cpu",
            "confidence_threshold": 0.38,
            "iou_threshold": 0.45,
            "image_size": 640,
            "agnostic_nms": False,
            "allowed_classes": ["car", "truck", "bus", "motorcycle", "3wheeler"],
            "bbox_quality": {
                "enabled": True,
                "minimum_width_pixels": 20,
                "minimum_height_pixels": 20,
                "minimum_area_ratio": 0.005,
                "maximum_area_ratio": 0.90,
                "minimum_aspect_ratio": 0.30,
                "maximum_aspect_ratio": 4.50,
                "reject_edge_truncated": True,
                "edge_margin_pixels": 8,
            },
        },
        "tracking": {
            "backend": "supervision_bytetrack",
            "isolation_mode": "per_camera",
            "supported_isolation_modes": ["per_camera", "per_camera_class"],
            "track_activation_threshold": 0.15,
            "lost_track_buffer": 30,
            "minimum_matching_threshold": 0.80,
            "minimum_consecutive_frames": 1,
        },
        "lifecycle": {"minimum_observations": 3, "maximum_lost_frames": 30, "keep_discarded_tracks": True},
        "track_class": {
            "minimum_observations": 3,
            "minimum_winner_ratio": 0.60,
            "strategy": "confidence_weighted_majority",
            "unknown_class_name": "UNKNOWN",
        },
        "evidence": {"enabled": True},
        "visualization": {
            "show_rejected_boxes": False,
            "detected_frames": {"enabled": True, "save_every_n_frames": 1, "max_saved_frames_per_camera": 3},
            "tracked_frames": {"enabled": True, "save_every_n_frames": 1, "max_saved_frames_per_camera": 3},
        },
        "output": {"root_directory": "outputs/runs", "save_run_config": True},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _tracker_factory(fake_model: SequencedFakeModel):
    def _factory(validated_config: dict, logger):
        return VehicleDetectorTracker(
            validated_config,
            logger,
            model_loader=lambda _: fake_model,
            tracker_factory=lambda frame_rate: (_ for _ in ()).throw(AssertionError("ByteTrack must not be created")),
        )

    return _factory


def test_select_frame_numbers_supports_frame_step_and_all_frames() -> None:
    assert select_frame_numbers(6, all_frames=False, start_frame=0, end_frame=None, frame_step=2) == [0, 2, 4]
    assert select_frame_numbers(6, all_frames=True, start_frame=1, end_frame=3, frame_step=5) == [1, 2, 3]


def test_compute_sha256_changes_when_file_changes(tmp_path: Path) -> None:
    file_path = tmp_path / "model.pt"
    file_path.write_bytes(b"abc")
    hash_a = compute_sha256(file_path)
    file_path.write_bytes(b"abcd")
    hash_b = compute_sha256(file_path)
    assert hash_a != hash_b


def test_clean_profile_writes_expected_outputs_without_tracking(tmp_path: Path) -> None:
    video_path = _create_test_video(tmp_path / "sample.mp4", frame_count=3)
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake-model")
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, video_path=video_path, model_path=model_path)

    fake_model = SequencedFakeModel(
        str(model_path),
        results=[
            FakeResult(
                xyxy=[[10, 10, 50, 50], [1, 1, 10, 10], [20, 20, 55, 55]],
                cls=[0, 0, 5],
                conf=[0.9, 0.8, 0.7],
            ),
            FakeResult(
                xyxy=[[12, 12, 52, 52], [30, 30, 60, 60]],
                cls=[0, 3],
                conf=[0.88, 0.81],
            ),
        ],
    )

    output_root = run_diagnostic(
        video_path=video_path,
        config_path=config_path,
        profile=PROFILE_CLEAN,
        frame_step=2,
        tracker_factory=_tracker_factory(fake_model),
    )

    runtime = json.loads((output_root / "runtime_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_root / "rejection_metrics.json").read_text(encoding="utf-8"))
    rejected_csv = (output_root / "rejected_detections.csv").read_text(encoding="utf-8")
    accepted_csv = (output_root / "accepted_detections.csv").read_text(encoding="utf-8")

    assert output_root.name.endswith("_clean")
    assert fake_model.predict_calls[0]["conf"] == 0.38
    assert fake_model.predict_calls[0]["iou"] == 0.45
    assert fake_model.predict_calls[0]["imgsz"] == 640
    assert fake_model.predict_calls[0]["device"] == "cpu"
    assert fake_model.predict_calls[0]["agnostic_nms"] is False
    assert len(fake_model.predict_calls) == 2
    assert runtime["profile"] == PROFILE_CLEAN
    assert runtime["video_probe"]["processed_frames"] == 2
    assert runtime["selected_frames"] == [0, 2]
    assert runtime["detection"]["model_sha256"] == compute_sha256(model_path)
    assert "CLASS_NOT_ALLOWED" in rejected_csv
    assert "BBOX_TOO_NARROW" in rejected_csv
    assert accepted_csv.count("\n") == 3
    assert metrics["rejected_by_reason"]["CLASS_NOT_ALLOWED"] == 1
    assert metrics["rejected_by_reason"]["BBOX_TOO_NARROW"] == 1
    assert metrics["rejected_by_reason"]["EDGE_TRUNCATED"] == 1

    side_by_side_files = sorted((output_root / "side_by_side_frames").glob("*.jpg"))
    assert len(side_by_side_files) == 2
    image = cv2.imread(str(side_by_side_files[0]))
    assert image is not None
    assert image.shape[1] == 96 * 2


def test_reference_profile_uses_per_class_thresholds(tmp_path: Path) -> None:
    video_path = _create_test_video(tmp_path / "sample.mp4", frame_count=1)
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake-model")
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, video_path=video_path, model_path=model_path)

    fake_model = SequencedFakeModel(
        str(model_path),
        results=[
            FakeResult(
                xyxy=[[10, 10, 50, 50], [20, 20, 60, 60], [30, 30, 70, 70]],
                cls=[0, 3, 2],
                conf=[0.52, 0.30, 0.90],
            )
        ],
    )

    output_root = run_diagnostic(
        video_path=video_path,
        config_path=config_path,
        profile=PROFILE_REFERENCE,
        all_frames=True,
        tracker_factory=_tracker_factory(fake_model),
    )

    runtime = json.loads((output_root / "runtime_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_root / "rejection_metrics.json").read_text(encoding="utf-8"))
    accepted_csv = (output_root / "accepted_detections.csv").read_text(encoding="utf-8")
    rejected_csv = (output_root / "rejected_detections.csv").read_text(encoding="utf-8")

    assert output_root.name.endswith("_reference")
    assert fake_model.predict_calls[0]["conf"] == 0.25
    assert "agnostic_nms" not in fake_model.predict_calls[0]
    assert runtime["profile"] == PROFILE_REFERENCE
    assert runtime["profile_metadata"]["class_confidence_thresholds"]["car"] == 0.7
    assert "motorcycle" in accepted_csv
    assert "bus" in accepted_csv
    assert "BELOW_CLASS_CONFIDENCE_THRESHOLD" in rejected_csv
    assert metrics["rejected_by_reason"]["BELOW_CLASS_CONFIDENCE_THRESHOLD"] == 1


def test_ocr_mukul_profile_uses_call_style_and_imgsz_1024(tmp_path: Path) -> None:
    video_path = _create_test_video(tmp_path / "sample.mp4", frame_count=1)
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake-model")
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, video_path=video_path, model_path=model_path)

    fake_model = SequencedFakeModel(
        str(model_path),
        results=[
            FakeResult(
                xyxy=[[10, 10, 50, 50], [20, 20, 60, 60]],
                cls=[7, 9],
                conf=[0.40, 0.50],
            )
        ],
    )

    output_root = run_diagnostic(
        video_path=video_path,
        config_path=config_path,
        profile=PROFILE_OCR_MUKUL,
        all_frames=True,
        tracker_factory=_tracker_factory(fake_model),
    )

    runtime = json.loads((output_root / "runtime_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_root / "rejection_metrics.json").read_text(encoding="utf-8"))
    accepted_csv = (output_root / "accepted_detections.csv").read_text(encoding="utf-8")
    rejected_csv = (output_root / "rejected_detections.csv").read_text(encoding="utf-8")

    assert output_root.name.endswith("_ocr_mukul")
    assert fake_model.call_calls[0]["conf"] == 0.2
    assert fake_model.call_calls[0]["imgsz"] == 1024
    assert fake_model.predict_calls == []
    assert runtime["profile"] == PROFILE_OCR_MUKUL
    assert runtime["profile_metadata"]["allowed_class_ids"] == list(range(8))
    assert "tractor" in accepted_csv
    assert "CLASS_ID_NOT_TRACKED_BY_OCR_MUKUL" in rejected_csv
    assert metrics["rejected_by_reason"]["CLASS_ID_NOT_TRACKED_BY_OCR_MUKUL"] == 1
