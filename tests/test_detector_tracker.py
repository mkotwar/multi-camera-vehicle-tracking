from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.detector_tracker import VehicleDetectorTracker
from src.models import ConfigurationError, FramePacket


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


class FakeModel:
    def __init__(self, path: str):
        self.path = path
        self.names = {0: "car", 1: "truck", 2: "bus", 3: "motorcycle", 4: "3 wheeler"}
        self.predict_calls = []
        self.next_result = FakeResult([], [], [])

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return [self.next_result]


class FakeTrackedDetections:
    def __init__(self, xyxy, confidence, class_id, tracker_id):
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.confidence = np.asarray(confidence, dtype=np.float32)
        self.class_id = np.asarray(class_id, dtype=np.int32)
        self.tracker_id = np.asarray(tracker_id, dtype=np.int32)


class FakeTracker:
    def __init__(self, frame_rate: float):
        self.frame_rate = frame_rate
        self.calls = []

    def update_with_detections(self, detections):
        self.calls.append(detections)
        if len(detections.xyxy) == 0:
            return FakeTrackedDetections([], [], [], [])
        return FakeTrackedDetections(
            xyxy=detections.xyxy,
            confidence=detections.confidence,
            class_id=detections.class_id,
            tracker_id=list(range(1, len(detections.xyxy) + 1)),
        )


def _logger():
    import logging

    logger = logging.getLogger("detector-tracker-test")
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(logging.NullHandler())
    return logger


def _frame_packet(
    camera_id: str = "CAM_001",
    frame_number: int = 0,
    fps: float = 10.0,
    width: int = 320,
    height: int = 240,
) -> FramePacket:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    return FramePacket(
        camera_id=camera_id,
        frame_number=frame_number,
        timestamp_seconds=frame_number / fps,
        source_fps=fps,
        frame=frame,
        worker_id=0,
        captured_at="2026-07-29T00:00:00+00:00",
        source_type="video",
    )


def _bbox_quality_config(**overrides) -> dict:
    config = {
        "enabled": True,
        "minimum_width_pixels": 60,
        "minimum_height_pixels": 60,
        "minimum_area_ratio": 0.005,
        "maximum_area_ratio": 0.90,
        "minimum_aspect_ratio": 0.30,
        "maximum_aspect_ratio": 4.50,
        "reject_edge_truncated": True,
        "edge_margin_pixels": 8,
    }
    config.update(overrides)
    return config


def _per_class_bbox_quality_config(*, default_overrides: dict | None = None, classes: dict[str, dict] | None = None) -> dict:
    default_payload = _bbox_quality_config()
    if default_overrides:
        default_payload.update(default_overrides)
    return {
        "enabled": True,
        "default": default_payload,
        "classes": classes or {},
    }


def _config(model_path: str, *, bbox_quality: dict | None = None, show_rejected_boxes: bool = False) -> dict:
    return {
        "detection": {
            "model_path": model_path,
            "device": "cpu",
            "confidence_threshold": 0.38,
            "iou_threshold": 0.45,
            "image_size": 640,
            "allowed_classes": ["car", "truck", "bus", "motorcycle", "3wheeler"],
            "bbox_quality": bbox_quality or _bbox_quality_config(),
        },
        "tracking": {
            "backend": "supervision_bytetrack",
            "track_activation_threshold": 0.15,
            "lost_track_buffer": 30,
            "minimum_matching_threshold": 0.80,
            "minimum_consecutive_frames": 1,
        },
        "visualization": {
            "show_rejected_boxes": show_rejected_boxes,
        },
    }


def _build_tracker(tmp_path: Path, *, bbox_quality: dict | None = None, show_rejected_boxes: bool = False):
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"x")
    model = FakeModel(str(model_path))
    created_trackers: list[FakeTracker] = []

    def tracker_factory(frame_rate: float):
        tracker = FakeTracker(frame_rate)
        created_trackers.append(tracker)
        return tracker

    detector_tracker = VehicleDetectorTracker(
        _config(str(model_path), bbox_quality=bbox_quality, show_rejected_boxes=show_rejected_boxes),
        _logger(),
        model_loader=lambda _: model,
        tracker_factory=tracker_factory,
    )
    return detector_tracker, model, created_trackers


def _build_profiled_tracker(tmp_path: Path, *, bbox_quality: dict):
    return _build_tracker(tmp_path, bbox_quality=bbox_quality)


def test_missing_model_path_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        VehicleDetectorTracker(_config(str(tmp_path / "missing.pt")), _logger())


def test_model_is_loaded_only_once_and_config_is_passed(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"x")
    load_count = {"count": 0}

    def loader(path: str):
        load_count["count"] += 1
        return FakeModel(path)

    tracker = VehicleDetectorTracker(
        _config(str(model_path)),
        _logger(),
        model_loader=loader,
        tracker_factory=lambda frame_rate: FakeTracker(frame_rate),
    )
    tracker.process_frame(_frame_packet())
    tracker.process_frame(_frame_packet(frame_number=1))
    model = tracker._model
    assert load_count["count"] == 1
    assert isinstance(model, FakeModel)
    assert model.predict_calls[0]["conf"] == 0.38
    assert model.predict_calls[0]["iou"] == 0.45
    assert model.predict_calls[0]["imgsz"] == 640


def test_allowed_classes_are_filtered_and_empty_results_do_not_crash(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"x")
    model = FakeModel(str(model_path))
    model.next_result = FakeResult(
        xyxy=[[20, 20, 120, 120], [30, 30, 140, 150]],
        cls=[0, 99],
        conf=[0.9, 0.8],
    )
    tracker = VehicleDetectorTracker(
        _config(str(model_path)),
        _logger(),
        model_loader=lambda _: model,
        tracker_factory=lambda frame_rate: FakeTracker(frame_rate),
    )
    result = tracker.process_frame(_frame_packet())
    assert len(result.detections) == 1
    assert result.detections[0].class_name == "car"
    model.next_result = FakeResult([], [], [])
    empty_result = tracker.process_frame(_frame_packet(frame_number=1))
    assert empty_result.detections == []
    assert empty_result.tracked_detections == []


def test_unknown_configured_class_names_are_reported_clearly(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"x")
    config = _config(str(model_path))
    config["detection"]["allowed_classes"] = ["tractor"]
    with pytest.raises(ConfigurationError):
        VehicleDetectorTracker(config, _logger(), model_loader=lambda _: FakeModel(str(model_path)))


def test_one_tracker_per_camera_and_reset_methods(tmp_path: Path) -> None:
    detector_tracker, model, created_trackers = _build_tracker(tmp_path)
    model.next_result = FakeResult(xyxy=[[20, 20, 120, 120]], cls=[0], conf=[0.9])
    detector_tracker.process_frame(_frame_packet(camera_id="CAM_001", fps=12.0))
    detector_tracker.process_frame(_frame_packet(camera_id="CAM_001", frame_number=1, fps=12.0))
    detector_tracker.process_frame(_frame_packet(camera_id="CAM_002", fps=20.0))
    assert len(created_trackers) == 2
    assert created_trackers[0].frame_rate == 12.0
    assert created_trackers[1].frame_rate == 20.0
    detector_tracker.reset_camera("CAM_001")
    assert "CAM_001" not in detector_tracker._trackers
    assert "CAM_002" in detector_tracker._trackers
    detector_tracker.reset_all()
    assert detector_tracker._trackers == {}


def test_native_tracker_ids_and_raw_class_confidence_are_preserved_and_annotations_split_roles(tmp_path: Path) -> None:
    detector_tracker, model, _created_trackers = _build_tracker(tmp_path)
    model.next_result = FakeResult(xyxy=[[20, 20, 120, 120]], cls=[0], conf=[0.84])
    result = detector_tracker.process_frame(_frame_packet())
    assert result.tracked_detections[0].tracker_id == 1
    assert result.tracked_detections[0].raw_class_name == "car"
    assert result.tracked_detections[0].confidence == pytest.approx(0.84)
    assert detector_tracker.build_detected_label(result.detections[0]) == "CAR 0.84"
    assert detector_tracker.build_tracked_label("CAM_001", result.tracked_detections[0]) == "CAM_001 | TRACK_1 | CAR | 0.84"
    assert result.detected_frame.shape == (240, 320, 3)
    assert result.tracked_frame.shape == (240, 320, 3)


def test_normal_vehicle_bbox_is_accepted(tmp_path: Path) -> None:
    detector_tracker, model, created_trackers = _build_tracker(tmp_path)
    model.next_result = FakeResult(xyxy=[[40, 40, 160, 150]], cls=[0], conf=[0.91])
    result = detector_tracker.process_frame(_frame_packet())
    assert len(result.detections) == 1
    assert result.bbox_quality_diagnostics[0].accepted_by_bbox_quality is True
    assert result.bbox_quality_diagnostics[0].rejection_reason is None
    assert len(created_trackers[0].calls[0].xyxy) == 1


@pytest.mark.parametrize(
    ("bbox_quality", "bbox", "expected_reason"),
    [
        (_bbox_quality_config(minimum_width_pixels=5, minimum_height_pixels=5, minimum_area_ratio=0.02), [40, 40, 80, 70], "BBOX_TOO_SMALL"),
        (_bbox_quality_config(), [40, 40, 90, 150], "BBOX_TOO_NARROW"),
        (_bbox_quality_config(), [40, 40, 150, 90], "BBOX_TOO_SHORT"),
        (_bbox_quality_config(minimum_width_pixels=5, minimum_height_pixels=5), [40, 40, 55, 180], "ASPECT_RATIO_TOO_LOW"),
        (_bbox_quality_config(minimum_width_pixels=5, minimum_height_pixels=5), [40, 40, 260, 80], "ASPECT_RATIO_TOO_HIGH"),
        (_bbox_quality_config(minimum_width_pixels=5, minimum_height_pixels=5, maximum_area_ratio=0.50), [10, 10, 310, 230], "BBOX_TOO_LARGE"),
        (_bbox_quality_config(minimum_width_pixels=5, minimum_height_pixels=5), [250, 40, 319, 160], "EDGE_TRUNCATED"),
        (_bbox_quality_config(minimum_width_pixels=5, minimum_height_pixels=5), [0, 40, 80, 160], "EDGE_TRUNCATED"),
        (_bbox_quality_config(minimum_width_pixels=5, minimum_height_pixels=5), [40, 0, 160, 120], "EDGE_TRUNCATED"),
        (_bbox_quality_config(minimum_width_pixels=5, minimum_height_pixels=5), [40, 150, 160, 239], "EDGE_TRUNCATED"),
    ],
)
def test_bbox_quality_rejections_return_clear_reasons(
    tmp_path: Path,
    bbox_quality: dict,
    bbox: list[float],
    expected_reason: str,
) -> None:
    detector_tracker, model, created_trackers = _build_tracker(tmp_path, bbox_quality=bbox_quality)
    model.next_result = FakeResult(xyxy=[bbox], cls=[0], conf=[0.88])
    result = detector_tracker.process_frame(_frame_packet())
    assert result.detections == []
    assert result.tracked_detections == []
    assert result.bbox_quality_diagnostics[0].accepted_by_bbox_quality is False
    assert result.bbox_quality_diagnostics[0].rejection_reason == expected_reason
    assert len(created_trackers[0].calls[0].xyxy) == 0


def test_edge_rejection_can_be_disabled(tmp_path: Path) -> None:
    detector_tracker, model, created_trackers = _build_tracker(
        tmp_path,
        bbox_quality=_bbox_quality_config(minimum_width_pixels=5, minimum_height_pixels=5, reject_edge_truncated=False),
    )
    model.next_result = FakeResult(xyxy=[[0, 40, 100, 160]], cls=[0], conf=[0.92])
    result = detector_tracker.process_frame(_frame_packet())
    assert len(result.detections) == 1
    assert result.bbox_quality_diagnostics[0].touches_edge is True
    assert result.bbox_quality_diagnostics[0].accepted_by_bbox_quality is True
    assert len(created_trackers[0].calls[0].xyxy) == 1


def test_rejected_detection_never_reaches_bytetrack(tmp_path: Path) -> None:
    detector_tracker, model, created_trackers = _build_tracker(tmp_path)
    model.next_result = FakeResult(xyxy=[[40, 40, 80, 80]], cls=[0], conf=[0.90])
    result = detector_tracker.process_frame(_frame_packet())
    assert result.detections == []
    assert len(created_trackers[0].calls[0].xyxy) == 0


def test_accepted_detection_reaches_bytetrack_unchanged(tmp_path: Path) -> None:
    detector_tracker, model, created_trackers = _build_tracker(tmp_path)
    bbox = [40, 40, 170, 160]
    model.next_result = FakeResult(xyxy=[bbox], cls=[0], conf=[0.90])
    result = detector_tracker.process_frame(_frame_packet())
    forwarded = created_trackers[0].calls[0]
    assert len(result.detections) == 1
    assert forwarded.xyxy.tolist() == [bbox]
    assert forwarded.class_id.tolist() == [0]
    assert forwarded.confidence.tolist() == [pytest.approx(0.90)]


def test_per_class_profile_lookup_uses_specific_class_and_unknown_uses_default(tmp_path: Path) -> None:
    detector_tracker, _model, _trackers = _build_profiled_tracker(
        tmp_path,
        bbox_quality=_per_class_bbox_quality_config(
            default_overrides={"minimum_width_pixels": 60},
            classes={"car": {"minimum_width_pixels": 25}, "motorcycle": {"minimum_width_pixels": 12}},
        ),
    )
    assert detector_tracker.get_bbox_quality_profile("car").minimum_width_pixels == 25
    assert detector_tracker.get_bbox_quality_profile("bike").minimum_width_pixels == 12
    assert detector_tracker.get_bbox_quality_profile("unknown").minimum_width_pixels == 60


def test_every_yaml_value_is_configurable_per_class(tmp_path: Path) -> None:
    detector_tracker, _model, _trackers = _build_profiled_tracker(
        tmp_path,
        bbox_quality=_per_class_bbox_quality_config(
            default_overrides={"edge_mode": "A"},
            classes={
                "3wheeler": {
                    "minimum_width_pixels": 21,
                    "minimum_height_pixels": 22,
                    "minimum_area_ratio": 0.0003,
                    "maximum_area_ratio": 0.88,
                    "minimum_aspect_ratio": 0.19,
                    "maximum_aspect_ratio": 2.9,
                    "edge_margin_pixels": 11,
                    "edge_mode": "B",
                }
            },
        ),
    )
    profile = detector_tracker.get_bbox_quality_profile("3 wheeler")
    assert profile.minimum_width_pixels == 21
    assert profile.minimum_height_pixels == 22
    assert profile.minimum_area_ratio == pytest.approx(0.0003)
    assert profile.maximum_area_ratio == pytest.approx(0.88)
    assert profile.minimum_aspect_ratio == pytest.approx(0.19)
    assert profile.maximum_aspect_ratio == pytest.approx(2.9)
    assert profile.edge_margin_pixels == pytest.approx(11)
    assert profile.edge_mode == "B"


def test_width_height_area_and_aspect_thresholds_work_independently_per_class(tmp_path: Path) -> None:
    detector_tracker, model, _trackers = _build_profiled_tracker(
        tmp_path,
        bbox_quality=_per_class_bbox_quality_config(
            default_overrides={"minimum_width_pixels": 5, "minimum_height_pixels": 5, "minimum_area_ratio": 0.0, "edge_mode": "A"},
            classes={
                "car": {"minimum_width_pixels": 50, "minimum_height_pixels": 40, "minimum_area_ratio": 0.0020, "minimum_aspect_ratio": 0.50, "maximum_aspect_ratio": 3.00, "edge_mode": "A"},
                "motorcycle": {"minimum_width_pixels": 15, "minimum_height_pixels": 20, "minimum_area_ratio": 0.0002, "minimum_aspect_ratio": 0.15, "maximum_aspect_ratio": 3.50, "edge_mode": "A"},
            },
        ),
    )
    model.next_result = FakeResult(
        xyxy=[
                [40, 40, 75, 95],
                [40, 40, 58, 95],
                [40, 40, 90, 70],
                [40, 40, 95, 170],
            ],
        cls=[0, 3, 0, 0],
        conf=[0.9, 0.9, 0.9, 0.9],
    )
    result = detector_tracker.process_frame(_frame_packet())
    reasons = [item.rejection_reason for item in result.bbox_quality_diagnostics]
    assert reasons == ["BBOX_TOO_NARROW", None, "BBOX_TOO_SHORT", "ASPECT_RATIO_TOO_LOW"]


def test_small_motorcycle_can_pass_while_same_sized_car_fails(tmp_path: Path) -> None:
    detector_tracker, model, created_trackers = _build_profiled_tracker(
        tmp_path,
        bbox_quality=_per_class_bbox_quality_config(
            default_overrides={"minimum_width_pixels": 5, "minimum_height_pixels": 5, "minimum_area_ratio": 0.0, "edge_mode": "A"},
            classes={
                "car": {"minimum_width_pixels": 40, "minimum_height_pixels": 35, "minimum_area_ratio": 0.0015, "minimum_aspect_ratio": 0.40, "maximum_aspect_ratio": 3.50, "edge_mode": "A"},
                "motorcycle": {"minimum_width_pixels": 12, "minimum_height_pixels": 15, "minimum_area_ratio": 0.0002, "minimum_aspect_ratio": 0.15, "maximum_aspect_ratio": 3.50, "edge_mode": "A"},
            },
        ),
    )
    bbox = [40, 40, 62, 76]
    model.next_result = FakeResult(xyxy=[bbox, bbox], cls=[0, 3], conf=[0.9, 0.9])
    result = detector_tracker.process_frame(_frame_packet())
    accepted_classes = [item.class_name for item in result.detections]
    reasons = [item.rejection_reason for item in result.bbox_quality_diagnostics]
    assert accepted_classes == ["motorcycle"]
    assert reasons == ["BBOX_TOO_NARROW", None]
    assert len(created_trackers[0].calls[0].xyxy) == 1


def test_tall_3wheeler_can_pass_its_class_profile(tmp_path: Path) -> None:
    detector_tracker, model, _trackers = _build_profiled_tracker(
        tmp_path,
        bbox_quality=_per_class_bbox_quality_config(
            default_overrides={"minimum_width_pixels": 40, "minimum_height_pixels": 40, "minimum_area_ratio": 0.001, "minimum_aspect_ratio": 0.50, "maximum_aspect_ratio": 3.00, "edge_mode": "A"},
            classes={
                "3wheeler": {"minimum_width_pixels": 25, "minimum_height_pixels": 35, "minimum_area_ratio": 0.0005, "minimum_aspect_ratio": 0.20, "maximum_aspect_ratio": 3.00, "edge_mode": "A"},
            },
        ),
    )
    model.next_result = FakeResult(xyxy=[[40, 40, 75, 150]], cls=[4], conf=[0.92])
    result = detector_tracker.process_frame(_frame_packet())
    assert [item.class_name for item in result.detections] == ["3wheeler"]
    assert result.bbox_quality_diagnostics[0].rejection_reason is None


def test_edge_mode_a_keeps_edge_detections(tmp_path: Path) -> None:
    detector_tracker, model, _trackers = _build_profiled_tracker(
        tmp_path,
        bbox_quality=_per_class_bbox_quality_config(
            default_overrides={"minimum_width_pixels": 20, "minimum_height_pixels": 20, "minimum_area_ratio": 0.001, "edge_mode": "A"},
        ),
    )
    model.next_result = FakeResult(xyxy=[[0, 40, 80, 150]], cls=[0], conf=[0.9])
    result = detector_tracker.process_frame(_frame_packet())
    assert len(result.detections) == 1
    assert result.bbox_quality_diagnostics[0].accepted_by_bbox_quality is True


def test_edge_mode_b_rejects_only_edge_plus_small_area(tmp_path: Path) -> None:
    detector_tracker, model, _trackers = _build_profiled_tracker(
        tmp_path,
        bbox_quality=_per_class_bbox_quality_config(
            default_overrides={"minimum_width_pixels": 5, "minimum_height_pixels": 5, "minimum_area_ratio": 0.05, "edge_mode": "B"},
        ),
    )
    model.next_result = FakeResult(xyxy=[[0, 40, 40, 80]], cls=[0], conf=[0.9])
    result = detector_tracker.process_frame(_frame_packet())
    assert result.detections == []
    assert result.bbox_quality_diagnostics[0].rejection_reason == "BBOX_TOO_SMALL"


def test_edge_mode_c_rejects_only_edge_plus_insufficient_dimensions(tmp_path: Path) -> None:
    detector_tracker, model, _trackers = _build_profiled_tracker(
        tmp_path,
        bbox_quality=_per_class_bbox_quality_config(
            default_overrides={"minimum_width_pixels": 60, "minimum_height_pixels": 60, "minimum_area_ratio": 0.0, "edge_mode": "C"},
        ),
    )
    model.next_result = FakeResult(xyxy=[[0, 40, 50, 150]], cls=[0], conf=[0.9])
    result = detector_tracker.process_frame(_frame_packet())
    assert result.detections == []
    assert result.bbox_quality_diagnostics[0].rejection_reason == "BBOX_TOO_NARROW"
