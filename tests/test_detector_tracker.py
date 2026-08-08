from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.detector_tracker as detector_tracker_module
import src.runtime_device as runtime_device_module
from src.detector_tracker import VehicleDetectorTracker, resolve_runtime_device
from src.models import ConfigurationError, FramePacket
from src.runtime_device import CPU_ONLY_BUILD_REASON


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
    def __init__(self, xyxy, cls, conf, speed=None):
        self.boxes = FakeBoxes(xyxy, cls, conf)
        self.speed = speed or {}


class FakeModel:
    def __init__(self, path: str):
        self.path = path
        self.names = {0: "car", 1: "truck", 2: "bus", 3: "motorcycle", 4: "3 wheeler", 5: "van", 6: "tractor", 7: "pickup"}
        self.predict_calls = []
        self.call_calls = []
        self.next_result = FakeResult([], [], [])
        self.next_results = None

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        source = kwargs.get("source")
        if isinstance(source, list):
            if self.next_results is not None:
                return list(self.next_results)
            return [self.next_result for _ in source]
        return [self.next_result]

    def __call__(self, source, **kwargs):
        self.call_calls.append({"source": source, **kwargs})
        if isinstance(source, list):
            if self.next_results is not None:
                return list(self.next_results)
            return [self.next_result for _ in source]
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
        source_frame_width=width,
        source_frame_height=height,
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


def _config(
    model_path: str,
    *,
    bbox_quality: dict | None = None,
    show_rejected_boxes: bool = False,
    isolation_mode: str = "per_camera",
    agnostic_nms: bool = False,
    device: str = "cpu",
    dtype: str = "auto",
    detection_backend: str = "legacy_clean",
    tracking_backend: str = "supervision_bytetrack",
    image_size: int | None = None,
) -> dict:
    return {
        "detection": {
            "backend": detection_backend,
            "model_path": model_path,
            "device": device,
            "dtype": dtype,
            "confidence_threshold": 0.2 if detection_backend == "ocr_mukul" else 0.38,
            "iou_threshold": 0.45,
            "image_size": image_size if image_size is not None else (1024 if detection_backend == "ocr_mukul" else 640),
            "agnostic_nms": agnostic_nms,
            "allowed_classes": ["car", "truck", "bus", "motorcycle", "3wheeler"],
            "allowed_class_ids": list(range(8)),
            "bbox_quality": bbox_quality or _bbox_quality_config(),
        },
        "tracking": {
            "backend": tracking_backend,
            "isolation_mode": isolation_mode,
            "supported_isolation_modes": ["per_camera"] if tracking_backend == "ocr_mukul_supervision_bytetrack" else ["per_camera", "per_camera_class"],
            "track_activation_threshold": 0.3 if tracking_backend == "ocr_mukul_supervision_bytetrack" else 0.15,
            "lost_track_buffer": 40 if tracking_backend == "ocr_mukul_supervision_bytetrack" else 30,
            "minimum_matching_threshold": 0.6 if tracking_backend == "ocr_mukul_supervision_bytetrack" else 0.80,
            "minimum_consecutive_frames": 3 if tracking_backend == "ocr_mukul_supervision_bytetrack" else 1,
        },
        "visualization": {
            "show_rejected_boxes": show_rejected_boxes,
        },
    }


def _build_tracker(
    tmp_path: Path,
    *,
    bbox_quality: dict | None = None,
    show_rejected_boxes: bool = False,
    isolation_mode: str = "per_camera",
    agnostic_nms: bool = False,
    detection_backend: str = "legacy_clean",
    tracking_backend: str = "supervision_bytetrack",
    image_size: int | None = None,
):
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"x")
    model = FakeModel(str(model_path))
    created_trackers: list[FakeTracker] = []

    def tracker_factory(frame_rate: float):
        tracker = FakeTracker(frame_rate)
        created_trackers.append(tracker)
        return tracker

    detector_tracker = VehicleDetectorTracker(
        _config(
            str(model_path),
            bbox_quality=bbox_quality,
            show_rejected_boxes=show_rejected_boxes,
            isolation_mode=isolation_mode,
            agnostic_nms=agnostic_nms,
            detection_backend=detection_backend,
            tracking_backend=tracking_backend,
            image_size=image_size,
        ),
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
    assert model.predict_calls[0]["agnostic_nms"] is False
    assert model.predict_calls[0]["device"] == "cpu"


def test_auto_resolves_to_cuda0_when_cuda_is_available(monkeypatch) -> None:
    monkeypatch.setattr(runtime_device_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime_device_module.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(runtime_device_module.torch.cuda, "get_device_name", lambda index: f"GPU-{index}")
    info = resolve_runtime_device("auto", "auto")
    assert info.configured_device == "auto"
    assert info.configured_dtype == "auto"
    assert info.resolved_device == "cuda:0"
    assert info.resolved_dtype == "float16"
    assert info.cuda_available is True
    assert info.cuda_device_count == 2
    assert info.cuda_device_name == "GPU-0"


def test_auto_resolves_to_cpu_when_cuda_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(runtime_device_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runtime_device_module.torch.version, "cuda", None)
    info = resolve_runtime_device("auto", "auto")
    assert info.resolved_device == "cpu"
    assert info.resolved_dtype == "float32"
    assert info.cuda_available is False
    assert info.cuda_device_count == 0
    assert info.cuda_device_name is None
    assert info.reason == CPU_ONLY_BUILD_REASON


def test_cpu_always_resolves_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(runtime_device_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime_device_module.torch.cuda, "device_count", lambda: 1)
    info = resolve_runtime_device("cpu", "auto")
    assert info.resolved_device == "cpu"
    assert info.resolved_dtype == "float32"
    assert info.cuda_device_name is None


def test_explicit_cuda_resolves_to_cuda0_when_available(monkeypatch) -> None:
    monkeypatch.setattr(runtime_device_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime_device_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(runtime_device_module.torch.cuda, "get_device_name", lambda index: "GPU-0")
    info = resolve_runtime_device("cuda", "auto")
    assert info.resolved_device == "cuda:0"
    assert info.cuda_device_name == "GPU-0"


def test_explicit_cuda0_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(runtime_device_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime_device_module.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(runtime_device_module.torch.cuda, "get_device_name", lambda index: f"GPU-{index}")
    info = resolve_runtime_device("cuda:0", "auto")
    assert info.resolved_device == "cuda:0"
    assert info.cuda_device_name == "GPU-0"


def test_explicit_cuda_raises_when_cuda_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(runtime_device_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runtime_device_module.torch.version, "cuda", "12.1")
    with pytest.raises(ConfigurationError, match="explicitly requested"):
        resolve_runtime_device("cuda", "auto")


def test_explicit_valid_cuda_index_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(runtime_device_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime_device_module.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(runtime_device_module.torch.cuda, "get_device_name", lambda index: f"GPU-{index}")
    info = resolve_runtime_device("cuda:1", "auto")
    assert info.resolved_device == "cuda:1"
    assert info.cuda_device_name == "GPU-1"


def test_explicit_invalid_cuda_index_raises(monkeypatch) -> None:
    monkeypatch.setattr(runtime_device_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime_device_module.torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ConfigurationError, match="invalid"):
        resolve_runtime_device("cuda:1", "auto")


def test_unsupported_device_value_raises(monkeypatch) -> None:
    monkeypatch.setattr(runtime_device_module.torch.cuda, "is_available", lambda: False)
    with pytest.raises(ConfigurationError, match="Unsupported device value"):
        resolve_runtime_device("tpu", "auto")


def test_explicit_cpu_with_float16_raises(monkeypatch) -> None:
    monkeypatch.setattr(runtime_device_module.torch.cuda, "is_available", lambda: True)
    with pytest.raises(ConfigurationError, match="float16 requires a CUDA device"):
        resolve_runtime_device("cpu", "float16")


def test_gpu_name_is_included_when_cuda_is_selected(monkeypatch) -> None:
    monkeypatch.setattr(runtime_device_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime_device_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(runtime_device_module.torch.cuda, "get_device_name", lambda index: "NVIDIA Test GPU")
    info = resolve_runtime_device("auto", "auto")
    assert info.cuda_device_name == "NVIDIA Test GPU"


def test_gpu_name_is_null_when_cpu_is_selected(monkeypatch) -> None:
    monkeypatch.setattr(runtime_device_module.torch.cuda, "is_available", lambda: False)
    info = resolve_runtime_device("auto", "auto")
    assert info.cuda_device_name is None


def test_tracker_instance_count_reflects_created_tracker_count(tmp_path: Path) -> None:
    tracker, model, created_trackers = _build_tracker(tmp_path)
    model.next_result = FakeResult(
        xyxy=[[20, 20, 120, 120]],
        cls=[0],
        conf=[0.9],
    )
    tracker.process_frame(_frame_packet(camera_id="CAM_001"))
    tracker.process_frame(_frame_packet(camera_id="CAM_002", frame_number=1))
    metrics = tracker.metrics
    assert len(created_trackers) == 2
    assert metrics["tracker_instance_count"] == 2


def test_resolved_device_is_passed_to_yolo_inference(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runtime_device_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime_device_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(runtime_device_module.torch.cuda, "get_device_name", lambda index: "GPU-0")
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"x")
    model = FakeModel(str(model_path))
    tracker = VehicleDetectorTracker(
        _config(str(model_path), device="auto"),
        _logger(),
        model_loader=lambda _: model,
        tracker_factory=lambda frame_rate: FakeTracker(frame_rate),
    )
    tracker.process_frame(_frame_packet())
    assert model.predict_calls[0]["device"] == 0
    assert model.predict_calls[0]["half"] is True


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
    config["detection"]["allowed_classes"] = ["airplane"]
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
    assert result.tracked_detections[0].tracker_namespace == "camera"
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


def test_default_isolation_mode_remains_per_camera(tmp_path: Path) -> None:
    detector_tracker, _model, _trackers = _build_tracker(tmp_path)
    assert detector_tracker.isolation_mode == "per_camera"


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


def test_per_camera_class_uses_one_tracker_per_camera_and_class_and_yolo_once(tmp_path: Path) -> None:
    detector_tracker, model, created_trackers = _build_tracker(
        tmp_path,
        isolation_mode="per_camera_class",
        bbox_quality=_bbox_quality_config(minimum_width_pixels=5, minimum_height_pixels=5, reject_edge_truncated=False),
    )
    model.next_result = FakeResult(
        xyxy=[[20, 20, 120, 120], [30, 30, 100, 160]],
        cls=[0, 3],
        conf=[0.9, 0.85],
    )
    result = detector_tracker.process_frame(_frame_packet())
    assert len(model.predict_calls) == 1
    assert len(created_trackers) == 2
    assert {item.tracker_namespace for item in result.tracked_detections} == {"car", "motorcycle"}
    assert all(item.tracker_id == 1 for item in result.tracked_detections)


def test_detections_are_grouped_by_normalized_class_and_enter_only_matching_tracker(tmp_path: Path) -> None:
    detector_tracker, model, created_trackers = _build_tracker(
        tmp_path,
        isolation_mode="per_camera_class",
        bbox_quality=_bbox_quality_config(minimum_width_pixels=5, minimum_height_pixels=5, reject_edge_truncated=False),
    )
    model.next_result = FakeResult(
        xyxy=[[20, 20, 120, 120], [30, 30, 100, 160], [40, 40, 150, 180]],
        cls=[0, 3, 4],
        conf=[0.9, 0.85, 0.8],
    )
    detector_tracker.process_frame(_frame_packet())
    assert len(created_trackers) == 3
    forwarded_class_ids = sorted(tracker.calls[0].class_id.tolist() for tracker in created_trackers)
    assert forwarded_class_ids == [[0], [3], [4]]


def test_same_native_tracker_id_from_different_class_trackers_produces_different_namespaced_results(tmp_path: Path) -> None:
    detector_tracker, model, _created_trackers = _build_tracker(
        tmp_path,
        isolation_mode="per_camera_class",
        bbox_quality=_bbox_quality_config(minimum_width_pixels=5, minimum_height_pixels=5, reject_edge_truncated=False),
    )
    model.next_result = FakeResult(
        xyxy=[[20, 20, 120, 120], [30, 30, 100, 160]],
        cls=[0, 3],
        conf=[0.9, 0.85],
    )
    result = detector_tracker.process_frame(_frame_packet())
    local_ids = {f"{item.camera_id}:{item.tracker_namespace.upper()}:TRACK_{item.tracker_id}" for item in result.tracked_detections}
    assert local_ids == {"CAM_001:CAR:TRACK_1", "CAM_001:MOTORCYCLE:TRACK_1"}


def test_empty_class_tracker_updates_are_sent_for_existing_class_trackers(tmp_path: Path) -> None:
    detector_tracker, model, created_trackers = _build_tracker(
        tmp_path,
        isolation_mode="per_camera_class",
        bbox_quality=_bbox_quality_config(minimum_width_pixels=5, minimum_height_pixels=5, reject_edge_truncated=False),
    )
    model.next_result = FakeResult(
        xyxy=[[20, 20, 120, 120], [30, 30, 100, 160]],
        cls=[0, 3],
        conf=[0.9, 0.85],
    )
    detector_tracker.process_frame(_frame_packet())
    model.next_result = FakeResult(xyxy=[[20, 20, 120, 120]], cls=[0], conf=[0.92])
    detector_tracker.process_frame(_frame_packet(frame_number=1))
    assert len(created_trackers) == 2
    assert created_trackers[0].calls[1].xyxy.tolist() == [[20.0, 20.0, 120.0, 120.0]]
    assert created_trackers[1].calls[1].xyxy.tolist() == []


def test_reset_camera_resets_every_class_tracker_for_that_camera(tmp_path: Path) -> None:
    detector_tracker, model, _created_trackers = _build_tracker(
        tmp_path,
        isolation_mode="per_camera_class",
        bbox_quality=_bbox_quality_config(minimum_width_pixels=5, minimum_height_pixels=5, reject_edge_truncated=False),
    )
    model.next_result = FakeResult(
        xyxy=[[20, 20, 120, 120], [30, 30, 100, 160]],
        cls=[0, 3],
        conf=[0.9, 0.85],
    )
    detector_tracker.process_frame(_frame_packet(camera_id="CAM_001"))
    detector_tracker.process_frame(_frame_packet(camera_id="CAM_002", frame_number=0))
    detector_tracker.reset_camera("CAM_001")
    assert ("CAM_001", "car") not in detector_tracker._trackers
    assert ("CAM_001", "motorcycle") not in detector_tracker._trackers
    assert ("CAM_002", "car") in detector_tracker._trackers


def test_ocr_mukul_backend_uses_callable_model_and_ocr_tracker_params(monkeypatch, tmp_path: Path) -> None:
    captured_kwargs: list[dict] = []

    class ByteTrackStub:
        def __init__(self, **kwargs):
            captured_kwargs.append(kwargs)

        def update_with_detections(self, detections):
            if len(detections.xyxy) == 0:
                return FakeTrackedDetections([], [], [], [])
            return FakeTrackedDetections(
                xyxy=detections.xyxy,
                confidence=detections.confidence,
                class_id=detections.class_id,
                tracker_id=list(range(1, len(detections.xyxy) + 1)),
            )

    monkeypatch.setattr(detector_tracker_module.sv, "ByteTrack", ByteTrackStub)
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"x")
    model = FakeModel(str(model_path))
    model.next_result = FakeResult(xyxy=[[20, 20, 120, 120]], cls=[6], conf=[0.91])
    tracker = VehicleDetectorTracker(
        _config(
            str(model_path),
            detection_backend="ocr_mukul",
            tracking_backend="ocr_mukul_supervision_bytetrack",
        ),
        _logger(),
        model_loader=lambda _: model,
    )
    result = tracker.process_frame(_frame_packet())
    assert len(model.call_calls) == 1
    assert model.call_calls[0]["conf"] == 0.2
    assert model.call_calls[0]["imgsz"] == 1024
    assert model.predict_calls == []
    assert captured_kwargs == [
        {
            "lost_track_buffer": 40,
            "track_activation_threshold": 0.3,
            "minimum_matching_threshold": 0.6,
            "minimum_consecutive_frames": 3,
        }
    ]
    assert result.tracked_detections[0].raw_class_name == "tractor"


def test_ocr_mukul_backend_creates_one_tracker_per_camera(tmp_path: Path) -> None:
    detector_tracker, model, created_trackers = _build_tracker(
        tmp_path,
        detection_backend="ocr_mukul",
        tracking_backend="ocr_mukul_supervision_bytetrack",
    )
    model.next_result = FakeResult(xyxy=[[20, 20, 120, 120]], cls=[6], conf=[0.9])
    detector_tracker.process_frame(_frame_packet(camera_id="CAM_001", fps=12.0))
    detector_tracker.process_frame(_frame_packet(camera_id="CAM_002", fps=20.0))
    assert len(created_trackers) == 2
    assert created_trackers[0].frame_rate == 12.0
    assert created_trackers[1].frame_rate == 20.0
    assert detector_tracker.metrics["tracker_camera_ids"] == ["CAM_001", "CAM_002"]


def test_process_frames_batch_size_one_preserves_single_frame_behavior(tmp_path: Path) -> None:
    detector_tracker, model, _created_trackers = _build_tracker(tmp_path)
    model.next_result = FakeResult(xyxy=[[20, 20, 120, 120]], cls=[0], conf=[0.9])
    result = detector_tracker.process_frames([_frame_packet()])[0]
    assert len(model.predict_calls) == 1
    assert not isinstance(model.predict_calls[0]["source"], list)
    assert len(result.detections) == 1
    assert detector_tracker.metrics["yolo_model_invocations"] == 1


def test_process_frames_uses_one_true_batch_for_multiple_frames(tmp_path: Path) -> None:
    detector_tracker, model, _created_trackers = _build_tracker(tmp_path)
    model.next_results = [
        FakeResult(xyxy=[[20, 20, 120, 120]], cls=[0], conf=[0.91]),
        FakeResult(xyxy=[[30, 30, 140, 150]], cls=[1], conf=[0.88]),
        FakeResult(xyxy=[[40, 40, 160, 170]], cls=[3], conf=[0.86]),
        FakeResult(xyxy=[[50, 50, 180, 190]], cls=[4], conf=[0.84]),
    ]
    packets = [
        _frame_packet(camera_id="CAM_001", frame_number=10),
        _frame_packet(camera_id="CAM_002", frame_number=20),
        _frame_packet(camera_id="CAM_003", frame_number=30),
        _frame_packet(camera_id="CAM_004", frame_number=40),
    ]
    results = detector_tracker.process_frames(packets)
    assert len(model.predict_calls) == 1
    assert isinstance(model.predict_calls[0]["source"], list)
    assert len(model.predict_calls[0]["source"]) == 4
    assert [result.tracked_detections[0].camera_id for result in results] == ["CAM_001", "CAM_002", "CAM_003", "CAM_004"]
    assert [result.tracked_detections[0].frame_number for result in results] == [10, 20, 30, 40]
    assert detector_tracker.metrics["detection_batches_total"] == 1
    assert detector_tracker.metrics["max_detection_batch_size_observed"] == 4


def test_process_frames_preserves_timestamp_and_routes_trackers_per_camera(tmp_path: Path) -> None:
    detector_tracker, model, created_trackers = _build_tracker(tmp_path)
    model.next_results = [
        FakeResult(xyxy=[[20, 20, 120, 120]], cls=[0], conf=[0.91]),
        FakeResult(xyxy=[[30, 30, 140, 150]], cls=[0], conf=[0.88]),
    ]
    packets = [
        _frame_packet(camera_id="CAM_001", frame_number=10, fps=10.0),
        _frame_packet(camera_id="CAM_002", frame_number=20, fps=20.0),
    ]
    results = detector_tracker.process_frames(packets)
    assert results[0].tracked_detections[0].timestamp_seconds == pytest.approx(1.0)
    assert results[1].tracked_detections[0].timestamp_seconds == pytest.approx(1.0)
    assert len(created_trackers) == 2
    assert created_trackers[0].frame_rate == 10.0
    assert created_trackers[1].frame_rate == 20.0


def test_process_frames_records_partial_final_batch_metrics(tmp_path: Path) -> None:
    detector_tracker, model, _created_trackers = _build_tracker(tmp_path)
    detector_tracker._metrics["detection_batch_size_configured"] = 4
    model.next_results = [
        FakeResult(xyxy=[[20, 20, 120, 120]], cls=[0], conf=[0.9]),
        FakeResult(xyxy=[[30, 30, 140, 150]], cls=[0], conf=[0.88]),
    ]
    detector_tracker.process_frames([
        _frame_packet(camera_id="CAM_001"),
        _frame_packet(camera_id="CAM_002", frame_number=1),
    ])
    metrics = detector_tracker.metrics
    assert metrics["partial_detection_batches"] == 1
    assert metrics["detection_frames_total"] == 2
    assert metrics["yolo_model_invocations"] == 1


@pytest.mark.parametrize("image_size", [1024, 896, 768])
def test_configured_imgsz_is_forwarded_to_yolo_inference(tmp_path: Path, image_size: int) -> None:
    detector_tracker, model, _created_trackers = _build_tracker(tmp_path, image_size=image_size)
    model.next_result = FakeResult(xyxy=[[20, 20, 120, 120]], cls=[0], conf=[0.9])
    detector_tracker.process_frame(_frame_packet())
    assert model.predict_calls[0]["imgsz"] == image_size


def test_process_frame_records_stage_timing_metrics_from_result_speed(tmp_path: Path) -> None:
    detector_tracker, model, _created_trackers = _build_tracker(tmp_path)
    model.next_result = FakeResult(
        xyxy=[[20, 20, 120, 120]],
        cls=[0],
        conf=[0.9],
        speed={"preprocess": 1.5, "inference": 8.25, "postprocess": 2.75},
    )
    result = detector_tracker.process_frame(_frame_packet())
    metrics = detector_tracker.metrics
    assert result.preprocess_ms == pytest.approx(1.5)
    assert result.model_inference_ms == pytest.approx(8.25)
    assert result.postprocess_ms == pytest.approx(2.75)
    assert result.result_conversion_ms >= 0.0
    assert result.result_routing_ms >= 0.0
    assert result.tracker_update_ms >= 0.0
    assert result.total_detection_ms >= result.result_conversion_ms + result.result_routing_ms
    assert metrics["preprocess_times_ms"] == [pytest.approx(1.5)]
    assert metrics["model_inference_stage_times_ms"] == [pytest.approx(8.25)]
    assert metrics["postprocess_times_ms"] == [pytest.approx(2.75)]
    assert len(metrics["total_detection_times_ms"]) == 1
