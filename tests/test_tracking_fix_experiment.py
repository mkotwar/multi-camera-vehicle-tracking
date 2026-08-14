from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import supervision as sv

import src.tracking_fix_experiment as experiment_module
from src.tracking_fix_experiment import RuntimeTrackingFixExperiment


class FakeShadowTracker:
    created_kwargs: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.removed_tracks = []
        FakeShadowTracker.created_kwargs.append(kwargs)

    def update_with_detections(self, detections):
        self.calls.append(
            {
                "xyxy": detections.xyxy.copy(),
                "confidence": detections.confidence.copy(),
                "class_id": detections.class_id.copy(),
            }
        )
        detections.tracker_id = np.arange(1, len(detections.xyxy) + 1, dtype=np.int32)
        return detections


def _detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.asarray([[10, 20, 80, 100], [100, 120, 180, 210]], dtype=np.float32),
        confidence=np.asarray([0.91, 0.28], dtype=np.float32),
        class_id=np.asarray([4, 2], dtype=np.int32),
    )


def _base_config() -> dict:
    return {
        "tracking": {
            "track_activation_threshold": 0.3,
            "lost_track_buffer": 40,
            "minimum_matching_threshold": 0.6,
            "minimum_consecutive_frames": 3,
        }
    }


def test_runtime_shadow_variants_receive_identical_detection_inputs(monkeypatch, tmp_path: Path) -> None:
    FakeShadowTracker.created_kwargs = []
    monkeypatch.setattr(experiment_module.sv, "ByteTrack", FakeShadowTracker)
    experiment = RuntimeTrackingFixExperiment(
        output_dir=tmp_path / "tracking_fix_experiment",
        base_config=_base_config(),
        logger=type("Logger", (), {"info": lambda *args, **kwargs: None})(),
        run_threshold_experiment=True,
        run_activation_experiment=False,
    )
    source = _detections()

    experiment.observe(
        camera_id="CAM_001",
        frame_number=10,
        timestamp_seconds=1.0,
        source_fps=12.5,
        detections=source,
    )

    variants = experiment.variants_by_camera["CAM_001"]
    assert sorted(variants) == ["threshold_060", "threshold_065", "threshold_070", "threshold_075"]
    for variant in variants.values():
        call = variant.tracker.calls[0]
        np.testing.assert_allclose(call["xyxy"], source.xyxy)
        np.testing.assert_allclose(call["confidence"], source.confidence)
        np.testing.assert_array_equal(call["class_id"], source.class_id)
    assert {item["minimum_matching_threshold"] for item in FakeShadowTracker.created_kwargs} == {0.60, 0.65, 0.70, 0.75}
    assert {item["track_activation_threshold"] for item in FakeShadowTracker.created_kwargs} == {0.30}
    assert {item["frame_rate"] for item in FakeShadowTracker.created_kwargs} == {12.5}
    assert experiment.input_errors == []


def test_runtime_shadow_finalize_writes_validation_and_candidate_files(monkeypatch, tmp_path: Path) -> None:
    FakeShadowTracker.created_kwargs = []
    monkeypatch.setattr(experiment_module.sv, "ByteTrack", FakeShadowTracker)
    experiment = RuntimeTrackingFixExperiment(
        output_dir=tmp_path / "tracking_fix_experiment",
        base_config=_base_config(),
        logger=type("Logger", (), {"info": lambda *args, **kwargs: None})(),
        run_threshold_experiment=True,
        run_activation_experiment=True,
    )
    experiment.observe(
        camera_id="CAM_001",
        frame_number=0,
        timestamp_seconds=0.0,
        source_fps=30.0,
        detections=sv.Detections.empty(),
    )

    result = experiment.finalize()

    output_dir = Path(result["output_directory"])
    assert (output_dir / "runtime_input_validation.json").exists()
    assert (output_dir / "threshold_comparison.csv").exists()
    assert (output_dir / "activation_comparison.csv").exists()
    assert (output_dir / "final_candidate.json").exists()
    validation = json.loads((output_dir / "runtime_input_validation.json").read_text(encoding="utf-8"))
    assert validation["frame_counts_by_camera"] == {"CAM_001": 1}
    assert validation["empty_frame_counts_by_camera"] == {"CAM_001": 1}
    assert validation["production_unaffected"] is True


def test_supervision_bytetrack_low_conf_and_det_thresh_behavior() -> None:
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=40,
        minimum_matching_threshold=0.6,
        frame_rate=30.0,
        minimum_consecutive_frames=3,
    )

    assert tracker.det_thresh == 0.35
    assert tracker.max_time_lost == 40
