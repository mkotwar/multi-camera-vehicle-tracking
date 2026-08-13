from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.bytetrack_association_experiment import (
    FrozenFrame,
    assert_frame_order,
    inspect_threshold_semantics,
    read_frozen_detections,
)


def test_frozen_detection_round_trip_preserves_values(tmp_path: Path) -> None:
    frame = FrozenFrame(
        camera_id="CAM_001",
        frame_number=7,
        timestamp_seconds=0.25,
        source_fps=29.9707,
        xyxy=[[1.25, 2.5, 30.75, 40.0]],
        confidence=[0.456],
        class_id=[4],
    )
    path = tmp_path / "frozen.jsonl"
    path.write_text(json.dumps(frame.to_json()) + "\n", encoding="utf-8")

    loaded = read_frozen_detections(path)[0]

    assert loaded.camera_id == frame.camera_id
    assert loaded.frame_number == frame.frame_number
    assert loaded.xyxy == frame.xyxy
    assert loaded.confidence == frame.confidence
    assert loaded.class_id == frame.class_id


def test_frame_order_must_be_strictly_increasing() -> None:
    frames = [
        FrozenFrame("CAM_001", 2, 0.0, 30.0, [], [], []),
        FrozenFrame("CAM_001", 2, 0.0, 30.0, [], [], []),
    ]

    with pytest.raises(ValueError):
        assert_frame_order(frames)


def test_to_detections_preserves_all_detections() -> None:
    frame = FrozenFrame(
        camera_id="CAM_001",
        frame_number=1,
        timestamp_seconds=0.0,
        source_fps=30.0,
        xyxy=[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]],
        confidence=[0.2, 0.9],
        class_id=[2, 4],
    )

    detections = frame.to_detections()

    assert len(detections.xyxy) == 2
    assert detections.confidence.tolist() == pytest.approx([0.2, 0.9])
    assert detections.class_id.tolist() == [2, 4]


def test_threshold_semantics_are_documented_from_installed_code() -> None:
    semantics = inspect_threshold_semantics()

    assert semantics["source_excerpt_present"] is True
    assert "Larger values are more permissive" in semantics["interpretation"]
