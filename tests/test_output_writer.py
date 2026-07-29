from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.models import RUN_STATUS_CREATED, RunMetadata
from src.output_writer import RunOutputManager


def test_run_directory_is_created_and_runs_do_not_overwrite(tmp_path: Path) -> None:
    manager1 = RunOutputManager(tmp_path)
    manager2 = RunOutputManager(tmp_path)
    assert manager1.run_directory.exists()
    assert manager2.run_directory.exists()
    assert manager1.run_directory != manager2.run_directory


def test_effective_config_metadata_summary_and_subdirectories_are_saved(tmp_path: Path) -> None:
    manager = RunOutputManager(tmp_path)
    config = {"project": {"name": "demo"}}
    metadata = RunMetadata(
        run_id=manager.run_id,
        project_name="demo",
        started_at="2026-07-29T00:00:00+00:00",
        completed_at=None,
        status=RUN_STATUS_CREATED,
        camera_count=1,
        processed_frames=0,
        completed_tracks=0,
        error_count=0,
        config_path="config.yaml",
    )
    summary = {"status": "CREATED"}
    manager.save_effective_config(config)
    manager.save_metadata(metadata)
    manager.save_summary(summary)
    manager.save_ingestion_metrics({"worker_count": 7})
    manager.save_detection_tracking_metrics({"tracker_instance_count": 2})
    manager.save_bbox_quality_metrics({"raw_detections": 1, "accepted_detections": 1, "rejected_detections": 0})
    assert (manager.run_directory / "run_config.yaml").exists()
    assert (manager.run_directory / "run_metadata.json").exists()
    assert (manager.run_directory / "summary.json").exists()
    assert (manager.run_directory / "ingestion_metrics.json").exists()
    assert (manager.run_directory / "detection_tracking_metrics.json").exists()
    assert (manager.run_directory / "bbox_quality_metrics.json").exists()
    assert manager.evidence_directory.exists()
    assert manager.errors_directory.exists()
    assert manager.raw_frames_directory.exists()
    assert manager.detected_frames_directory.exists()
    assert manager.tracked_frames_directory.exists()
    assert (manager.detected_frames_directory / "README.txt").exists()
    saved_config = yaml.safe_load((manager.run_directory / "run_config.yaml").read_text(encoding="utf-8"))
    saved_metadata = json.loads((manager.run_directory / "run_metadata.json").read_text(encoding="utf-8"))
    saved_summary = json.loads((manager.run_directory / "summary.json").read_text(encoding="utf-8"))
    assert saved_config["project"]["name"] == "demo"
    assert saved_metadata["status"] == RUN_STATUS_CREATED
    assert saved_summary["status"] == "CREATED"
