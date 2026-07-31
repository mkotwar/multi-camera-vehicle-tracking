from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
    manager.save_tracks([{"local_track_id": "CAM_001:TRACK_1"}])
    manager.save_observations(
        [
            {
                "local_track_id": "CAM_001:TRACK_1",
                "camera_id": "CAM_001",
                "tracker_namespace": "camera",
                "native_tracker_id": 1,
                "frame_number": 0,
                "timestamp_seconds": 0.0,
                "x1": 1.0,
                "y1": 2.0,
                "x2": 3.0,
                "y2": 4.0,
                "confidence": 0.8,
                "raw_class_id": 0,
                "raw_class_name": "car",
            }
        ]
    )
    manager.save_track_lifecycle_metrics({"active_tracks_at_shutdown": 0})
    manager.save_evidence_index([{"local_track_id": "CAM_001:TRACK_1", "role": "FIRST"}])
    manager.save_evidence_metrics({"tracks_with_evidence": 1})
    manager.save_vehicle_enrichment([{"local_track_id": "CAM_001:TRACK_1", "status": "disabled"}])
    manager.save_vehicle_enrichment_metrics({"completed_tracks_received": 1})
    manager.save_track_evidence("CAM_001", "CAM_001_TRACK_1", [{"local_track_id": "CAM_001:TRACK_1", "role": "FIRST"}])
    manager.save_vehicle_enrichment_crop("CAM_001:TRACK_1", 0, np.zeros((8, 8, 3), dtype=np.uint8))
    assert (manager.run_directory / "run_config.yaml").exists()
    assert (manager.run_directory / "run_metadata.json").exists()
    assert (manager.run_directory / "summary.json").exists()
    assert (manager.run_directory / "ingestion_metrics.json").exists()
    assert (manager.run_directory / "detection_tracking_metrics.json").exists()
    assert (manager.run_directory / "bbox_quality_metrics.json").exists()
    assert (manager.run_directory / "tracks.json").exists()
    assert (manager.run_directory / "observations.csv").exists()
    assert (manager.run_directory / "track_lifecycle_metrics.json").exists()
    assert (manager.run_directory / "evidence_index.json").exists()
    assert (manager.run_directory / "evidence_metrics.json").exists()
    assert (manager.run_directory / "vehicle_enrichment.json").exists()
    assert (manager.run_directory / "vehicle_enrichment_metrics.json").exists()
    assert (manager.evidence_directory / "CAM_001" / "CAM_001_TRACK_1" / "evidence.json").exists()
    assert any(manager.vehicle_enrichment_crops_directory.rglob("frame_000000.jpg"))
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
