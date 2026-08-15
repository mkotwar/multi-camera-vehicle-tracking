from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from src.api_app import create_app
from src.pipeline_run_manager import PipelineRunConflictError, PipelineRunInvalidStateError, PipelineRunJob, PipelineRunJobNotFoundError


class FakePipelineRunManager:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.job = PipelineRunJob(
            job_id="JOB_TEST",
            config_name="validation_rectangle_roi.yaml",
            status="RUNNING",
            created_at="2026-08-15T10:00:00+00:00",
            started_at="2026-08-15T10:00:01+00:00",
            pid=1234,
            current_stage="DETECTION_TRACKING",
            processed_frames=12,
            log_file="outputs/run_jobs/JOB_TEST/pipeline.log",
        )
        self.conflict = False

    def create_run(self, config_name: str) -> PipelineRunJob:
        if self.conflict:
            raise PipelineRunConflictError("Pipeline job already active: JOB_TEST")
        return self.job

    def list_jobs(self, limit: int = 25) -> list[PipelineRunJob]:
        return [self.job]

    def get_job(self, job_id: str) -> PipelineRunJob:
        if job_id != self.job.job_id:
            raise PipelineRunJobNotFoundError(job_id)
        return self.job

    def get_logs(self, job_id: str, limit: int = 200) -> dict[str, Any]:
        self.get_job(job_id)
        return {"job_id": job_id, "log_file": str(self.job.log_file), "lines": ["Pipeline started"], "limit": limit}

    def cancel_job(self, job_id: str) -> PipelineRunJob:
        self.get_job(job_id)
        if self.job.status == "COMPLETED":
            raise PipelineRunInvalidStateError("Cannot cancel job with status COMPLETED.")
        self.job.status = "CANCELLED"
        self.job.current_stage = "CANCELLED"
        return self.job

    def launch_summary(self, config_name: str) -> dict[str, Any]:
        return {
            "config_name": config_name,
            "valid": True,
            "errors": [],
            "input_sources": [{"camera_id": "CAM_001", "source_type": "video", "source": "video.mp4"}],
            "tracking_roi": {"enabled": True, "mode": "rectangle", "rectangle": {}, "anchor": "bottom_center"},
            "plate_ocr_enabled": False,
            "plate_detector_enabled": False,
            "physical_identity_enabled": True,
            "stationary_recovery_enabled": False,
            "db_import_after_run": False,
        }


def test_pipeline_run_api_start_status_logs_and_cancel(monkeypatch, tmp_path: Path) -> None:
    fake = FakePipelineRunManager()
    monkeypatch.setattr("src.api_app.PipelineRunManager", lambda *args, **kwargs: fake)
    client = TestClient(create_app(outputs_root=tmp_path / "runs", config_dir=tmp_path / "config"))

    start_response = client.post("/api/pipeline-runs", json={"config_name": "validation_rectangle_roi.yaml"})
    assert start_response.status_code == 200
    assert start_response.json()["job_id"] == "JOB_TEST"
    assert start_response.json()["status"] == "RUNNING"

    list_response = client.get("/api/pipeline-runs")
    assert list_response.status_code == 200
    assert list_response.json()[0]["processed_frames"] == 12

    status_response = client.get("/api/pipeline-runs/JOB_TEST")
    assert status_response.status_code == 200
    assert status_response.json()["current_stage"] == "DETECTION_TRACKING"

    logs_response = client.get("/api/pipeline-runs/JOB_TEST/logs", params={"limit": 20})
    assert logs_response.status_code == 200
    assert logs_response.json()["lines"] == ["Pipeline started"]

    summary_response = client.get("/api/pipeline-runs/launch-summary/validation_rectangle_roi.yaml")
    assert summary_response.status_code == 200
    assert summary_response.json()["physical_identity_enabled"] is True

    cancel_response = client.post("/api/pipeline-runs/JOB_TEST/cancel")
    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] == "CANCELLED"


def test_pipeline_run_api_conflict_not_found_and_invalid_cancel(monkeypatch, tmp_path: Path) -> None:
    fake = FakePipelineRunManager()
    fake.conflict = True
    monkeypatch.setattr("src.api_app.PipelineRunManager", lambda *args, **kwargs: fake)
    client = TestClient(create_app(outputs_root=tmp_path / "runs", config_dir=tmp_path / "config"))

    conflict_response = client.post("/api/pipeline-runs", json={"config_name": "validation_rectangle_roi.yaml"})
    assert conflict_response.status_code == 409

    missing_response = client.get("/api/pipeline-runs/NOPE")
    assert missing_response.status_code == 404

    fake.job.status = "COMPLETED"
    invalid_cancel = client.post("/api/pipeline-runs/JOB_TEST/cancel")
    assert invalid_cancel.status_code == 409


def test_pipeline_run_api_db_auto_import_runtime_setting(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DB_IMPORT_AFTER_RUN", raising=False)
    fake = FakePipelineRunManager()
    monkeypatch.setattr("src.api_app.PipelineRunManager", lambda *args, **kwargs: fake)
    env_path = tmp_path / ".env"
    env_path.write_text("DATABASE_URL=postgresql://example\nDB_IMPORT_AFTER_RUN=false\n", encoding="utf-8")
    client = TestClient(create_app(outputs_root=tmp_path / "runs", config_dir=tmp_path / "config", env_path=env_path))

    initial = client.get("/api/runtime-settings/db-auto-import")
    assert initial.status_code == 200
    assert initial.json()["enabled"] is False

    updated = client.put("/api/runtime-settings/db-auto-import", json={"enabled": True})
    assert updated.status_code == 200
    assert updated.json()["enabled"] is True
    assert "DB_IMPORT_AFTER_RUN=true" in env_path.read_text(encoding="utf-8")
