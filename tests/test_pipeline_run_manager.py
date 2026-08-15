from __future__ import annotations

import queue
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.config_service import ConfigService, ConfigServiceError
from src.pipeline_run_manager import PipelineRunConflictError, PipelineRunInvalidStateError, PipelineRunManager


class FakeStdout:
    def __init__(self, lines: list[str], delay: float = 0.0) -> None:
        self.lines = lines
        self.delay = delay

    def __iter__(self):
        for line in self.lines:
            if self.delay:
                time.sleep(self.delay)
            yield line


class FakeProcess:
    _pid = 2000

    def __init__(self, lines: list[str], exit_code: int = 0, delay: float = 0.0) -> None:
        FakeProcess._pid += 1
        self.pid = FakeProcess._pid
        self.stdout = FakeStdout(lines, delay)
        self._exit_code = exit_code
        self._terminated = False

    def wait(self, timeout: float | None = None) -> int:
        if self._terminated:
            return -15
        return self._exit_code

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        self._terminated = True

    def kill(self) -> None:
        self._terminated = True


def _write_config(project_root: Path, *, valid: bool = True) -> None:
    config_dir = project_root / "config"
    config_dir.mkdir()
    model_path = config_dir / "model.pt"
    model_path.write_bytes(b"model")
    config = {
        "project": {"name": "test"},
        "input": {"cameras": [{"camera_id": "CAM_001", "source_type": "video", "source": str(config_dir / "video.mp4"), "enabled": True}], "max_frames_per_camera": None},
        "ingestion": {"worker_count": 1, "frame_queue_size": 2, "per_camera_buffer_size": 1, "scheduler_policy": "round_robin"},
        "detection": {"model_path": str(model_path), "confidence_threshold": 1.5 if not valid else 0.2, "iou_threshold": 0.45, "image_size": 640},
        "tracking": {"track_activation_threshold": 0.25, "lost_track_buffer": 150, "minimum_matching_threshold": 0.7, "minimum_consecutive_frames": 3},
        "tracking_roi": {"enabled": True, "mode": "rectangle", "rectangle": {"x_min_fraction": 0.0, "y_min_fraction": 0.4, "x_max_fraction": 1.0, "y_max_fraction": 0.75}, "anchor": "bottom_center"},
        "visualization": {},
        "output": {"root_directory": "outputs/runs", "save_run_config": True},
    }
    (config_dir / "validation_rectangle_roi.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (project_root / ".venv" / "Scripts").mkdir(parents=True)
    (project_root / ".venv" / "Scripts" / "python.exe").write_bytes(b"python")


def _wait_for_terminal(manager: PipelineRunManager, job_id: str) -> Any:
    for _ in range(100):
        job = manager.get_job(job_id)
        if job.status in {"COMPLETED", "FAILED", "CANCELLED"}:
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_pipeline_run_manager_completes_and_extracts_run_id(tmp_path: Path) -> None:
    _write_config(tmp_path)
    calls: queue.Queue[list[str]] = queue.Queue()

    def factory(command: list[str], **kwargs: Any) -> FakeProcess:
        calls.put(command)
        return FakeProcess(
            [
                "2026-08-15 | INFO | pipeline | Pipeline started\n",
                "2026-08-15 | INFO | pipeline | Run summary run_id=20260815_155243 processed_frames=6241 completed_tracks=125\n",
                "Run completed: 20260815_155243\n",
                f"Output: {tmp_path / 'outputs' / 'runs' / '20260815_155243'}\n",
            ],
            exit_code=0,
        )

    manager = PipelineRunManager(project_root=tmp_path, config_service=ConfigService(tmp_path / "config"), jobs_root=tmp_path / "run_jobs", process_factory=factory)
    job = manager.create_run("validation_rectangle_roi.yaml")
    finished = _wait_for_terminal(manager, job.job_id)

    assert finished.status == "COMPLETED"
    assert finished.run_id == "20260815_155243"
    assert finished.processed_frames == 6241
    command = calls.get_nowait()
    assert command[0].endswith("python.exe")
    assert command[-2:] == ["--config", "config\\validation_rectangle_roi.yaml"] or command[-2:] == ["--config", "config/validation_rectangle_roi.yaml"]
    assert "Run completed: 20260815_155243" in "\n".join(manager.get_logs(job.job_id)["lines"])


def test_pipeline_run_manager_rejects_invalid_config(tmp_path: Path) -> None:
    _write_config(tmp_path, valid=False)
    manager = PipelineRunManager(project_root=tmp_path, config_service=ConfigService(tmp_path / "config"), jobs_root=tmp_path / "run_jobs", process_factory=lambda *args, **kwargs: FakeProcess([]))

    with pytest.raises(ConfigServiceError):
        manager.create_run("validation_rectangle_roi.yaml")


def test_pipeline_run_manager_prevents_second_concurrent_run_and_cancels(tmp_path: Path) -> None:
    _write_config(tmp_path)

    def factory(command: list[str], **kwargs: Any) -> FakeProcess:
        return FakeProcess(["Pipeline started\n", "still running\n"], exit_code=0, delay=0.1)

    manager = PipelineRunManager(project_root=tmp_path, config_service=ConfigService(tmp_path / "config"), jobs_root=tmp_path / "run_jobs", process_factory=factory)
    job = manager.create_run("validation_rectangle_roi.yaml")
    with pytest.raises(PipelineRunConflictError):
        manager.create_run("validation_rectangle_roi.yaml")

    cancelled = manager.cancel_job(job.job_id)
    assert cancelled.status == "CANCELLED"
    _wait_for_terminal(manager, job.job_id)
    with pytest.raises(PipelineRunInvalidStateError):
        manager.cancel_job(job.job_id)


def test_pipeline_run_manager_marks_active_persisted_job_failed_after_restart(tmp_path: Path) -> None:
    _write_config(tmp_path)
    first = PipelineRunManager(project_root=tmp_path, config_service=ConfigService(tmp_path / "config"), jobs_root=tmp_path / "run_jobs", process_factory=lambda *args, **kwargs: FakeProcess(["Pipeline started\n", "still running\n", "still running\n"], delay=0.2))
    job = first.create_run("validation_rectangle_roi.yaml")
    job_file = tmp_path / "run_jobs" / job.job_id / "job.json"
    for _ in range(50):
        if job_file.exists():
            break
        time.sleep(0.02)
    assert job_file.exists()

    restarted = PipelineRunManager(project_root=tmp_path, config_service=ConfigService(tmp_path / "config"), jobs_root=tmp_path / "run_jobs", process_factory=lambda *args, **kwargs: FakeProcess([]))
    loaded = restarted.get_job(job.job_id)

    assert loaded.status == "FAILED"
    assert "cannot be safely reattached" in str(loaded.error_message)
