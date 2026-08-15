from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, IO

from .config_service import ConfigService, ConfigServiceError, ConfigValidationError
from .env_loader import merged_project_env
from .env_flags import db_import_after_run_enabled


ACTIVE_STATUSES = {"QUEUED", "STARTING", "RUNNING", "CANCEL_REQUESTED"}
TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}


class PipelineRunConflictError(Exception):
    pass


class PipelineRunJobNotFoundError(Exception):
    pass


class PipelineRunInvalidStateError(Exception):
    pass


@dataclass(slots=True)
class PipelineRunJob:
    job_id: str
    config_name: str
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    pid: int | None = None
    exit_code: int | None = None
    run_id: str | None = None
    run_directory: str | None = None
    current_stage: str = "STARTING"
    processed_frames: int | None = None
    error_message: str | None = None
    log_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["elapsed_seconds"] = self.elapsed_seconds()
        return payload

    def elapsed_seconds(self) -> float | None:
        started = _parse_iso(self.started_at)
        if started is None:
            return None
        end = _parse_iso(self.finished_at) or datetime.now(timezone.utc)
        return max(0.0, (end - started).total_seconds())


ProcessFactory = Callable[..., subprocess.Popen[str]]


class PipelineRunManager:
    def __init__(
        self,
        *,
        project_root: str | Path = ".",
        config_service: ConfigService | None = None,
        jobs_root: str | Path = "outputs/run_jobs",
        process_factory: ProcessFactory | None = None,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.config_service = config_service or ConfigService(self.project_root / "config")
        self.jobs_root = (self.project_root / jobs_root).resolve() if not Path(jobs_root).is_absolute() else Path(jobs_root).resolve()
        self._process_factory = process_factory or subprocess.Popen
        self._lock = threading.RLock()
        self._jobs: dict[str, PipelineRunJob] = {}
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._load_jobs()

    def create_run(self, config_name: str) -> PipelineRunJob:
        with self._lock:
            active = self._active_job_locked()
            if active is not None:
                raise PipelineRunConflictError(f"Pipeline job already active: {active.job_id}")
            detail = self.config_service.load_config(config_name)
            validation = detail["validation"]
            if not validation["valid"]:
                errors = [
                    ConfigValidationError(
                        rule=str(error.get("rule") or "config.invalid"),
                        path=str(error.get("path") or "config"),
                        message=str(error.get("message") or "Invalid config."),
                        expected=error.get("expected"),
                        actual=error.get("actual"),
                    )
                    for error in validation["errors"]
                ]
                raise ConfigServiceError("Configuration validation failed.", status_code=422, errors=errors)
            job_id = _new_job_id()
            job_dir = self.jobs_root / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            job = PipelineRunJob(
                job_id=job_id,
                config_name=str(detail["config_name"]),
                status="STARTING",
                created_at=_utc_now_iso(),
                log_file=str(job_dir / "pipeline.log"),
            )
            self._jobs[job_id] = job
            self._persist_job(job)
            thread = threading.Thread(target=self._run_job_supervisor, args=(job.job_id,), daemon=True)
            thread.start()
            return job

    def list_jobs(self, *, limit: int = 25) -> list[PipelineRunJob]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            return jobs[: max(1, min(int(limit), 100))]

    def get_job(self, job_id: str) -> PipelineRunJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise PipelineRunJobNotFoundError(job_id)
            self._refresh_process_state_locked(job)
            return job

    def get_logs(self, job_id: str, *, limit: int = 200) -> dict[str, Any]:
        job = self.get_job(job_id)
        log_path = Path(str(job.log_file or ""))
        lines = _tail_lines(log_path, max(1, min(int(limit), 1000)))
        return {"job_id": job_id, "log_file": str(log_path), "lines": lines, "limit": limit}

    def cancel_job(self, job_id: str) -> PipelineRunJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise PipelineRunJobNotFoundError(job_id)
            if job.status in TERMINAL_STATUSES:
                raise PipelineRunInvalidStateError(f"Cannot cancel job with status {job.status}.")
            job.status = "CANCEL_REQUESTED"
            job.current_stage = "CANCEL_REQUESTED"
            self._persist_job(job)
            process = self._processes.get(job_id)
        if process is not None and process.poll() is None:
            self._terminate_process_tree(process)
        with self._lock:
            job = self._jobs[job_id]
            if job.status != "CANCELLED":
                job.status = "CANCELLED"
                job.current_stage = "CANCELLED"
                job.finished_at = job.finished_at or _utc_now_iso()
                job.exit_code = job.exit_code if job.exit_code is not None else -signal.SIGTERM
                self._persist_job(job)
            return job

    def launch_summary(self, config_name: str) -> dict[str, Any]:
        detail = self.config_service.load_config(config_name)
        config = detail["config"]
        cameras = list(dict(config.get("input", {}) or {}).get("cameras", []) or [])
        enabled = [dict(item) for item in cameras if isinstance(item, dict) and bool(item.get("enabled", False))]
        roi = dict(config.get("tracking_roi", {}) or {})
        vehicle_identity = dict(config.get("vehicle_identity", {}) or {})
        plate = _plate_section(config)
        return {
            "config_name": detail["config_name"],
            "valid": bool(detail["validation"]["valid"]),
            "errors": detail["validation"]["errors"],
            "input_sources": [
                {
                    "camera_id": item.get("camera_id"),
                    "source_type": item.get("source_type"),
                    "source": item.get("source"),
                }
                for item in enabled
            ],
            "tracking_roi": {
                "enabled": bool(roi.get("enabled", False)),
                "mode": roi.get("mode"),
                "rectangle": roi.get("rectangle"),
                "anchor": roi.get("anchor"),
            },
            "plate_ocr_enabled": bool(_nested_get(plate, ["ocr", "enabled"]) or plate.get("recognition_enabled")),
            "plate_detector_enabled": bool(_nested_get(plate, ["detector", "enabled"]) or plate.get("detection_enabled")),
            "physical_identity_enabled": bool(vehicle_identity.get("enabled", False)),
            "stationary_recovery_enabled": bool(_nested_get(vehicle_identity, ["stationary_recovery", "enabled"])),
            "db_import_after_run": db_import_after_run_enabled(),
        }

    def _run_job_supervisor(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "STARTING"
            job.started_at = _utc_now_iso()
            job.current_stage = "STARTING"
            self._persist_job(job)
            log_file = Path(str(job.log_file))
        command = [str(self._venv_python()), "app.py", "--config", str(Path("config") / job.config_name)]
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        try:
            with log_file.open("a", encoding="utf-8", errors="replace") as log_handle:
                log_handle.write(f"RUN CONTROL COMMAND: {' '.join(command)}\n")
                log_handle.flush()
                process = self._process_factory(
                    command,
                    cwd=str(self.project_root),
                    env=merged_project_env(env_path=self.project_root / ".env"),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                    creationflags=creationflags,
                )
                cancel_before_running = False
                with self._lock:
                    job = self._jobs[job_id]
                    job.pid = process.pid
                    self._processes[job_id] = process
                    if job.status in {"CANCEL_REQUESTED", "CANCELLED"}:
                        cancel_before_running = True
                    else:
                        job.status = "RUNNING"
                    self._persist_job(job)
                if cancel_before_running:
                    self._terminate_process_tree(process)
                else:
                    self._consume_process_output(job_id, process, log_handle)
                exit_code = process.wait()
            with self._lock:
                job = self._jobs[job_id]
                job.exit_code = int(exit_code)
                job.finished_at = _utc_now_iso()
                if job.status in {"CANCEL_REQUESTED", "CANCELLED"}:
                    job.status = "CANCELLED"
                    job.current_stage = "CANCELLED"
                elif exit_code == 0:
                    job.status = "COMPLETED"
                    job.current_stage = "COMPLETED"
                else:
                    job.status = "FAILED"
                    job.current_stage = "FAILED"
                    job.error_message = job.error_message or _last_error_line(log_file) or f"Pipeline exited with code {exit_code}."
                self._processes.pop(job_id, None)
                self._persist_job(job)
        except Exception as exc:
            with self._lock:
                job = self._jobs[job_id]
                job.status = "FAILED"
                job.current_stage = "FAILED"
                job.finished_at = _utc_now_iso()
                job.error_message = str(exc)
                self._processes.pop(job_id, None)
                self._persist_job(job)

    def _consume_process_output(self, job_id: str, process: subprocess.Popen[str], log_handle: IO[str]) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            log_handle.write(_strip_ansi(line))
            log_handle.flush()
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    continue
                self._apply_log_line(job, line)
                self._persist_job(job)

    def _apply_log_line(self, job: PipelineRunJob, line: str) -> None:
        clean = _strip_ansi(line).strip()
        if not clean:
            return
        run_completed = re.search(r"Run completed:\s*([0-9_]+)", clean)
        if run_completed:
            job.run_id = run_completed.group(1)
            job.run_directory = str((self.project_root / "outputs" / "runs" / job.run_id).resolve())
        output_match = re.search(r"Output:\s*(.+)$", clean)
        if output_match:
            job.run_directory = output_match.group(1).strip()
        summary_match = re.search(r"Run summary run_id=([0-9_]+)\s+processed_frames=(\d+)", clean)
        if summary_match:
            job.run_id = summary_match.group(1)
            job.processed_frames = int(summary_match.group(2))
        processed_match = re.search(r"processed_frames=(\d+)", clean)
        if processed_match:
            job.processed_frames = int(processed_match.group(1))
        stage = _stage_from_log_line(clean)
        if stage:
            job.current_stage = stage
        if "ERROR" in clean or "Traceback" in clean or "Run failed" in clean:
            job.error_message = clean[-500:]

    def _active_job_locked(self) -> PipelineRunJob | None:
        for job in self._jobs.values():
            self._refresh_process_state_locked(job)
            if job.status in ACTIVE_STATUSES:
                return job
        return None

    def _refresh_process_state_locked(self, job: PipelineRunJob) -> None:
        process = self._processes.get(job.job_id)
        if process is not None:
            exit_code = process.poll()
            if exit_code is not None and job.status in ACTIVE_STATUSES:
                job.exit_code = int(exit_code)
                job.finished_at = job.finished_at or _utc_now_iso()
                job.status = "COMPLETED" if exit_code == 0 else "FAILED"
                job.current_stage = job.status
                self._processes.pop(job.job_id, None)
                self._persist_job(job)

    def _venv_python(self) -> Path:
        candidates = [
            self.project_root / ".venv" / "Scripts" / "python.exe",
            self.project_root / ".venv" / "bin" / "python",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError(f"Unable to resolve project virtualenv Python under {self.project_root / '.venv'}.")

    def _terminate_process_tree(self, process: subprocess.Popen[str]) -> None:
        try:
            process.terminate()
            process.wait(timeout=8)
            return
        except Exception:
            pass
        if os.name == "nt" and process.pid:
            try:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False, capture_output=True, text=True)
                process.wait(timeout=5)
                return
            except Exception:
                pass
        try:
            process.kill()
            process.wait(timeout=5)
        except Exception:
            pass

    def _persist_job(self, job: PipelineRunJob) -> None:
        job_dir = self.jobs_root / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = job_dir / f"job.{uuid.uuid4().hex}.json.tmp"
        target = job_dir / "job.json"
        tmp_path.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")
        for attempt in range(10):
            try:
                os.replace(tmp_path, target)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.03)

    def _load_jobs(self) -> None:
        for path in sorted(self.jobs_root.glob("*/job.json")):
            try:
                payload = _read_job_json(path)
                job = PipelineRunJob(
                    job_id=str(payload["job_id"]),
                    config_name=str(payload["config_name"]),
                    status=str(payload["status"]),
                    created_at=str(payload["created_at"]),
                    started_at=payload.get("started_at"),
                    finished_at=payload.get("finished_at"),
                    pid=payload.get("pid"),
                    exit_code=payload.get("exit_code"),
                    run_id=payload.get("run_id"),
                    run_directory=payload.get("run_directory"),
                    current_stage=str(payload.get("current_stage") or "STARTING"),
                    processed_frames=payload.get("processed_frames"),
                    error_message=payload.get("error_message"),
                    log_file=payload.get("log_file"),
                )
                if job.status in ACTIVE_STATUSES:
                    job.status = "FAILED"
                    job.current_stage = "FAILED"
                    job.finished_at = job.finished_at or _utc_now_iso()
                    job.error_message = "API restarted while this job was active; process cannot be safely reattached."
                    self._persist_job(job)
                self._jobs[job.job_id] = job
            except Exception:
                continue


_PIPELINE_RUN_MANAGER: PipelineRunManager | None = None


def get_pipeline_run_manager(
    *,
    project_root: str | Path = ".",
    config_service: ConfigService | None = None,
    jobs_root: str | Path = "outputs/run_jobs",
) -> PipelineRunManager:
    global _PIPELINE_RUN_MANAGER
    if _PIPELINE_RUN_MANAGER is None:
        _PIPELINE_RUN_MANAGER = PipelineRunManager(project_root=project_root, config_service=config_service, jobs_root=jobs_root)
    return _PIPELINE_RUN_MANAGER


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _new_job_id() -> str:
    return f"JOB_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _tail_lines(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-limit:]


def _read_job_json(path: Path) -> dict[str, Any]:
    last_error: Exception | None = None
    for _ in range(10):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (PermissionError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(0.03)
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(path)


def _last_error_line(path: Path) -> str | None:
    for line in reversed(_tail_lines(path, 80)):
        if "ERROR" in line or "Traceback" in line or "Run failed" in line:
            return line
    return None


def _strip_ansi(value: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", value)


def _stage_from_log_line(line: str) -> str | None:
    lower = line.lower()
    if "pipeline started" in lower:
        return "DETECTION_TRACKING"
    if "colour worker started" in lower or "vehicle enrichment" in lower:
        return "ENRICHMENT"
    if "physical vehicle" in lower or "vehicle identity" in lower:
        return "IDENTITY"
    if "post-run postgresql import enabled" in lower or "db import" in lower:
        return "DB_IMPORT"
    if "post-run postgresql import completed" in lower:
        return "PERSISTENCE"
    if "pipeline completed" in lower:
        return "COMPLETED"
    if "pipeline failed" in lower:
        return "FAILED"
    return None


def _plate_section(config: dict[str, Any]) -> dict[str, Any]:
    vehicle_enrichment = dict(config.get("vehicle_enrichment", {}) or {})
    enrichment = dict(vehicle_enrichment.get("enrichment", {}) or {})
    if isinstance(enrichment.get("plate"), dict):
        return dict(enrichment["plate"])
    if isinstance(vehicle_enrichment.get("plate"), dict):
        return dict(vehicle_enrichment["plate"])
    return {}


def _nested_get(payload: dict[str, Any], path: list[str]) -> Any:
    current: Any = payload
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current
