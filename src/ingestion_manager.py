from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .camera_reader import VideoCameraReader
from .models import ConfigurationError, FramePacket, VideoOpenError


class MultiCameraIngestionManager:
    def __init__(self, config: dict[str, Any], logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.input_config = dict(config.get("input", {}) or {})
        self.ingestion_config = dict(config.get("ingestion", {}) or {})
        self.enabled_cameras = self._validate_and_select_enabled_cameras()
        self.worker_count = int(self.ingestion_config.get("worker_count", 7))
        self.frame_queue_size = int(self.ingestion_config.get("frame_queue_size", 200))
        self.queue_put_timeout_seconds = float(self.ingestion_config.get("queue_put_timeout_seconds", 2.0))
        self.queue_get_timeout_seconds = float(self.ingestion_config.get("queue_get_timeout_seconds", 1.0))
        self.target_read_fps = self.ingestion_config.get("target_read_fps")
        self.stop_on_camera_error = bool(self.ingestion_config.get("stop_on_camera_error", False))
        self.round_robin = bool(self.ingestion_config.get("round_robin", True))
        if self.worker_count < 1:
            raise ConfigurationError("ingestion.worker_count must be at least 1.")
        if self.frame_queue_size < 1:
            raise ConfigurationError("ingestion.frame_queue_size must be at least 1.")
        _missing = object()
        raw_max_frames_per_camera = self.input_config.get("max_frames_per_camera", _missing)
        if raw_max_frames_per_camera is None:
            self.max_frames_per_camera: int | None = None
        else:
            default_value = 0 if raw_max_frames_per_camera is _missing else raw_max_frames_per_camera
            try:
                self.max_frames_per_camera = int(default_value)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError("input.max_frames_per_camera must be a positive integer or null.") from exc
            if self.max_frames_per_camera <= 0:
                raise ConfigurationError("input.max_frames_per_camera must be a positive integer or null.")

        self.frame_queue: queue.Queue[FramePacket] = queue.Queue(maxsize=self.frame_queue_size)
        self.readers_by_camera = {
            camera["camera_id"]: VideoCameraReader(
                camera["camera_id"],
                camera["source_type"],
                camera["source"],
                target_read_fps=self.target_read_fps,
            )
            for camera in self.enabled_cameras
        }
        self.camera_assignments = self._build_camera_assignments()
        self.worker_threads = [
            threading.Thread(target=self._worker_loop, args=(worker_id,), name=f"ingestion-worker-{worker_id}")
            for worker_id in range(self.worker_count)
        ]
        self._stop_event = threading.Event()
        self._metrics_lock = threading.Lock()
        self._active_workers = 0
        self._workers_finished = 0
        self._completed_cameras: set[str] = set()
        self._failed_cameras: set[str] = set()
        self.metrics: dict[str, Any] = {
            "configured_camera_count": len(self.input_config.get("cameras", []) or []),
            "enabled_camera_count": len(self.enabled_cameras),
            "worker_count": self.worker_count,
            "camera_assignments": {str(worker_id): list(camera_ids) for worker_id, camera_ids in self.camera_assignments.items()},
            "frames_by_camera": {camera["camera_id"]: 0 for camera in self.enabled_cameras},
            "frames_by_worker": {str(worker_id): 0 for worker_id in range(self.worker_count)},
            "saved_raw_frames_by_camera": {camera["camera_id"]: 0 for camera in self.enabled_cameras},
            "camera_errors": {},
            "queue_full_events": 0,
            "maximum_observed_queue_size": 0,
            "started_at": None,
            "completed_at": None,
            "duration_seconds": 0.0,
        }

    def start(self) -> None:
        started_at = datetime.now(timezone.utc).isoformat()
        self.metrics["started_at"] = started_at
        self.logger.info(
            "Starting ingestion manager enabled_cameras=%s worker_count=%s frame_queue_size=%s frame_limit=%s",
            len(self.enabled_cameras),
            self.worker_count,
            self.frame_queue_size,
            "unlimited" if self.max_frames_per_camera is None else self.max_frames_per_camera,
        )
        for worker_id, camera_ids in self.camera_assignments.items():
            self.logger.info("Worker assignment worker=%s cameras=%s", worker_id, list(camera_ids))
        for thread in self.worker_threads:
            thread.start()

    def get_packet(self, timeout: float | None = None) -> FramePacket:
        effective_timeout = self.queue_get_timeout_seconds if timeout is None else timeout
        return self.frame_queue.get(timeout=effective_timeout)

    def mark_task_done(self) -> None:
        self.frame_queue.task_done()

    def is_finished(self) -> bool:
        with self._metrics_lock:
            all_cameras_done = len(self._completed_cameras) + len(self._failed_cameras) == len(self.enabled_cameras)
            return all_cameras_done and self._workers_finished == self.worker_count

    def stop(self) -> None:
        self._stop_event.set()
        for thread in self.worker_threads:
            thread.join(timeout=10.0)
        completed_at = datetime.now(timezone.utc).isoformat()
        self.metrics["completed_at"] = completed_at
        started_at = self.metrics.get("started_at")
        if isinstance(started_at, str):
            started = datetime.fromisoformat(started_at)
            completed = datetime.fromisoformat(completed_at)
            self.metrics["duration_seconds"] = max(0.0, (completed - started).total_seconds())

    def all_workers_stopped(self) -> bool:
        return all(not thread.is_alive() for thread in self.worker_threads)

    def set_saved_raw_frames_by_camera(self, counts: dict[str, int]) -> None:
        with self._metrics_lock:
            self.metrics["saved_raw_frames_by_camera"] = dict(counts)

    def get_metrics(self) -> dict[str, Any]:
        with self._metrics_lock:
            return {
                **self.metrics,
                "frames_by_camera": dict(self.metrics["frames_by_camera"]),
                "frames_by_worker": dict(self.metrics["frames_by_worker"]),
                "saved_raw_frames_by_camera": dict(self.metrics["saved_raw_frames_by_camera"]),
                "camera_errors": dict(self.metrics["camera_errors"]),
                "camera_assignments": {key: list(value) for key, value in self.metrics["camera_assignments"].items()},
            }

    def _validate_and_select_enabled_cameras(self) -> list[dict[str, Any]]:
        cameras = self.input_config.get("cameras")
        if not isinstance(cameras, list):
            raise ConfigurationError("input.cameras must be a list.")
        enabled_cameras: list[dict[str, Any]] = []
        seen_camera_ids: set[str] = set()
        for camera in cameras:
            if not isinstance(camera, dict):
                raise ConfigurationError("Each camera entry must be a mapping.")
            camera_id = str(camera.get("camera_id", "")).strip()
            source_type = str(camera.get("source_type", "")).strip().lower()
            source = camera.get("source")
            enabled = bool(camera.get("enabled", False))
            if not camera_id:
                raise ConfigurationError("camera_id is required for every camera.")
            if camera_id in seen_camera_ids:
                raise ConfigurationError(f"Duplicate camera_id found: {camera_id}")
            seen_camera_ids.add(camera_id)
            if source_type not in {"video", "rtsp", "webcam"}:
                raise ConfigurationError(f"Unsupported source type for camera '{camera_id}': {source_type or '<empty>'}")
            if source in (None, ""):
                raise ConfigurationError(f"source is required for camera '{camera_id}'.")
            normalized_source: str | int
            if source_type == "webcam":
                try:
                    normalized_source = int(source)
                except Exception as exc:
                    raise ConfigurationError(f"Webcam source must be an integer for camera '{camera_id}'.") from exc
            else:
                normalized_source = str(source).strip()
                if not normalized_source:
                    raise ConfigurationError(f"source is required for camera '{camera_id}'.")
                if source_type == "video":
                    video_path = Path(normalized_source).expanduser().resolve()
                    normalized_source = str(video_path)
            normalized_camera = {
                "camera_id": camera_id,
                "source_type": source_type,
                "source": normalized_source,
                "enabled": enabled,
            }
            if enabled:
                enabled_cameras.append(normalized_camera)
        if not enabled_cameras:
            raise ConfigurationError("At least one enabled camera is required.")
        return enabled_cameras

    def _build_camera_assignments(self) -> dict[int, list[str]]:
        assignments: dict[int, list[str]] = {worker_id: [] for worker_id in range(self.worker_count)}
        for index, camera in enumerate(self.enabled_cameras):
            worker_id = index % self.worker_count
            assignments[worker_id].append(camera["camera_id"])
        return assignments

    def _worker_loop(self, worker_id: int) -> None:
        assigned_camera_ids = list(self.camera_assignments.get(worker_id, []))
        with self._metrics_lock:
            self._active_workers += 1
        try:
            if not assigned_camera_ids:
                self.logger.info("Worker started worker=%s cameras=[]", worker_id)
                return
            self.logger.info("Worker started worker=%s cameras=%s", worker_id, assigned_camera_ids)
            active_camera_ids = list(assigned_camera_ids)
            while active_camera_ids and not self._stop_event.is_set():
                next_active: list[str] = []
                for camera_id in active_camera_ids:
                    if self._stop_event.is_set():
                        break
                    reader = self.readers_by_camera[camera_id]
                    try:
                        packet = reader.read_next_frame(worker_id=worker_id)
                    except Exception as exc:
                        self._handle_camera_error(worker_id, camera_id, reader, exc)
                        if self.stop_on_camera_error:
                            self._stop_event.set()
                            break
                        continue
                    if packet is None or (
                        self.max_frames_per_camera is not None
                        and self.max_frames_per_camera > 0
                        and packet.frame_number >= self.max_frames_per_camera
                    ):
                        self._mark_camera_completed(camera_id)
                        continue
                    if not self._enqueue_packet(packet):
                        break
                    next_active.append(camera_id)
                active_camera_ids = next_active
        finally:
            for camera_id in assigned_camera_ids:
                self.readers_by_camera[camera_id].close()
            with self._metrics_lock:
                self._workers_finished += 1
            self.logger.info("Worker stopped worker=%s", worker_id)

    def _enqueue_packet(self, packet: FramePacket) -> bool:
        while not self._stop_event.is_set():
            try:
                self.frame_queue.put(packet, timeout=self.queue_put_timeout_seconds)
                queue_size = self.frame_queue.qsize()
                with self._metrics_lock:
                    self.metrics["frames_by_camera"][packet.camera_id] += 1
                    self.metrics["frames_by_worker"][str(packet.worker_id)] += 1
                    self.metrics["maximum_observed_queue_size"] = max(
                        int(self.metrics["maximum_observed_queue_size"]),
                        queue_size,
                    )
                return True
            except queue.Full:
                with self._metrics_lock:
                    self.metrics["queue_full_events"] += 1
                self.logger.warning(
                    "Frame queue full worker=%s camera=%s frame=%s queue_size=%s",
                    packet.worker_id,
                    packet.camera_id,
                    packet.frame_number,
                    self.frame_queue.qsize(),
                )
        return False

    def _mark_camera_completed(self, camera_id: str) -> None:
        with self._metrics_lock:
            self._completed_cameras.add(camera_id)
        self.logger.info("Camera completed camera=%s", camera_id)

    def _handle_camera_error(self, worker_id: int, camera_id: str, reader: VideoCameraReader, exc: Exception) -> None:
        self.logger.error(
            "Camera error worker=%s camera=%s source_type=%s source=%s error=%s",
            worker_id,
            camera_id,
            reader.source_type,
            reader.source_display,
            exc,
        )
        with self._metrics_lock:
            self._failed_cameras.add(camera_id)
            self.metrics["camera_errors"][camera_id] = {
                "worker_id": worker_id,
                "source_type": reader.source_type,
                "source": reader.source_display,
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
            }
