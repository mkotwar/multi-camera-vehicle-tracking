from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .camera_reader import VideoCameraReader
from .models import ConfigurationError, FramePacket


class MultiCameraIngestionManager:
    def __init__(self, config: dict[str, Any], logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.input_config = dict(config.get("input", {}) or {})
        self.ingestion_config = dict(config.get("ingestion", {}) or {})
        self.enabled_cameras = self._validate_and_select_enabled_cameras()
        self.worker_count = int(self.ingestion_config.get("worker_count", 7))
        self.frame_queue_size = int(self.ingestion_config.get("frame_queue_size", 200))
        self.per_camera_buffer_size = int(self.ingestion_config.get("per_camera_buffer_size", 2))
        self.scheduler_policy = str(self.ingestion_config.get("scheduler_policy", "round_robin")).strip().lower() or "round_robin"
        self.queue_put_timeout_seconds = float(self.ingestion_config.get("queue_put_timeout_seconds", 2.0))
        self.queue_get_timeout_seconds = float(self.ingestion_config.get("queue_get_timeout_seconds", 1.0))
        self.target_read_fps = self.ingestion_config.get("target_read_fps")
        self.stop_on_camera_error = bool(self.ingestion_config.get("stop_on_camera_error", False))
        self.round_robin = bool(self.ingestion_config.get("round_robin", True))
        if self.worker_count < 1:
            raise ConfigurationError("ingestion.worker_count must be at least 1.")
        if self.frame_queue_size < 1:
            raise ConfigurationError("ingestion.frame_queue_size must be at least 1.")
        if self.per_camera_buffer_size < 1:
            raise ConfigurationError("ingestion.per_camera_buffer_size must be at least 1.")
        if self.scheduler_policy != "round_robin":
            raise ConfigurationError("ingestion.scheduler_policy must be round_robin.")
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
        self.camera_sources = dict(self.readers_by_camera)
        self.per_camera_buffers: dict[str, queue.Queue[FramePacket]] = {
            camera["camera_id"]: queue.Queue(maxsize=self.per_camera_buffer_size)
            for camera in self.enabled_cameras
        }
        self.camera_order = [camera["camera_id"] for camera in self.enabled_cameras]
        self.camera_task_queue: queue.Queue[str] = queue.Queue(maxsize=max(1, len(self.enabled_cameras)))
        self.worker_threads = [
            threading.Thread(target=self._worker_loop, args=(worker_id,), name=f"ingestion-worker-{worker_id}")
            for worker_id in range(self.worker_count)
        ]
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, name="ingestion-scheduler")
        self._stop_event = threading.Event()
        self._metrics_lock = threading.Lock()
        self._scheduler_condition = threading.Condition(self._metrics_lock)
        self._workers_finished = 0
        self._scheduler_finished = False
        self._completed_cameras: set[str] = set()
        self._failed_cameras: set[str] = set()
        self._camera_state: dict[str, dict[str, bool]] = {
            camera_id: {
                "completed": False,
                "failed": False,
                "task_enqueued": False,
                "read_in_flight": False,
            }
            for camera_id in self.camera_order
        }
        self._scheduler_index = 0
        self._last_scheduled_camera_id: str | None = None
        self._current_same_camera_streak = 0
        self._queue_full_log_interval_seconds = 10.0
        self._queue_full_warning_state: dict[str, dict[str, Any]] = {}
        self.metrics: dict[str, Any] = {
            "configured_camera_count": len(self.input_config.get("cameras", []) or []),
            "enabled_camera_count": len(self.enabled_cameras),
            "worker_count": self.worker_count,
            "ingestion_worker_count": self.worker_count,
            "camera_count": len(self.enabled_cameras),
            "camera_source_registry_count": len(self.enabled_cameras),
            "camera_assignment_mode": "dynamic_task_queue",
            "scheduler_policy": self.scheduler_policy,
            "per_camera_buffer_count": len(self.enabled_cameras),
            "per_camera_buffer_size": self.per_camera_buffer_size,
            "frames_by_camera": {camera["camera_id"]: 0 for camera in self.enabled_cameras},
            "frames_scheduled_by_camera": {camera["camera_id"]: 0 for camera in self.enabled_cameras},
            "frames_consumed_by_camera": {camera["camera_id"]: 0 for camera in self.enabled_cameras},
            "frames_by_worker": {str(worker_id): 0 for worker_id in range(self.worker_count)},
            "saved_raw_frames_by_camera": {camera["camera_id"]: 0 for camera in self.enabled_cameras},
            "camera_errors": {},
            "camera_read_jobs": 0,
            "camera_read_failures": 0,
            "round_robin_cycles": 0,
            "scheduler_skipped_empty_camera": 0,
            "queue_full_events": 0,
            "maximum_observed_queue_size": 0,
            "per_camera_buffer_peak": {camera["camera_id"]: 0 for camera in self.enabled_cameras},
            "buffer_full_count": 0,
            "buffer_full_count_by_camera": {camera["camera_id"]: 0 for camera in self.enabled_cameras},
            "max_consecutive_frames_same_camera": 0,
            "started_at": None,
            "completed_at": None,
            "duration_seconds": 0.0,
        }

    def start(self) -> None:
        started_at = datetime.now(timezone.utc).isoformat()
        self.metrics["started_at"] = started_at
        self.logger.info(
            "Starting ingestion manager enabled_cameras=%s worker_count=%s frame_queue_size=%s per_camera_buffer_size=%s frame_limit=%s",
            len(self.enabled_cameras),
            self.worker_count,
            self.frame_queue_size,
            self.per_camera_buffer_size,
            "unlimited" if self.max_frames_per_camera is None else self.max_frames_per_camera,
        )
        self.logger.info(
            "Ingestion scheduler policy=%s camera_registry=%s",
            self.scheduler_policy,
            self.camera_order,
        )
        for thread in self.worker_threads:
            thread.start()
        self.scheduler_thread.start()
        with self._scheduler_condition:
            for camera_id in self.camera_order:
                self._schedule_read_if_possible(camera_id)
            self._scheduler_condition.notify_all()

    def get_packet(self, timeout: float | None = None) -> FramePacket:
        effective_timeout = self.queue_get_timeout_seconds if timeout is None else timeout
        packet = self.frame_queue.get(timeout=effective_timeout)
        with self._metrics_lock:
            self.metrics["frames_consumed_by_camera"][packet.camera_id] += 1
        return packet

    def mark_task_done(self) -> None:
        self.frame_queue.task_done()

    def is_finished(self) -> bool:
        with self._metrics_lock:
            all_cameras_done = len(self._completed_cameras) + len(self._failed_cameras) == len(self.enabled_cameras)
            buffers_empty = all(buffer.empty() for buffer in self.per_camera_buffers.values())
            return all_cameras_done and buffers_empty and self._workers_finished == self.worker_count and self._scheduler_finished

    def stop(self) -> None:
        self._stop_event.set()
        with self._scheduler_condition:
            self._scheduler_condition.notify_all()
        if self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=10.0)
        for thread in self.worker_threads:
            thread.join(timeout=10.0)
        for reader in self.readers_by_camera.values():
            reader.close()
        completed_at = datetime.now(timezone.utc).isoformat()
        self.metrics["completed_at"] = completed_at
        started_at = self.metrics.get("started_at")
        if isinstance(started_at, str):
            started = datetime.fromisoformat(started_at)
            completed = datetime.fromisoformat(completed_at)
            self.metrics["duration_seconds"] = max(0.0, (completed - started).total_seconds())
        self._log_unrecovered_queue_full_events()

    def all_workers_stopped(self) -> bool:
        return all(not thread.is_alive() for thread in self.worker_threads) and not self.scheduler_thread.is_alive()

    def set_saved_raw_frames_by_camera(self, counts: dict[str, int]) -> None:
        with self._metrics_lock:
            self.metrics["saved_raw_frames_by_camera"] = dict(counts)

    def get_metrics(self) -> dict[str, Any]:
        with self._metrics_lock:
            return {
                **self.metrics,
                "frames_by_camera": dict(self.metrics["frames_by_camera"]),
                "frames_scheduled_by_camera": dict(self.metrics["frames_scheduled_by_camera"]),
                "frames_consumed_by_camera": dict(self.metrics["frames_consumed_by_camera"]),
                "frames_by_worker": dict(self.metrics["frames_by_worker"]),
                "saved_raw_frames_by_camera": dict(self.metrics["saved_raw_frames_by_camera"]),
                "camera_errors": dict(self.metrics["camera_errors"]),
                "per_camera_buffer_peak": dict(self.metrics["per_camera_buffer_peak"]),
                "buffer_full_count_by_camera": dict(self.metrics["buffer_full_count_by_camera"]),
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

    def _schedule_read_if_possible(self, camera_id: str) -> None:
        state = self._camera_state[camera_id]
        if state["completed"] or state["failed"] or state["task_enqueued"] or state["read_in_flight"]:
            return
        buffer = self.per_camera_buffers[camera_id]
        if buffer.full():
            self.metrics["buffer_full_count"] += 1
            self.metrics["buffer_full_count_by_camera"][camera_id] += 1
            return
        self.camera_task_queue.put_nowait(camera_id)
        state["task_enqueued"] = True

    def _worker_loop(self, worker_id: int) -> None:
        self.logger.info("Worker started worker=%s", worker_id)
        try:
            while not self._stop_event.is_set():
                try:
                    camera_id = self.camera_task_queue.get(timeout=0.1)
                except queue.Empty:
                    if self._all_camera_reads_finished():
                        break
                    continue
                try:
                    with self._scheduler_condition:
                        state = self._camera_state[camera_id]
                        state["task_enqueued"] = False
                        if state["completed"] or state["failed"] or state["read_in_flight"]:
                            continue
                        state["read_in_flight"] = True
                    reader = self.readers_by_camera[camera_id]
                    with self._metrics_lock:
                        self.metrics["camera_read_jobs"] += 1
                    try:
                        packet = reader.read_next_frame(worker_id=worker_id)
                    except Exception as exc:
                        self._handle_camera_error(worker_id, camera_id, reader, exc)
                        if self.stop_on_camera_error:
                            self._stop_event.set()
                        continue
                    if packet is None or (
                        self.max_frames_per_camera is not None
                        and self.max_frames_per_camera > 0
                        and packet.frame_number >= self.max_frames_per_camera
                    ):
                        self._mark_camera_completed(camera_id)
                        continue
                    buffer = self.per_camera_buffers[camera_id]
                    buffer.put(packet)
                    with self._scheduler_condition:
                        self.metrics["frames_by_camera"][camera_id] += 1
                        self.metrics["frames_by_worker"][str(worker_id)] += 1
                        self.metrics["per_camera_buffer_peak"][camera_id] = max(
                            int(self.metrics["per_camera_buffer_peak"][camera_id]),
                            buffer.qsize(),
                        )
                        self._camera_state[camera_id]["read_in_flight"] = False
                        self._scheduler_condition.notify_all()
                finally:
                    with self._scheduler_condition:
                        if camera_id in self._camera_state:
                            self._camera_state[camera_id]["read_in_flight"] = False
                            self._scheduler_condition.notify_all()
                    self.camera_task_queue.task_done()
        finally:
            with self._metrics_lock:
                self._workers_finished += 1
            self.logger.info("Worker stopped worker=%s", worker_id)

    def _scheduler_loop(self) -> None:
        self.logger.info("Scheduler started policy=%s", self.scheduler_policy)
        try:
            while not self._stop_event.is_set():
                packet = self._next_scheduled_packet()
                if packet is None:
                    if self._all_cameras_done_and_buffers_empty():
                        break
                    with self._scheduler_condition:
                        self._scheduler_condition.wait(timeout=0.05)
                    continue
                if not self._enqueue_packet(packet):
                    break
        finally:
            with self._metrics_lock:
                self._scheduler_finished = True
            self.logger.info("Scheduler stopped")

    def _next_scheduled_packet(self) -> FramePacket | None:
        with self._scheduler_condition:
            if not self.camera_order:
                return None
            camera_count = len(self.camera_order)
            for offset in range(camera_count):
                index = (self._scheduler_index + offset) % camera_count
                camera_id = self.camera_order[index]
                buffer = self.per_camera_buffers[camera_id]
                if buffer.empty():
                    state = self._camera_state[camera_id]
                    if not state["completed"] and not state["failed"]:
                        self.metrics["scheduler_skipped_empty_camera"] += 1
                    continue
                packet = buffer.get_nowait()
                self.metrics["frames_scheduled_by_camera"][camera_id] += 1
                self.metrics["round_robin_cycles"] += 1
                if self._last_scheduled_camera_id == camera_id:
                    self._current_same_camera_streak += 1
                else:
                    self._last_scheduled_camera_id = camera_id
                    self._current_same_camera_streak = 1
                self.metrics["max_consecutive_frames_same_camera"] = max(
                    int(self.metrics["max_consecutive_frames_same_camera"]),
                    self._current_same_camera_streak,
                )
                self._scheduler_index = (index + 1) % camera_count
                self._schedule_read_if_possible(camera_id)
                return packet
            return None

    def _enqueue_packet(self, packet: FramePacket) -> bool:
        while not self._stop_event.is_set():
            try:
                self.frame_queue.put(packet, timeout=self.queue_put_timeout_seconds)
                queue_size = self.frame_queue.qsize()
                with self._metrics_lock:
                    self.metrics["maximum_observed_queue_size"] = max(
                        int(self.metrics["maximum_observed_queue_size"]),
                        queue_size,
                    )
                self._record_queue_recovered(packet.camera_id)
                return True
            except queue.Full:
                with self._metrics_lock:
                    self.metrics["queue_full_events"] += 1
                    queue_full_events = int(self.metrics["queue_full_events"])
                self._record_queue_full(packet, queue_full_events)
        return False

    def _record_queue_full(self, packet: FramePacket, queue_full_events: int) -> None:
        now = time.monotonic()
        queue_size = self.frame_queue.qsize()
        with self._metrics_lock:
            state = self._queue_full_warning_state.setdefault(
                packet.camera_id,
                {
                    "active": False,
                    "started_at": now,
                    "last_logged_at": 0.0,
                    "events": 0,
                    "suppressed": 0,
                    "last_queue_size": queue_size,
                },
            )
            if not bool(state["active"]):
                state["active"] = True
                state["started_at"] = now
                state["events"] = 0
                state["suppressed"] = 0
            state["events"] = int(state["events"]) + 1
            state["last_queue_size"] = queue_size
            should_log = float(state["last_logged_at"]) == 0.0 or now - float(state["last_logged_at"]) >= self._queue_full_log_interval_seconds
            if should_log:
                suppressed = int(state["suppressed"])
                state["suppressed"] = 0
                state["last_logged_at"] = now
            else:
                state["suppressed"] = int(state["suppressed"]) + 1
                suppressed = 0
        if should_log:
            self.logger.warning(
                "Frame queue full camera=%s queue_size=%s queue_full_events=%s suppressed_since_last=%s",
                packet.camera_id,
                queue_size,
                queue_full_events,
                suppressed,
            )

    def _record_queue_recovered(self, camera_id: str) -> None:
        now = time.monotonic()
        with self._metrics_lock:
            state = self._queue_full_warning_state.get(camera_id)
            if not state or not bool(state["active"]):
                return
            events = int(state["events"])
            suppressed = int(state["suppressed"])
            stall_seconds = max(0.0, now - float(state["started_at"]))
            state["active"] = False
            state["events"] = 0
            state["suppressed"] = 0
            state["last_logged_at"] = 0.0
        self.logger.info(
            "Frame queue recovered camera=%s queue_full_events=%s suppressed_warnings=%s stall_seconds=%.1f",
            camera_id,
            events,
            suppressed,
            stall_seconds,
        )

    def _log_unrecovered_queue_full_events(self) -> None:
        summaries: list[tuple[str, int, int, float]] = []
        now = time.monotonic()
        with self._metrics_lock:
            for camera_id, state in self._queue_full_warning_state.items():
                if not bool(state.get("active")):
                    continue
                summaries.append(
                    (
                        camera_id,
                        int(state.get("events", 0)),
                        int(state.get("suppressed", 0)),
                        max(0.0, now - float(state.get("started_at", now))),
                    )
                )
                state["active"] = False
        for camera_id, events, suppressed, stall_seconds in summaries:
            self.logger.info(
                "Frame queue stopped while full camera=%s queue_full_events=%s suppressed_warnings=%s stall_seconds=%.1f",
                camera_id,
                events,
                suppressed,
                stall_seconds,
            )

    def _mark_camera_completed(self, camera_id: str) -> None:
        with self._scheduler_condition:
            self._camera_state[camera_id]["completed"] = True
            self._camera_state[camera_id]["task_enqueued"] = False
            self._camera_state[camera_id]["read_in_flight"] = False
            self._completed_cameras.add(camera_id)
            self._scheduler_condition.notify_all()
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
        with self._scheduler_condition:
            self._camera_state[camera_id]["failed"] = True
            self._camera_state[camera_id]["task_enqueued"] = False
            self._camera_state[camera_id]["read_in_flight"] = False
            self._failed_cameras.add(camera_id)
            self.metrics["camera_read_failures"] += 1
            self.metrics["camera_errors"][camera_id] = {
                "worker_id": worker_id,
                "source_type": reader.source_type,
                "source": reader.source_display,
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
            }
            self._scheduler_condition.notify_all()

    def _all_cameras_done_and_buffers_empty(self) -> bool:
        with self._metrics_lock:
            all_cameras_done = len(self._completed_cameras) + len(self._failed_cameras) == len(self.enabled_cameras)
            return all_cameras_done and all(buffer.empty() for buffer in self.per_camera_buffers.values())

    def _all_camera_reads_finished(self) -> bool:
        with self._metrics_lock:
            if len(self._completed_cameras) + len(self._failed_cameras) != len(self.enabled_cameras):
                return False
            if not self.camera_task_queue.empty():
                return False
            return all(not state["task_enqueued"] and not state["read_in_flight"] for state in self._camera_state.values())
