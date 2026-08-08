from __future__ import annotations

import base64
import copy
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _deepcopy_jsonish(value: Any) -> Any:
    return copy.deepcopy(value)


class RuntimeStateManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._current_run_id: str | None = None
        self._current_run_directory: str | None = None
        self._pipeline_status: str = "idle"
        self._camera_state: dict[str, dict[str, Any]] = {}
        self._camera_frame_bytes: dict[str, bytes] = {}
        self._track_state: dict[str, dict[str, Any]] = {}
        self._system_status: dict[str, Any] = {
            "pipeline_status": "idle",
            "camera_count": 0,
            "processing_camera_count": 0,
            "online_camera_count": 0,
            "processed_fps": 0.0,
            "yolo_status": "idle",
            "colour_worker_status": "idle",
            "colour_queue_depth": 0,
            "colour_queue_capacity": 0,
            "pending_colour_jobs": 0,
            "cache_misses": 0,
            "frame_loss": 0,
            "order_violations": 0,
            "last_update": _utc_now_iso(),
        }
        self._subscribers: list[queue.Queue[dict[str, Any]]] = []

    def reset(self) -> None:
        with self._lock:
            self._current_run_id = None
            self._current_run_directory = None
            self._pipeline_status = "idle"
            self._camera_state.clear()
            self._camera_frame_bytes.clear()
            self._track_state.clear()
            self._system_status = {
                **self._system_status,
                "pipeline_status": "idle",
                "camera_count": 0,
                "processing_camera_count": 0,
                "online_camera_count": 0,
                "processed_fps": 0.0,
                "yolo_status": "idle",
                "colour_worker_status": "idle",
                "colour_queue_depth": 0,
                "colour_queue_capacity": 0,
                "pending_colour_jobs": 0,
                "cache_misses": 0,
                "frame_loss": 0,
                "order_violations": 0,
                "last_update": _utc_now_iso(),
            }

    def initialize_run(self, *, run_id: str, run_directory: str, cameras: list[dict[str, Any]]) -> None:
        with self._lock:
            self._current_run_id = str(run_id)
            self._current_run_directory = str(run_directory)
            self._pipeline_status = "running"
            self._camera_state.clear()
            self._camera_frame_bytes.clear()
            self._track_state.clear()
            for camera in cameras:
                camera_id = str(camera.get("camera_id", "")).strip()
                if not camera_id:
                    continue
                self._camera_state[camera_id] = {
                    "camera_id": camera_id,
                    "name": camera_id,
                    "status": "initialized",
                    "frame_number": None,
                    "timestamp_seconds": None,
                    "processed_fps": 0.0,
                    "input_fps": None,
                    "active_vehicle_count": 0,
                    "active_track_ids": [],
                    "detections": [],
                    "source_type": camera.get("source_type"),
                    "source": camera.get("source"),
                    "last_update": _utc_now_iso(),
                    "_frame_times": [],
                }
            self._system_status.update(
                {
                    "pipeline_status": "running",
                    "camera_count": len(self._camera_state),
                    "processing_camera_count": 0,
                    "online_camera_count": len(self._camera_state),
                    "last_update": _utc_now_iso(),
                }
            )
        self.publish_event(
            {
                "type": "run_initialized",
                "run_id": run_id,
                "run_directory": run_directory,
                "camera_count": len(cameras),
            }
        )

    def set_pipeline_status(self, status: str, **extra: Any) -> None:
        with self._lock:
            self._pipeline_status = str(status)
            self._system_status["pipeline_status"] = str(status)
            self._system_status["last_update"] = _utc_now_iso()
            self._system_status.update(extra)
        self.publish_event({"type": "pipeline_status", "status": status, **extra})

    def subscribe(self, *, maxsize: int = 200) -> queue.Queue[dict[str, Any]]:
        subscriber: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._subscribers = [item for item in self._subscribers if item is not subscriber]

    def publish_event(self, event: dict[str, Any]) -> None:
        payload = _deepcopy_jsonish(event)
        payload.setdefault("emitted_at", _utc_now_iso())
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(payload)
            except queue.Full:
                try:
                    subscriber.get_nowait()
                except queue.Empty:
                    pass
                try:
                    subscriber.put_nowait(payload)
                except queue.Full:
                    continue

    def update_camera_runtime(
        self,
        *,
        camera_id: str,
        frame_number: int,
        timestamp_seconds: float,
        input_fps: float | None,
        detections: list[dict[str, Any]],
        active_track_ids: list[str],
        active_vehicle_count: int,
        frame_bgr: np.ndarray | None = None,
        status: str = "processing",
    ) -> None:
        event_payload: dict[str, Any]
        with self._lock:
            camera = self._camera_state.setdefault(
                camera_id,
                {
                    "camera_id": camera_id,
                    "name": camera_id,
                    "status": status,
                    "frame_number": None,
                    "timestamp_seconds": None,
                    "processed_fps": 0.0,
                    "input_fps": input_fps,
                    "active_vehicle_count": 0,
                    "active_track_ids": [],
                    "detections": [],
                    "source_type": None,
                    "source": None,
                    "last_update": _utc_now_iso(),
                    "_frame_times": [],
                },
            )
            camera["status"] = status
            camera["frame_number"] = int(frame_number)
            camera["timestamp_seconds"] = float(timestamp_seconds)
            camera["input_fps"] = None if input_fps is None else float(input_fps)
            camera["detections"] = _deepcopy_jsonish(detections)
            camera["active_track_ids"] = list(active_track_ids)
            camera["active_vehicle_count"] = int(active_vehicle_count)
            frame_times = list(camera.get("_frame_times", []))
            frame_times.append(time.perf_counter())
            if len(frame_times) > 32:
                frame_times = frame_times[-32:]
            camera["_frame_times"] = frame_times
            if len(frame_times) >= 2:
                elapsed = frame_times[-1] - frame_times[0]
                if elapsed > 0.0:
                    camera["processed_fps"] = float((len(frame_times) - 1) / elapsed)
            camera["last_update"] = _utc_now_iso()
            if frame_bgr is not None:
                encoded = self._encode_jpeg(frame_bgr)
                if encoded is not None:
                    self._camera_frame_bytes[camera_id] = encoded
            processing_count = len([item for item in self._camera_state.values() if item.get("status") == "processing"])
            self._system_status["processing_camera_count"] = processing_count
            self._system_status["online_camera_count"] = len(self._camera_state)
            processed_fps_values = [
                float(item.get("processed_fps", 0.0) or 0.0)
                for item in self._camera_state.values()
                if float(item.get("processed_fps", 0.0) or 0.0) > 0.0
            ]
            self._system_status["processed_fps"] = float(sum(processed_fps_values)) if processed_fps_values else 0.0
            self._system_status["last_update"] = _utc_now_iso()
            event_payload = {
                "type": "camera_update",
                "camera_id": camera_id,
                "frame_number": int(frame_number),
                "timestamp_seconds": float(timestamp_seconds),
                "processed_fps": float(camera.get("processed_fps", 0.0) or 0.0),
                "active_vehicle_count": int(active_vehicle_count),
                "status": status,
            }
        self.publish_event(event_payload)
        self.publish_event(
            {
                "type": "detections",
                "camera_id": camera_id,
                "frame_number": int(frame_number),
                "detections": _deepcopy_jsonish(detections),
            }
        )

    def update_track_runtime(
        self,
        *,
        camera_id: str,
        local_track_id: str,
        short_track_id: str,
        vehicle_class: str | None,
        bbox: list[float] | tuple[float, float, float, float] | None,
        confidence: float | None,
        timestamp_seconds: float | None,
        frame_number: int | None,
        colour: str | None,
        colour_status: str,
        status: str,
        evidence: list[dict[str, Any]] | None = None,
    ) -> None:
        with self._lock:
            record = self._track_state.get(local_track_id, {})
            record.update(
                {
                    "local_track_id": local_track_id,
                    "camera_id": camera_id,
                    "track_id": short_track_id,
                    "vehicle_class": vehicle_class,
                    "bbox": list(bbox) if bbox is not None else None,
                    "confidence": confidence,
                    "last_seen": timestamp_seconds,
                    "frame_number": frame_number,
                    "colour": colour,
                    "colour_status": colour_status,
                    "status": status,
                    "last_update": _utc_now_iso(),
                }
            )
            if "first_seen" not in record and timestamp_seconds is not None:
                record["first_seen"] = timestamp_seconds
            if evidence is not None:
                record["evidence"] = _deepcopy_jsonish(evidence)
            self._track_state[local_track_id] = record
        self.publish_event(
            {
                "type": "track_update",
                "camera_id": camera_id,
                "track_id": short_track_id,
                "local_track_id": local_track_id,
                "vehicle_class": vehicle_class,
                "colour": colour,
                "colour_status": colour_status,
                "status": status,
            }
        )

    def update_track_colour(
        self,
        *,
        camera_id: str,
        local_track_id: str,
        short_track_id: str,
        colour: str | None,
        colour_status: str,
    ) -> None:
        with self._lock:
            record = self._track_state.setdefault(
                local_track_id,
                {
                    "local_track_id": local_track_id,
                    "camera_id": camera_id,
                    "track_id": short_track_id,
                    "status": "completed",
                    "last_update": _utc_now_iso(),
                },
            )
            record["colour"] = colour
            record["colour_status"] = colour_status
            record["last_update"] = _utc_now_iso()
        self.publish_event(
            {
                "type": "track_colour_update",
                "camera_id": camera_id,
                "track_id": short_track_id,
                "local_track_id": local_track_id,
                "colour": colour,
                "colour_status": colour_status,
            }
        )

    def update_system_status(self, **metrics: Any) -> None:
        with self._lock:
            self._system_status.update(metrics)
            self._system_status["last_update"] = _utc_now_iso()
            payload = {"type": "system_status", **_deepcopy_jsonish(self._system_status)}
        self.publish_event(payload)

    def mark_run_completed(self, *, status: str, summary: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._pipeline_status = status.lower()
            self._system_status["pipeline_status"] = self._pipeline_status
            self._system_status["last_update"] = _utc_now_iso()
            for camera in self._camera_state.values():
                if camera.get("status") == "processing":
                    camera["status"] = "completed" if status.upper() == "COMPLETED" else "stopped"
                    camera["last_update"] = _utc_now_iso()
        self.publish_event({"type": "run_completed", "status": status, "summary": _deepcopy_jsonish(summary or {})})

    def get_current_run(self) -> dict[str, Any]:
        with self._lock:
            return {
                "run_id": self._current_run_id,
                "run_directory": self._current_run_directory,
                "pipeline_status": self._pipeline_status,
            }

    def list_cameras(self) -> list[dict[str, Any]]:
        with self._lock:
            cameras = []
            for item in self._camera_state.values():
                payload = {key: value for key, value in item.items() if not key.startswith("_")}
                cameras.append(_deepcopy_jsonish(payload))
            cameras.sort(key=lambda item: item["camera_id"])
            return cameras

    def get_camera(self, camera_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._camera_state.get(camera_id)
            if item is None:
                return None
            return _deepcopy_jsonish({key: value for key, value in item.items() if not key.startswith("_")})

    def get_frame_bytes(self, camera_id: str) -> bytes | None:
        with self._lock:
            payload = self._camera_frame_bytes.get(camera_id)
            if payload is None:
                return None
            return bytes(payload)

    def list_tracks(self) -> list[dict[str, Any]]:
        with self._lock:
            items = [_deepcopy_jsonish(item) for item in self._track_state.values()]
        items.sort(key=lambda item: (item.get("camera_id", ""), str(item.get("track_id", ""))))
        return items

    def get_track(self, local_track_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._track_state.get(local_track_id)
            return None if item is None else _deepcopy_jsonish(item)

    def get_system_status(self) -> dict[str, Any]:
        with self._lock:
            return _deepcopy_jsonish(self._system_status)

    def _encode_jpeg(self, frame_bgr: np.ndarray) -> bytes | None:
        try:
            ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        except Exception:
            return None
        if not ok:
            return None
        return bytes(encoded.tobytes())

    def get_frame_data_url(self, camera_id: str) -> str | None:
        frame_bytes = self.get_frame_bytes(camera_id)
        if frame_bytes is None:
            return None
        return f"data:image/jpeg;base64,{base64.b64encode(frame_bytes).decode('ascii')}"


_RUNTIME_STATE_MANAGER = RuntimeStateManager()


def get_runtime_state_manager() -> RuntimeStateManager:
    return _RUNTIME_STATE_MANAGER
