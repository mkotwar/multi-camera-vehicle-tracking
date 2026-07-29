from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .models import FramePacket, RunMetadata


class RunOutputManager:
    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.run_id = self._generate_run_id()
        self.run_directory = self._create_run_directory(self.run_id)
        self.evidence_directory = self.run_directory / "evidence"
        self.errors_directory = self.run_directory / "errors"
        self.raw_frames_directory = self.run_directory / "raw_frames"
        self.detected_frames_directory = self.run_directory / "detected_frames"
        self.tracked_frames_directory = self.run_directory / "tracked_frames"
        self.evidence_directory.mkdir(parents=True, exist_ok=True)
        self.errors_directory.mkdir(parents=True, exist_ok=True)
        self.raw_frames_directory.mkdir(parents=True, exist_ok=True)
        self.detected_frames_directory.mkdir(parents=True, exist_ok=True)
        self.tracked_frames_directory.mkdir(parents=True, exist_ok=True)
        self._write_detected_frames_note()

    def _generate_run_id(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _create_run_directory(self, base_run_id: str) -> Path:
        candidate = self.output_root / base_run_id
        suffix = 1
        while candidate.exists():
            candidate = self.output_root / f"{base_run_id}_{suffix:02d}"
            suffix += 1
        candidate.mkdir(parents=True, exist_ok=False)
        self.run_id = candidate.name
        return candidate

    def _write_detected_frames_note(self) -> None:
        note_path = self.detected_frames_directory / "README.txt"
        note_path.write_text(
            "raw_frames contains frames directly from ingestion.\n"
            "detected_frames contains YOLO-annotated frames.\n"
            "tracked_frames contains ByteTrack-native tracking annotations.\n",
            encoding="utf-8",
        )

    def save_effective_config(self, config: dict[str, Any]) -> Path:
        path = self.run_directory / "run_config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path

    def save_metadata(self, metadata: RunMetadata) -> Path:
        path = self.run_directory / "run_metadata.json"
        path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
        return path

    def save_summary(self, summary: dict[str, Any]) -> Path:
        path = self.run_directory / "summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return path

    def save_ingestion_metrics(self, metrics: dict[str, Any]) -> Path:
        path = self.run_directory / "ingestion_metrics.json"
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return path

    def save_detection_tracking_metrics(self, metrics: dict[str, Any]) -> Path:
        path = self.run_directory / "detection_tracking_metrics.json"
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return path

    def save_bbox_quality_metrics(self, metrics: dict[str, Any]) -> Path:
        path = self.run_directory / "bbox_quality_metrics.json"
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return path

    def save_error(self, error_name: str, payload: dict[str, Any]) -> Path:
        path = self.errors_directory / f"{error_name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def save_raw_frame(
        self,
        packet: FramePacket,
        *,
        image_format: str,
        jpeg_quality: int,
    ) -> Path:
        camera_directory = self.raw_frames_directory / packet.camera_id
        camera_directory.mkdir(parents=True, exist_ok=True)
        frame_path = camera_directory / f"frame_{packet.frame_number:06d}.{image_format}"
        params: list[int] = []
        if image_format.lower() in {"jpg", "jpeg"}:
            params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        cv2.imwrite(str(frame_path), packet.frame, params)
        return frame_path

    def save_detected_frame(self, camera_id: str, frame_number: int, frame: np.ndarray) -> Path:
        camera_directory = self.detected_frames_directory / camera_id
        camera_directory.mkdir(parents=True, exist_ok=True)
        frame_path = camera_directory / f"frame_{frame_number:06d}.jpg"
        cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return frame_path

    def save_tracked_frame(self, camera_id: str, frame_number: int, frame: np.ndarray) -> Path:
        camera_directory = self.tracked_frames_directory / camera_id
        camera_directory.mkdir(parents=True, exist_ok=True)
        frame_path = camera_directory / f"frame_{frame_number:06d}.jpg"
        cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return frame_path

    def future_output_path(self, *parts: str) -> Path:
        return self.run_directory.joinpath(*parts)
