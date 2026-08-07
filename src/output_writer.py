from __future__ import annotations

import csv
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
        self.extracted_frames_directory = self.run_directory / "01_extracted_frames"
        self.yolo_detected_frames_directory = self.run_directory / "02_yolo_detected_frames"
        self.tracked_debug_frames_directory = self.run_directory / "03_tracked_frames"
        self.track_crops_directory = self.run_directory / "04_track_crops"
        self.florence_selected_crops_directory = self.run_directory / "05_florence_selected_crops"
        self.florence_results_directory = self.run_directory / "06_florence_results"
        self.body_type_selected_crops_directory = self.run_directory / "07_body_type_selected_crops"
        self.body_type_results_directory = self.run_directory / "07_body_type_results"
        self.evidence_directory = self.run_directory / "evidence"
        self.errors_directory = self.run_directory / "errors"
        self.raw_frames_directory = self.run_directory / "raw_frames"
        self.detected_frames_directory = self.run_directory / "detected_frames"
        self.tracked_frames_directory = self.run_directory / "tracked_frames"
        self.vehicle_enrichment_directory = self.run_directory / "vehicle_enrichment"
        self.vehicle_enrichment_crops_directory = self.florence_selected_crops_directory
        self.evidence_capture_zone_directory = self.run_directory / "evidence_capture_zone"
        self.extracted_frames_directory.mkdir(parents=True, exist_ok=True)
        self.yolo_detected_frames_directory.mkdir(parents=True, exist_ok=True)
        self.tracked_debug_frames_directory.mkdir(parents=True, exist_ok=True)
        self.track_crops_directory.mkdir(parents=True, exist_ok=True)
        self.florence_selected_crops_directory.mkdir(parents=True, exist_ok=True)
        self.florence_results_directory.mkdir(parents=True, exist_ok=True)
        self.body_type_selected_crops_directory.mkdir(parents=True, exist_ok=True)
        self.body_type_results_directory.mkdir(parents=True, exist_ok=True)
        self.evidence_directory.mkdir(parents=True, exist_ok=True)
        self.errors_directory.mkdir(parents=True, exist_ok=True)
        self.raw_frames_directory.mkdir(parents=True, exist_ok=True)
        self.detected_frames_directory.mkdir(parents=True, exist_ok=True)
        self.tracked_frames_directory.mkdir(parents=True, exist_ok=True)
        self.vehicle_enrichment_directory.mkdir(parents=True, exist_ok=True)
        self.vehicle_enrichment_crops_directory.mkdir(parents=True, exist_ok=True)
        self.evidence_capture_zone_directory.mkdir(parents=True, exist_ok=True)
        self.debug_outputs_config: dict[str, Any] = {
            "enabled": False,
            "extracted_frames": {"enabled": False},
            "detected_frames": {"enabled": False},
            "tracked_frames": {"enabled": False},
            "track_crops": {"enabled": False},
            "florence_selected_crops": {"enabled": False},
        }
        self._write_detected_frames_note()

    def configure_debug_outputs(self, config: dict[str, Any]) -> None:
        self.debug_outputs_config = dict(config or {})

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
        (self.yolo_detected_frames_directory / "README.txt").write_text(
            "02_yolo_detected_frames contains YOLO-annotated frames for debugging.\n",
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

    def save_tracks(self, tracks: list[dict[str, Any]]) -> Path:
        path = self.run_directory / "tracks.json"
        path.write_text(json.dumps(tracks, indent=2), encoding="utf-8")
        return path

    def save_observations(self, observations: list[dict[str, Any]]) -> Path:
        path = self.run_directory / "observations.csv"
        fieldnames = [
            "local_track_id",
            "camera_id",
            "tracker_namespace",
            "native_tracker_id",
            "frame_number",
            "timestamp_seconds",
            "x1",
            "y1",
            "x2",
            "y2",
            "confidence",
            "raw_class_id",
            "raw_class_name",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for item in observations:
                writer.writerow(item)
        return path

    def save_track_lifecycle_metrics(self, metrics: dict[str, Any]) -> Path:
        path = self.run_directory / "track_lifecycle_metrics.json"
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return path

    def save_evidence_index(self, records: list[dict[str, Any]]) -> Path:
        path = self.run_directory / "evidence_index.json"
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return path

    def save_evidence_metrics(self, metrics: dict[str, Any]) -> Path:
        path = self.run_directory / "evidence_metrics.json"
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return path

    def save_capture_zone_index(self, records: list[dict[str, Any]]) -> Path:
        path = self.run_directory / "evidence_capture_zone_index.json"
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return path

    def save_capture_zone_metrics(self, metrics: dict[str, Any]) -> Path:
        path = self.run_directory / "evidence_capture_zone_metrics.json"
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return path

    def save_motorcycle_geometry_report(self, rows: list[dict[str, Any]]) -> Path:
        path = self.run_directory / "motorcycle_geometry_report.csv"
        fieldnames = [
            "camera_id",
            "local_track_id",
            "source_frame_width",
            "source_frame_height",
            "first_frame",
            "last_frame",
            "observation_count",
            "min_trigger_y",
            "max_trigger_y",
            "zone_top",
            "zone_bottom",
            "entered_zone",
            "first_zone_entry_frame",
            "last_zone_frame",
            "zone_exit_frame",
            "max_bbox_width",
            "max_bbox_height",
            "max_bbox_area",
            "frame_of_max_trigger_y",
            "frame_of_max_bbox_width",
            "frame_of_max_bbox_height",
            "frame_of_max_bbox_area",
            "largest_saved_crop_width",
            "largest_saved_crop_height",
            "largest_saved_crop_frame",
            "capture_candidates",
            "retained_candidates",
            "geometry_status",
            "geometry_reason",
            "final_class",
            "stable_class_name",
            "completion_reason",
            "track_status",
            "evidence_eligible_zone_crop",
            "florence_eligible_zone_crop",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
        return path

    def save_track_crop_manifest(self, rows: list[dict[str, Any]]) -> Path:
        path = self.track_crops_directory / "track_crop_manifest.csv"
        fieldnames = [
            "camera_id",
            "local_track_id",
            "frame_number",
            "timestamp_seconds",
            "vehicle_class",
            "confidence",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "crop_width",
            "crop_height",
            "crop_path",
            "trigger_y",
            "inside_capture_zone",
            "capture_zone_top",
            "capture_zone_bottom",
            "evidence_eligible",
            "evidence_rejection_reason",
            "florence_eligible",
            "florence_rejection_reason",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
        return path

    def save_vehicle_pipeline_trace(self, rows: list[dict[str, Any]]) -> Path:
        path = self.run_directory / "vehicle_pipeline_trace.csv"
        fieldnames = [
            "camera_id",
            "local_track_id",
            "vehicle_class",
            "detection_count",
            "tracking_observation_count",
            "raw_track_crop_count",
            "preferred_crop_count",
            "fallback_crop_count",
            "selected_colour_crop_count",
            "colour_selection_tier",
            "capture_zone_entered",
            "capture_zone_candidate_count",
            "capture_zone_retained_count",
            "evidence_candidate_count",
            "evidence_eligible_count",
            "florence_eligible_count",
            "florence_selected_count",
            "florence_call_count",
            "valid_colour_prediction_count",
            "final_colour",
            "body_type_eligible",
            "body_type_candidate_crop_count",
            "body_type_selected_crop_count",
            "body_type_florence_call_count",
            "body_type_valid_prediction_count",
            "body_type_final",
            "body_type_failure_reason",
            "failure_stage",
            "failure_reason",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
        return path

    def save_vehicle_enrichment(self, records: list[dict[str, Any]]) -> Path:
        path = self.run_directory / "vehicle_enrichment.json"
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return path

    def save_vehicle_enrichment_metrics(self, metrics: dict[str, Any]) -> Path:
        path = self.run_directory / "vehicle_enrichment_metrics.json"
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return path

    def save_vehicle_enrichment_validation_report(self, rows: list[dict[str, Any]]) -> Path:
        path = self.run_directory / "vehicle_enrichment_validation_report.csv"
        fieldnames = [
            "camera_id",
            "local_track_id",
            "vehicle_class",
            "crop_path",
            "candidate_crop_count",
            "eligible_crop_count",
            "preferred_crop_count",
            "selected_body_type_crop_paths",
            "selected_colour_crop_paths",
            "classification_trigger",
            "source_frame_width",
            "source_frame_height",
            "original_bbox",
            "expanded_crop_bbox",
            "context_padding_ratio",
            "original_crop_width",
            "original_crop_height",
            "resolution_tier",
            "sharpness",
            "brightness",
            "edge_truncated",
            "quality_score",
            "square_padding_applied",
            "padded_width",
            "padded_height",
            "florence_input_width",
            "florence_input_height",
            "predicted_body_type",
            "body_type_raw_response",
            "body_type_reason",
            "predicted_colour",
            "colour_raw_response",
            "colour_reason",
            "final_body_type",
            "final_colour",
            "final_reason",
            "manual_body_type",
            "manual_colour",
            "body_type_correct",
            "colour_correct",
            "review_notes",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
        return path

    def save_vehicle_enrichment_crop_diagnostics(self, rows: list[dict[str, Any]]) -> Path:
        path = self.run_directory / "vehicle_enrichment_crop_diagnostics.csv"
        fieldnames = [
            "camera_id",
            "local_track_id",
            "evidence_role",
            "frame_index",
            "timestamp",
            "candidate_rank",
            "candidate_retained",
            "candidate_rejection_reason",
            "frame_gap_from_previous_selected",
            "duplicate_score",
            "crop_path",
            "source_frame_width",
            "source_frame_height",
            "original_crop_width",
            "original_crop_height",
            "resolution_tier",
            "selection_tier",
            "sharpness",
            "brightness",
            "quality_score",
            "eligible_for_body_type",
            "eligible_for_colour",
            "body_type_skip_reason",
            "colour_skip_reason",
            "selected_for_body_type",
            "selected_for_colour",
            "body_type_crop_result",
            "colour_crop_result",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
        return path

    def save_vehicle_enrichment_track_evidence_summary(self, rows: list[dict[str, Any]]) -> Path:
        path = self.run_directory / "vehicle_enrichment_track_evidence_summary.csv"
        fieldnames = [
            "camera_id",
            "local_track_id",
            "vehicle_class",
            "track_start_frame",
            "track_end_frame",
            "track_duration_frames",
            "candidate_crops_seen",
            "candidate_crops_retained",
            "acceptable_crops",
            "preferred_crops",
            "selected_body_type_crops",
            "selected_colour_crops",
            "largest_original_crop_width",
            "largest_original_crop_height",
            "best_quality_score",
            "body_type_status",
            "body_type_label",
            "colour_status",
            "colour_label",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
        return path

    def evidence_track_directory(self, camera_id: str, safe_local_track_id: str) -> Path:
        path = self.evidence_directory / camera_id / safe_local_track_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_track_evidence(self, camera_id: str, safe_local_track_id: str, records: list[dict[str, Any]]) -> Path:
        track_directory = self.evidence_track_directory(camera_id, safe_local_track_id)
        path = track_directory / "evidence.json"
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return path

    def capture_zone_track_directory(self, camera_id: str, safe_local_track_id: str) -> Path:
        path = self.evidence_capture_zone_directory / camera_id / safe_local_track_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_capture_zone_track_evidence(self, camera_id: str, safe_local_track_id: str, records: list[dict[str, Any]]) -> Path:
        track_directory = self.capture_zone_track_directory(camera_id, safe_local_track_id)
        path = track_directory / "evidence.json"
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
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
        debug_directory = self.extracted_frames_directory / packet.camera_id
        debug_directory.mkdir(parents=True, exist_ok=True)
        frame_path = camera_directory / f"frame_{packet.frame_number:06d}.{image_format}"
        debug_path = debug_directory / f"frame_{packet.frame_number:06d}.{image_format}"
        params: list[int] = []
        if image_format.lower() in {"jpg", "jpeg"}:
            params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        cv2.imwrite(str(frame_path), packet.frame, params)
        if bool(self.debug_outputs_config.get("enabled", False)) and bool(dict(self.debug_outputs_config.get("extracted_frames", {}) or {}).get("enabled", False)) and debug_path != frame_path:
            cv2.imwrite(str(debug_path), packet.frame, params)
        return frame_path

    def save_detected_frame(self, camera_id: str, frame_number: int, frame: np.ndarray) -> Path:
        camera_directory = self.detected_frames_directory / camera_id
        camera_directory.mkdir(parents=True, exist_ok=True)
        debug_directory = self.yolo_detected_frames_directory / camera_id
        debug_directory.mkdir(parents=True, exist_ok=True)
        frame_path = camera_directory / f"frame_{frame_number:06d}.jpg"
        debug_path = debug_directory / f"frame_{frame_number:06d}.jpg"
        cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if bool(self.debug_outputs_config.get("enabled", False)) and bool(dict(self.debug_outputs_config.get("detected_frames", {}) or {}).get("enabled", False)) and debug_path != frame_path:
            cv2.imwrite(str(debug_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return frame_path

    def save_tracked_frame(self, camera_id: str, frame_number: int, frame: np.ndarray) -> Path:
        camera_directory = self.tracked_frames_directory / camera_id
        camera_directory.mkdir(parents=True, exist_ok=True)
        debug_directory = self.tracked_debug_frames_directory / camera_id
        debug_directory.mkdir(parents=True, exist_ok=True)
        frame_path = camera_directory / f"frame_{frame_number:06d}.jpg"
        debug_path = debug_directory / f"frame_{frame_number:06d}.jpg"
        self._write_image(frame_path, frame, jpeg_quality=90)
        if bool(self.debug_outputs_config.get("enabled", False)) and bool(dict(self.debug_outputs_config.get("tracked_frames", {}) or {}).get("enabled", False)) and debug_path != frame_path:
            self._write_image(debug_path, frame, jpeg_quality=90)
        return frame_path

    def save_evidence_crop(
        self,
        camera_id: str,
        safe_local_track_id: str,
        frame_number: int,
        crop: np.ndarray,
        *,
        jpeg_quality: int,
    ) -> Path:
        track_directory = self.evidence_track_directory(camera_id, safe_local_track_id)
        crop_directory = track_directory / "crops"
        crop_directory.mkdir(parents=True, exist_ok=True)
        path = crop_directory / f"frame_{frame_number:06d}.jpg"
        self._write_image(path, crop, jpeg_quality=jpeg_quality)
        return path

    def save_evidence_annotated_frame(
        self,
        camera_id: str,
        safe_local_track_id: str,
        frame_number: int,
        frame: np.ndarray,
        *,
        jpeg_quality: int,
    ) -> Path:
        track_directory = self.evidence_track_directory(camera_id, safe_local_track_id)
        annotated_directory = track_directory / "annotated_frames"
        annotated_directory.mkdir(parents=True, exist_ok=True)
        path = annotated_directory / f"frame_{frame_number:06d}.jpg"
        self._write_image(path, frame, jpeg_quality=jpeg_quality)
        return path

    def save_capture_zone_crop(
        self,
        camera_id: str,
        safe_local_track_id: str,
        frame_number: int,
        crop: np.ndarray,
        *,
        jpeg_quality: int,
    ) -> Path:
        track_directory = self.capture_zone_track_directory(camera_id, safe_local_track_id)
        path = track_directory / f"frame_{frame_number:06d}.jpg"
        self._write_image(path, crop, jpeg_quality=jpeg_quality)
        return path

    def vehicle_enrichment_track_directory(self, safe_local_track_id: str) -> Path:
        parts = safe_local_track_id.split(":")
        camera_id = parts[0] if parts else "UNKNOWN"
        track_name = parts[-1] if parts else safe_local_track_id.replace(":", "_")
        path = self.vehicle_enrichment_crops_directory / camera_id / track_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def vehicle_enrichment_track_crop_path(self, safe_local_track_id: str, frame_number: int, *, suffix: str | None = None) -> Path:
        directory = self.vehicle_enrichment_track_directory(safe_local_track_id)
        suffix_fragment = f"_{suffix}" if suffix else ""
        return directory / f"frame_{frame_number:06d}{suffix_fragment}.jpg"

    def save_vehicle_enrichment_crop(
        self,
        safe_local_track_id: str,
        frame_number: int,
        crop: np.ndarray,
        *,
        suffix: str | None = None,
        jpeg_quality: int = 90,
    ) -> Path:
        path = self.vehicle_enrichment_track_crop_path(safe_local_track_id, frame_number, suffix=suffix)
        self._write_image(path, crop, jpeg_quality=jpeg_quality)
        return path

    def track_crop_track_directory(self, camera_id: str, safe_local_track_id: str) -> Path:
        track_name = safe_local_track_id
        if "_TRACK_" in safe_local_track_id:
            track_name = f"TRACK_{safe_local_track_id.split('_TRACK_')[-1]}"
        path = self.track_crops_directory / camera_id / track_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_track_crop(
        self,
        camera_id: str,
        safe_local_track_id: str,
        frame_number: int,
        crop: np.ndarray,
        *,
        jpeg_quality: int = 90,
    ) -> Path:
        track_directory = self.track_crop_track_directory(camera_id, safe_local_track_id)
        path = track_directory / f"frame_{frame_number:06d}.jpg"
        self._write_image(path, crop, jpeg_quality=jpeg_quality)
        return path

    def future_output_path(self, *parts: str) -> Path:
        return self.run_directory.joinpath(*parts)

    def _write_image(self, path: Path, frame: np.ndarray, *, jpeg_quality: int) -> None:
        success = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not success:
            raise OSError(f"Failed to write image: {path}")
