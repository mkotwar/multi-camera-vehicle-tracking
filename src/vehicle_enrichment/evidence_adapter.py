from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2

from src.models import LocalTrack, TrackEvidence
from src.output_writer import RunOutputManager

from .schemas import EnrichmentEvidenceItem


class EvidenceAdapter:
    def __init__(self, config: dict[str, Any], output_manager: RunOutputManager, logger: logging.Logger) -> None:
        self.config = dict(config)
        self.output_manager = output_manager
        self.logger = logger
        evidence_config = dict(self.config.get("evidence", {}) or {})
        self.save_vehicle_crops = bool(evidence_config.get("save_vehicle_crops", True))

    def adapt_track(
        self,
        track: LocalTrack,
        finalized_evidence_records: list[TrackEvidence | dict[str, Any]],
    ) -> list[EnrichmentEvidenceItem]:
        items: list[EnrichmentEvidenceItem] = []
        seen_keys: set[tuple[int, tuple[int, int, int, int]]] = set()
        for record in finalized_evidence_records:
            normalized = self._normalize_record(track, record)
            if normalized is None:
                continue
            dedupe_key = (normalized.frame_number, tuple(int(round(item)) for item in normalized.bbox_xyxy))
            if dedupe_key in seen_keys:
                normalized.rejection_reasons.append("duplicate_evidence")
                continue
            seen_keys.add(dedupe_key)
            items.append(normalized)
        return items

    def _normalize_record(
        self,
        track: LocalTrack,
        record: TrackEvidence | dict[str, Any],
    ) -> EnrichmentEvidenceItem | None:
        payload = self._record_to_dict(record)
        if str(payload.get("local_track_id")) != track.local_track_id:
            return None

        bbox_xyxy = tuple(float(item) for item in payload.get("bbox_xyxy", (0, 0, 0, 0)))
        original_bbox_xyxy = tuple(float(item) for item in payload.get("original_bbox", bbox_xyxy))
        expanded_crop_bbox_xyxy = tuple(float(item) for item in payload.get("expanded_crop_bbox", bbox_xyxy))
        frame_number = int(payload.get("frame_number", 0))
        timestamp_seconds = float(payload.get("timestamp_seconds", 0.0))
        source_image_path = self._pick_source_image(payload)
        annotated_frame_path = self._resolve_path(payload.get("annotated_frame_path"))
        crop_path = self._resolve_path(payload.get("crop_path"))
        rejection_reasons: list[str] = []

        if bbox_xyxy[2] <= bbox_xyxy[0] or bbox_xyxy[3] <= bbox_xyxy[1]:
            rejection_reasons.append("invalid_bbox")
            return EnrichmentEvidenceItem(
                local_track_id=track.local_track_id,
                camera_id=track.camera_id,
                native_tracker_id=track.native_tracker_id,
                frame_number=frame_number,
                timestamp_seconds=timestamp_seconds,
                source_image_path=str(source_image_path) if source_image_path else None,
                vehicle_crop_path=str(crop_path) if crop_path else None,
                annotated_frame_path=str(annotated_frame_path) if annotated_frame_path else None,
                bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
                original_bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
                expanded_crop_bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
                evidence_role=str(payload.get("role", "UNKNOWN")),
                detection_confidence=float(payload.get("confidence", 0.0)),
                vehicle_class=str(payload.get("final_class", payload.get("raw_class_name", "UNKNOWN")) or "UNKNOWN"),
                source_frame_width=int(payload.get("source_frame_width", 0)),
                source_frame_height=int(payload.get("source_frame_height", 0)),
                context_padding_ratio=float(payload.get("context_padding_ratio", 0.0)),
                original_crop_width=0,
                original_crop_height=0,
                crop_width=0,
                crop_height=0,
                crop_area=0,
                sharpness_score=float(payload.get("sharpness_score", 0.0)),
                brightness_score=0.0,
                border_penalty=1.0,
                clipping_ratio=1.0,
                quality_score=0.0,
                evidence_source=str(payload.get("evidence_source", "existing_track_evidence")),
                rejection_reasons=rejection_reasons,
            )

        if source_image_path is None:
            rejection_reasons.append("missing_source_image")
        clipped_bbox, clipping_ratio, border_penalty, source_size = self._clip_bbox(bbox_xyxy, source_image_path or annotated_frame_path)
        if clipped_bbox is None:
            rejection_reasons.append("invalid_bbox")
            return EnrichmentEvidenceItem(
                local_track_id=track.local_track_id,
                camera_id=track.camera_id,
                native_tracker_id=track.native_tracker_id,
                frame_number=frame_number,
                timestamp_seconds=timestamp_seconds,
                source_image_path=str(source_image_path) if source_image_path else None,
                vehicle_crop_path=str(crop_path) if crop_path else None,
                annotated_frame_path=str(annotated_frame_path) if annotated_frame_path else None,
                bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
                original_bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
                expanded_crop_bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
                evidence_role=str(payload.get("role", "UNKNOWN")),
                detection_confidence=float(payload.get("confidence", 0.0)),
                vehicle_class=str(payload.get("final_class", payload.get("raw_class_name", "UNKNOWN")) or "UNKNOWN"),
                source_frame_width=int(payload.get("source_frame_width", 0)),
                source_frame_height=int(payload.get("source_frame_height", 0)),
                context_padding_ratio=float(payload.get("context_padding_ratio", 0.0)),
                original_crop_width=0,
                original_crop_height=0,
                crop_width=0,
                crop_height=0,
                crop_area=0,
                sharpness_score=float(payload.get("sharpness_score", 0.0)),
                brightness_score=0.0,
                border_penalty=1.0,
                clipping_ratio=1.0,
                quality_score=0.0,
                evidence_source=str(payload.get("evidence_source", "existing_track_evidence")),
                rejection_reasons=rejection_reasons,
            )

        if crop_path is None and (source_image_path or annotated_frame_path) is not None:
            crop_path = self._extract_fallback_crop(
                track=track,
                frame_number=frame_number,
                role=str(payload.get("role", "UNKNOWN")),
                source_image_path=source_image_path or annotated_frame_path,
                bbox_xyxy=clipped_bbox,
            )
            if crop_path is None:
                rejection_reasons.append("crop_extraction_failed")

        fallback_dimensions = self._read_image_dimensions(crop_path)
        original_crop_width = int(payload.get("original_crop_width", fallback_dimensions[0] if fallback_dimensions else 0))
        original_crop_height = int(payload.get("original_crop_height", fallback_dimensions[1] if fallback_dimensions else 0))
        crop_width = max(0, int(round(clipped_bbox[2] - clipped_bbox[0])))
        crop_height = max(0, int(round(clipped_bbox[3] - clipped_bbox[1])))
        if crop_width <= 0 or crop_height <= 0:
            rejection_reasons.append("empty_crop")

        return EnrichmentEvidenceItem(
            local_track_id=track.local_track_id,
            camera_id=track.camera_id,
            native_tracker_id=track.native_tracker_id,
            frame_number=frame_number,
            timestamp_seconds=timestamp_seconds,
            source_image_path=str(source_image_path) if source_image_path else None,
            vehicle_crop_path=str(crop_path) if crop_path else None,
            annotated_frame_path=str(annotated_frame_path) if annotated_frame_path else None,
            bbox_xyxy=clipped_bbox,
            original_bbox_xyxy=original_bbox_xyxy,
            expanded_crop_bbox_xyxy=expanded_crop_bbox_xyxy,
            evidence_role=str(payload.get("role", "UNKNOWN")),
            detection_confidence=float(payload.get("confidence", 0.0)),
            vehicle_class=str(payload.get("final_class", payload.get("raw_class_name", "UNKNOWN")) or "UNKNOWN"),
            source_frame_width=int(payload.get("source_frame_width", source_size[0] if source_size else 0)),
            source_frame_height=int(payload.get("source_frame_height", source_size[1] if source_size else 0)),
            context_padding_ratio=float(payload.get("context_padding_ratio", 0.0)),
            original_crop_width=original_crop_width,
            original_crop_height=original_crop_height,
            crop_width=crop_width,
            crop_height=crop_height,
            crop_area=crop_width * crop_height,
            sharpness_score=float(payload.get("sharpness_score", 0.0)),
            brightness_score=0.0,
            border_penalty=border_penalty if source_size is not None else 1.0,
            clipping_ratio=clipping_ratio,
            quality_score=0.0,
            evidence_source=str(payload.get("evidence_source", "existing_track_evidence")),
            trigger_x=float(payload.get("trigger_x")) if payload.get("trigger_x") is not None else None,
            trigger_y=float(payload.get("trigger_y")) if payload.get("trigger_y") is not None else None,
            zone_top=int(payload.get("zone_top")) if payload.get("zone_top") is not None else None,
            zone_bottom=int(payload.get("zone_bottom")) if payload.get("zone_bottom") is not None else None,
            rejection_reasons=rejection_reasons,
        )

    @staticmethod
    def _record_to_dict(record: TrackEvidence | dict[str, Any]) -> dict[str, Any]:
        if isinstance(record, dict):
            return dict(record)
        return {
            "local_track_id": record.local_track_id,
            "camera_id": record.camera_id,
            "native_tracker_id": record.native_tracker_id,
            "tracker_namespace": record.tracker_namespace,
            "role": record.role,
            "frame_number": record.frame_number,
            "timestamp_seconds": record.timestamp_seconds,
            "raw_class_name": record.raw_class_name,
            "final_class": record.final_class,
            "confidence": record.confidence,
            "crop_path": record.crop_path,
            "annotated_frame_path": record.annotated_frame_path,
            "bbox_xyxy": record.bbox_xyxy,
            "original_bbox": record.original_bbox_xyxy,
            "expanded_crop_bbox": record.expanded_crop_bbox_xyxy,
            "context_padding_ratio": record.context_padding_ratio,
            "source_frame_width": record.source_frame_width,
            "source_frame_height": record.source_frame_height,
            "original_crop_width": record.original_crop_width,
            "original_crop_height": record.original_crop_height,
            "sharpness_score": record.sharpness_score,
            "best_overall_score": record.best_overall_score,
            "evidence_source": "existing_track_evidence",
        }

    def _pick_source_image(self, payload: dict[str, Any]) -> Path | None:
        annotated_path = self._resolve_path(payload.get("annotated_frame_path"))
        if annotated_path is not None:
            return annotated_path
        return self._resolve_path(payload.get("crop_path"))

    @staticmethod
    def _resolve_path(value: Any) -> Path | None:
        if value in (None, ""):
            return None
        path = Path(str(value)).expanduser()
        try:
            resolved = path.resolve()
        except OSError:
            return None
        return resolved if resolved.exists() else None

    def _clip_bbox(
        self,
        bbox_xyxy: tuple[float, float, float, float],
        source_image_path: Path | None,
    ) -> tuple[tuple[float, float, float, float] | None, float, float, tuple[int, int] | None]:
        if source_image_path is None or not source_image_path.exists():
            return bbox_xyxy, 0.0, 0.0, None
        image = cv2.imread(str(source_image_path))
        if image is None or image.size == 0:
            return None, 1.0, 1.0, None
        height, width = image.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy
        original_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        clipped = (
            max(0.0, min(float(width), x1)),
            max(0.0, min(float(height), y1)),
            max(0.0, min(float(width), x2)),
            max(0.0, min(float(height), y2)),
        )
        clipped_area = max(0.0, clipped[2] - clipped[0]) * max(0.0, clipped[3] - clipped[1])
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            return None, 1.0, 1.0, (width, height)
        clipping_ratio = 0.0 if original_area <= 0.0 else max(0.0, min(1.0, 1.0 - (clipped_area / original_area)))
        horizontal_margin = min(clipped[0], max(0.0, width - clipped[2]))
        vertical_margin = min(clipped[1], max(0.0, height - clipped[3]))
        margin_threshold = max(1.0, min(width, height) * float(self.config.get("evidence", {}).get("border_margin_ratio", 0.02)))
        border_penalty = 1.0 - max(0.0, min(1.0, min(horizontal_margin, vertical_margin) / margin_threshold))
        return clipped, clipping_ratio, border_penalty, (width, height)

    def _extract_fallback_crop(
        self,
        *,
        track: LocalTrack,
        frame_number: int,
        role: str,
        source_image_path: Path,
        bbox_xyxy: tuple[float, float, float, float],
    ) -> Path | None:
        image = cv2.imread(str(source_image_path))
        if image is None or image.size == 0:
            return None
        x1, y1, x2, y2 = (int(round(item)) for item in bbox_xyxy)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return self.output_manager.save_vehicle_enrichment_crop(
            track.local_track_id,
            frame_number,
            crop,
            suffix=f"{str(role).upper()}_fallback",
        )

    @staticmethod
    def _read_image_dimensions(image_path: Path | None) -> tuple[int, int] | None:
        if image_path is None or not image_path.exists():
            return None
        image = cv2.imread(str(image_path))
        if image is None or image.size == 0:
            return None
        height, width = image.shape[:2]
        return int(width), int(height)
