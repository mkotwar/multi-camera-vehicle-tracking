from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.importers.models import DryRunReport


NAMESPACE = uuid.UUID("5c8c52c7-6579-47e6-90f5-4ad4bc7a9a07")
CORE_ARTIFACTS = [
    ("run_metadata", "run_metadata.json", "json"),
    ("summary", "summary.json", "json"),
    ("run_config", "run_config.yaml", "yaml"),
    ("tracks", "tracks.json", "json"),
    ("observations", "observations.csv", "csv"),
    ("evidence_index", "evidence_index.json", "json"),
    ("vehicle_enrichment", "vehicle_enrichment.json", "json"),
]


class DatabaseWriteConfigurationError(RuntimeError):
    pass


class DuplicateRunError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DatabaseWriteConfig:
    dsn: str
    supabase_url: str | None = None

    @property
    def supabase_host(self) -> str | None:
        if not self.supabase_url:
            return None
        return urlparse(self.supabase_url).netloc or None

    @classmethod
    def from_env(cls, env_path: str | Path = ".env") -> "DatabaseWriteConfig":
        values = _load_env_file(env_path)
        values.update(os.environ)
        dsn = values.get("DATABASE_URL") or values.get("SUPABASE_DB_URL") or values.get("POSTGRES_URL")
        if not dsn:
            raise DatabaseWriteConfigurationError(
                "DB write requested, but no PostgreSQL connection string was found. "
                "Set DATABASE_URL, SUPABASE_DB_URL, or POSTGRES_URL for transactional imports."
            )
        return cls(dsn=dsn, supabase_url=values.get("SUPABASE_URL"))


@dataclass(slots=True)
class CanonicalRunPayload:
    tables: dict[str, list[dict[str, Any]]]
    run_key: str

    @property
    def counts(self) -> dict[str, int]:
        return {table: len(rows) for table, rows in self.tables.items()}


class DatabaseRunWriter:
    """Transactional writer for canonical importer rows.

    This writer intentionally uses a PostgreSQL connection string instead of the
    Supabase REST client so a one-run import can be guarded by a transaction.
    """

    def __init__(self, config: DatabaseWriteConfig) -> None:
        self.config = config

    def apply_migration(self, migration_path: str | Path) -> None:
        sql = Path(migration_path).read_text(encoding="utf-8")
        with self._connect() as conn:
            with conn.transaction():
                conn.execute(sql)

    def insert_reference_run(self, payload: CanonicalRunPayload, *, replace: bool = False) -> dict[str, int]:
        with self._connect() as conn:
            with conn.transaction():
                exists = conn.execute("select id from public.processing_runs where run_key = %s", (payload.run_key,)).fetchone()
                if exists and not replace:
                    raise DuplicateRunError(f"Run key already exists: {payload.run_key}. Re-run with --replace to delete and re-import it.")
                if exists and replace:
                    conn.execute("delete from public.processing_runs where run_key = %s", (payload.run_key,))
                for table, rows in payload.tables.items():
                    if rows:
                        _insert_rows(conn, table, rows)
        return payload.counts

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:
            raise DatabaseWriteConfigurationError(
                "DB write requested, but psycopg is not installed in this environment. "
                "Install psycopg or run the migration/import from an environment that provides it."
            ) from exc
        return psycopg.connect(self.config.dsn)


def build_payload(report: DryRunReport) -> CanonicalRunPayload:
    run_id = _id("processing_runs", report.run_key)
    camera_ids = {row.camera_key: _id("run_cameras", report.run_key, row.camera_key) for row in report.rows.run_cameras}
    track_ids = {row.ref.key: _id("vehicle_tracks", row.ref.key) for row in report.rows.vehicle_tracks}
    media_ids = {
        _media_key(row.media_type, row.relative_path): _id("media_assets", report.run_key, row.media_type, row.relative_path or row.original_path or str(index))
        for index, row in enumerate(report.rows.media_assets)
    }

    tables: dict[str, list[dict[str, Any]]] = {
        "processing_runs": [],
        "run_cameras": [],
        "vehicle_tracks": [],
        "media_assets": [],
        "track_observations": [],
        "track_evidence": [],
        "colour_predictions": [],
        "vehicle_attribute_predictions": [],
        "plate_detections": [],
        "plate_readings": [],
        "pipeline_artifacts": [],
        "pipeline_errors": [],
        "chat_sessions": [],
        "chat_session_runs": [],
        "chat_messages": [],
    }
    run = report.rows.processing_runs[0]
    tables["processing_runs"].append(
        {
            "id": run_id,
            "run_key": run.run_key,
            "status": run.status,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
            "project_name": run.project_name,
            "detection_backend": run.detection_backend,
            "tracking_backend": run.tracking_backend,
            "enrichment_enabled": run.enrichment_enabled,
            "processed_frames": run.processed_frames,
            "total_detections": _sum_metric_map(run.metrics.get("detections_by_camera")),
            "raw_yolo_detections": run.raw_yolo_detections,
            "roi_filtered_detections": run.roi_filtered_detections,
            "completed_tracks": run.completed_tracks,
            "discarded_tracks": run.discarded_tracks,
            "config_path": run.config_path,
            "config_snapshot": run.config_snapshot,
            "metrics": run.metrics,
            "metadata": run.metadata,
        }
    )
    for camera in report.rows.run_cameras:
        tables["run_cameras"].append(
            {
                "id": camera_ids[camera.camera_key],
                "run_id": run_id,
                "camera_key": camera.camera_key,
                "source_uri": camera.source,
                "source_type": camera.source_type,
                "enabled": camera.enabled,
                "frame_width": camera.width,
                "frame_height": camera.height,
                "fps": camera.fps,
                "frames_processed": camera.frames_processed,
                "detections_count": camera.detections_count,
                "metadata": camera.metadata,
            }
        )
    for track in report.rows.vehicle_tracks:
        tables["vehicle_tracks"].append(
            {
                "id": track_ids[track.ref.key],
                "run_id": run_id,
                "camera_id": camera_ids[track.ref.camera_key],
                "camera_key": track.ref.camera_key,
                "local_track_id": track.ref.local_track_id,
                "short_track_id": track.ref.local_track_id,
                "tracker_namespace": track.tracker_namespace,
                "native_tracker_id": track.native_tracker_id,
                "track_status": track.track_status,
                "completion_reason": track.completion_reason,
                "first_frame": track.first_frame,
                "last_frame": track.last_frame,
                "first_seen_seconds": track.first_seen_seconds,
                "last_seen_seconds": track.last_seen_seconds,
                "observation_count": track.observation_count,
                "lost_frames": track.lost_frames,
                "vehicle_class": track.vehicle_class,
                "vehicle_class_confidence": track.vehicle_class_confidence,
                "vehicle_colour": track.vehicle_colour,
                "vehicle_colour_status": track.vehicle_colour_status,
                "body_type": track.body_type,
                "body_type_status": track.body_type_status,
                "plate_text": track.plate_text,
                "plate_detected": track.plate_detected,
                "plate_colour": track.plate_colour,
                "registration_category": track.registration_category,
                "class_counts": track.class_counts,
                "class_confidence_sums": track.class_confidence_sums,
                "raw_track": track.raw_track,
                "enrichment_summary": track.enrichment_summary,
            }
        )
    for media in report.rows.media_assets:
        tables["media_assets"].append(
            {
                "id": media_ids[_media_key(media.media_type, media.relative_path)],
                "run_id": run_id,
                "camera_id": camera_ids.get(media.camera_key) if media.camera_key else None,
                "track_id": _track_id_from_media(media, track_ids, report.run_key),
                "media_type": media.media_type,
                "storage_provider": "local",
                "bucket": None,
                "object_key": None,
                "relative_path": media.relative_path,
                "width": media.width,
                "height": media.height,
                "frame_number": media.frame_number,
                "metadata": media.metadata,
            }
        )
    evidence_ids: dict[tuple[str, str | None, int | None], str] = {}
    for evidence in report.rows.track_evidence:
        evidence_id = _id("track_evidence", evidence.ref.key, evidence.evidence_role, evidence.frame_number)
        evidence_ids[(evidence.ref.key, evidence.evidence_role, evidence.frame_number)] = evidence_id
        tables["track_evidence"].append(
            {
                "id": evidence_id,
                "track_id": track_ids[evidence.ref.key],
                "run_id": run_id,
                "camera_id": camera_ids[evidence.ref.camera_key],
                "evidence_role": evidence.evidence_role,
                "frame_number": evidence.frame_number,
                "timestamp_seconds": evidence.timestamp_seconds,
                "crop_media_id": media_ids.get(_media_key("crop", evidence.crop_relative_path)),
                "source_frame_media_id": media_ids.get(_media_key("source_full_frame", evidence.source_frame_relative_path)),
                "annotated_frame_media_id": media_ids.get(_media_key("annotated_frame", evidence.annotated_frame_relative_path)),
                **_bbox_columns("bbox", evidence.bbox_xyxy),
                **_bbox_columns("original_bbox", evidence.original_bbox_xyxy),
                **_bbox_columns("expanded_crop_bbox", evidence.expanded_crop_bbox_xyxy),
                "detection_confidence": evidence.detection_confidence,
                "quality_score": evidence.quality_score,
                "sharpness_score": evidence.sharpness_score,
                "brightness_score": evidence.brightness_score,
                "crop_width": evidence.crop_width,
                "crop_height": evidence.crop_height,
                "resolution_tier": evidence.resolution_tier,
                "selected_for_colour": evidence.selected_for_colour,
                "selected_for_body_type": evidence.selected_for_body_type,
                "evidence_source": evidence.evidence_source,
                "candidate_rank": evidence.candidate_rank,
                "metadata": evidence.metadata,
            }
        )
    for observation in report.rows.track_observations:
        tables["track_observations"].append(
            {
                "track_id": track_ids[observation.ref.key],
                "run_id": run_id,
                "camera_id": camera_ids[observation.ref.camera_key],
                "frame_number": observation.frame_number,
                "timestamp_seconds": observation.timestamp_seconds,
                "bbox_x1": observation.bbox_x1,
                "bbox_y1": observation.bbox_y1,
                "bbox_x2": observation.bbox_x2,
                "bbox_y2": observation.bbox_y2,
                "detection_confidence": observation.detection_confidence,
                "raw_class_id": observation.raw_class_id,
                "raw_class_name": observation.raw_class_name,
                "tracker_namespace": observation.tracker_namespace,
                "native_tracker_id": observation.native_tracker_id,
                "metadata": observation.metadata,
            }
        )
    for index, prediction in enumerate(report.rows.colour_predictions):
        tables["colour_predictions"].append(
            {
                "id": _id("colour_predictions", prediction.ref.key, index, prediction.evidence_relative_path, prediction.normalized_colour, prediction.status),
                "track_id": track_ids[prediction.ref.key],
                "evidence_id": None,
                "media_id": media_ids.get(_media_key("selected_colour_crop", prediction.evidence_relative_path)),
                "predicted_colour": prediction.predicted_colour,
                "normalized_colour": prediction.normalized_colour,
                "status": prediction.status,
                "confidence": prediction.confidence,
                "model_name": prediction.source_model,
                "model_version": None,
                "prompt": prediction.prompt,
                "raw_response": prediction.raw_response,
                "inference_time_ms": prediction.inference_duration_ms,
                "fallback_attempt": None,
                "selection_reason": prediction.metadata.get("selection_tier") or prediction.metadata.get("reason"),
                "metadata": prediction.metadata,
            }
        )
    for index, prediction in enumerate(report.rows.vehicle_attribute_predictions):
        tables["vehicle_attribute_predictions"].append(
            {
                "id": _id("vehicle_attribute_predictions", prediction.ref.key, index, prediction.attribute_type, prediction.label, prediction.status),
                "track_id": track_ids[prediction.ref.key],
                "evidence_id": None,
                "media_id": media_ids.get(_media_key(f"selected_{prediction.attribute_type}_crop", prediction.evidence_relative_path)),
                "attribute_type": prediction.attribute_type,
                "label": prediction.label,
                "normalized_label": prediction.label.upper() if prediction.label else None,
                "status": prediction.status,
                "confidence": prediction.confidence,
                "source_backend": prediction.source_backend,
                "source_model": prediction.source_model,
                "raw_response": prediction.raw_response,
                "metadata": prediction.metadata,
            }
        )
    plate_detection_ids: dict[str, str] = {}
    for index, detection in enumerate(report.rows.plate_detections):
        detection_id = _id("plate_detections", detection.ref.key, index)
        plate_detection_ids[detection.ref.key] = detection_id
        tables["plate_detections"].append(
            {
                "id": detection_id,
                "track_id": track_ids[detection.ref.key],
                "evidence_id": None,
                "media_id": None,
                "frame_number": None,
                "timestamp_seconds": None,
                "bbox": detection.plate_bbox,
                "confidence": detection.confidence,
                "crop_media_id": media_ids.get(_media_key("plate_crop", detection.crop_relative_path)),
                "status": detection.quality_status,
                "metadata": detection.metadata,
            }
        )
    for index, reading in enumerate(report.rows.plate_readings):
        tables["plate_readings"].append(
            {
                "id": _id("plate_readings", reading.ref.key, index, reading.plate_text),
                "plate_detection_id": plate_detection_ids.get(reading.ref.key),
                "track_id": track_ids[reading.ref.key],
                "raw_text": reading.plate_text,
                "normalized_text": reading.plate_text.upper() if reading.plate_text else None,
                "confidence": reading.confidence,
                "status": reading.reason,
                "plate_colour": reading.plate_colour,
                "verified": False,
                "verification_source": None,
                "model_name": reading.ocr_backend,
                "model_version": None,
                "raw_response": reading.raw_response,
                "metadata": reading.metadata,
            }
        )
    for artifact_type, relative_path, file_format in CORE_ARTIFACTS:
        if (report.run_dir / relative_path).exists():
            tables["pipeline_artifacts"].append(
                {
                    "id": _id("pipeline_artifacts", report.run_key, relative_path),
                    "run_id": run_id,
                    "artifact_type": artifact_type,
                    "relative_path": relative_path,
                    "format": file_format,
                    "metadata": {},
                }
            )
    return CanonicalRunPayload(tables=tables, run_key=report.run_key)


def _insert_rows(conn: Any, table: str, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0].keys())
    placeholders = ", ".join(["%s"] * len(columns))
    column_sql = ", ".join(columns)
    sql = f"insert into public.{table} ({column_sql}) values ({placeholders})"
    for row in rows:
        conn.execute(sql, tuple(_db_value(row[column]) for column in columns))


def _db_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        try:
            from psycopg.types.json import Jsonb
            return Jsonb(value)
        except ImportError:
            return json.dumps(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    return value


def _id(*parts: Any) -> str:
    return str(uuid.uuid5(NAMESPACE, "|".join(str(part) for part in parts)))


def _media_key(media_type: str, relative_path: str | None) -> tuple[str, str | None]:
    return (media_type, relative_path)


def _track_id_from_media(media: Any, track_ids: dict[str, str], run_key: str) -> str | None:
    if not media.camera_key or not media.track_local_id:
        return None
    return track_ids.get(f"{run_key}|{media.camera_key}|{media.track_local_id}")


def _bbox_columns(prefix: str, bbox: list[Any] | None) -> dict[str, Any]:
    names = ["x1", "y1", "x2", "y2"]
    values = list(bbox or [])
    values = values[:4] + [None] * max(0, 4 - len(values))
    return {f"{prefix}_{name}": values[index] for index, name in enumerate(names)}


def _load_env_file(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _sum_metric_map(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None
    total = 0
    for item in value.values():
        if item is None:
            continue
        total += int(item)
    return total
