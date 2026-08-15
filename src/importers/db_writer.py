from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.env_loader import load_env_file
from src.importers.models import DryRunReport


NAMESPACE = uuid.UUID("5c8c52c7-6579-47e6-90f5-4ad4bc7a9a07")
DEFAULT_DB_SCHEMA = "vehicle_analytics"
DEFAULT_OBSERVATION_BATCH_SIZE = 1000
CORE_ARTIFACTS = [
    ("run_metadata", "run_metadata.json", "json"),
    ("summary", "summary.json", "json"),
    ("run_config", "run_config.yaml", "yaml"),
    ("tracks", "tracks.json", "json"),
    ("observations", "observations.csv", "csv"),
    ("evidence_index", "evidence_index.json", "json"),
    ("vehicle_enrichment", "vehicle_enrichment.json", "json"),
    ("ingestion_metrics", "ingestion_metrics.json", "json"),
    ("detection_tracking_metrics", "detection_tracking_metrics.json", "json"),
    ("track_lifecycle_metrics", "track_lifecycle_metrics.json", "json"),
    ("evidence_metrics", "evidence_metrics.json", "json"),
    ("vehicle_enrichment_metrics", "vehicle_enrichment_metrics.json", "json"),
    ("physical_vehicles", "physical_vehicles.json", "json"),
    ("vehicle_identity_map", "vehicle_identity_map.json", "json"),
    ("identity_decisions", "identity_decisions.json", "json"),
]
MIRROR_TABLES = [
    "processing_runs",
    "run_cameras",
    "vehicle_tracks",
    "media_assets",
    "track_observations",
    "track_evidence",
    "colour_predictions",
    "vehicle_attribute_predictions",
    "plate_detections",
    "plate_readings",
    "physical_vehicles",
    "physical_vehicle_tracks",
    "identity_decisions",
    "pipeline_artifacts",
    "pipeline_errors",
]
OPTIONAL_MIRROR_TABLES: set[str] = set()


class DatabaseWriteConfigurationError(RuntimeError):
    pass


class DatabaseImportError(RuntimeError):
    pass


class DuplicateRunError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DatabaseWriteConfig:
    dsn: str
    schema: str = DEFAULT_DB_SCHEMA
    observation_batch_size: int = DEFAULT_OBSERVATION_BATCH_SIZE
    supabase_url: str | None = None

    def __post_init__(self) -> None:
        _quote_identifier(self.schema)
        if self.observation_batch_size < 1:
            raise DatabaseWriteConfigurationError("observation_batch_size must be at least 1.")

    @property
    def supabase_host(self) -> str | None:
        if not self.supabase_url:
            return None
        return urlparse(self.supabase_url).netloc or None

    @classmethod
    def from_env(cls, env_path: str | Path = ".env") -> "DatabaseWriteConfig":
        values = load_env_file(env_path)
        values.update(os.environ)
        dsn = values.get("DATABASE_URL") or values.get("SUPABASE_DB_URL") or values.get("POSTGRES_URL")
        if not dsn:
            raise DatabaseWriteConfigurationError(
                "DB write requested, but no PostgreSQL connection string was found. "
                "Set DATABASE_URL, SUPABASE_DB_URL, or POSTGRES_URL for transactional imports."
            )
        batch_size = int(values.get("IMPORT_OBSERVATION_BATCH_SIZE") or DEFAULT_OBSERVATION_BATCH_SIZE)
        return cls(
            dsn=dsn,
            schema=values.get("DB_SCHEMA") or DEFAULT_DB_SCHEMA,
            observation_batch_size=batch_size,
            supabase_url=values.get("SUPABASE_URL"),
        )


@dataclass(slots=True)
class CanonicalRunPayload:
    tables: dict[str, list[dict[str, Any]]]
    run_key: str

    @property
    def counts(self) -> dict[str, int]:
        return {table: len(rows) for table, rows in self.tables.items()}


@dataclass(frozen=True, slots=True)
class ImportResult:
    run_key: str
    table_counts: dict[str, int]
    observation_batch_size: int
    observation_batch_count: int
    elapsed_seconds: float

    @property
    def rows_per_second(self) -> float:
        total = sum(self.table_counts.values())
        return float(total / self.elapsed_seconds) if self.elapsed_seconds > 0 else float(total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_key": self.run_key,
            "table_counts": dict(self.table_counts),
            "observation_batch_size": self.observation_batch_size,
            "observation_batch_count": self.observation_batch_count,
            "elapsed_seconds": self.elapsed_seconds,
            "rows_per_second": self.rows_per_second,
        }


class DatabaseRunWriter:
    """Transactional PostgreSQL writer for one file-based run mirror import."""

    def __init__(self, config: DatabaseWriteConfig) -> None:
        self.config = config
        self.schema_sql = _quote_identifier(config.schema)

    def apply_migration(self, migration_path: str | Path) -> None:
        sql = Path(migration_path).read_text(encoding="utf-8")
        with self._connect() as conn:
            with conn.transaction():
                conn.execute(sql)

    def insert_reference_run(self, payload: CanonicalRunPayload, *, replace: bool = False) -> ImportResult:
        started = time.perf_counter()
        with self._connect() as conn:
            with conn.transaction():
                columns_by_table = self._load_columns_by_table(conn)
                run_id = payload.tables["processing_runs"][0]["id"]
                if replace:
                    self._execute(conn, f"delete from {self._table('processing_runs')} where run_key = %s", (payload.run_key,))
                self._upsert_table(conn, "processing_runs", payload.tables["processing_runs"], ["run_key"], columns_by_table)
                self._upsert_table(conn, "run_cameras", payload.tables["run_cameras"], ["run_id", "camera_key"], columns_by_table)
                self._upsert_table(
                    conn,
                    "vehicle_tracks",
                    payload.tables["vehicle_tracks"],
                    ["run_id", "camera_id", "local_track_id"],
                    columns_by_table,
                )
                self._upsert_table(conn, "media_assets", payload.tables["media_assets"], ["run_id", "media_type", "relative_path"], columns_by_table)
                self._upsert_batched(
                    conn,
                    "track_observations",
                    payload.tables["track_observations"],
                    ["track_id", "frame_number"],
                    columns_by_table,
                    batch_size=self.config.observation_batch_size,
                )
                self._upsert_table(conn, "track_evidence", payload.tables["track_evidence"], ["track_id", "evidence_role", "frame_number"], columns_by_table)
                self._delete_rebuilt_children(conn, run_id, columns_by_table)
                for table in (
                    "colour_predictions",
                    "vehicle_attribute_predictions",
                    "plate_detections",
                    "plate_readings",
                    "physical_vehicles",
                    "physical_vehicle_tracks",
                    "identity_decisions",
                    "pipeline_artifacts",
                ):
                    if table == "physical_vehicles":
                        self._upsert_table(conn, table, payload.tables[table], ["run_id", "vehicle_key"], columns_by_table)
                    elif table == "physical_vehicle_tracks":
                        self._upsert_table(conn, table, payload.tables[table], ["vehicle_track_id"], columns_by_table)
                    else:
                        self._insert_rows(conn, table, payload.tables[table], columns_by_table)
        elapsed = time.perf_counter() - started
        observation_count = len(payload.tables["track_observations"])
        return ImportResult(
            run_key=payload.run_key,
            table_counts=payload.counts,
            observation_batch_size=self.config.observation_batch_size,
            observation_batch_count=_batch_count(observation_count, self.config.observation_batch_size),
            elapsed_seconds=elapsed,
        )

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:
            raise DatabaseWriteConfigurationError(
                "DB write requested, but psycopg is not installed in this environment. "
                "Install psycopg or run the migration/import from an environment that provides it."
            ) from exc
        return psycopg.connect(self.config.dsn)

    def _load_columns_by_table(self, conn: Any) -> dict[str, set[str]]:
        rows = self._fetchall(
            conn,
            """
            select table_name, column_name
            from information_schema.columns
            where table_schema = %s
              and table_name = any(%s)
            """,
            (self.config.schema, MIRROR_TABLES),
        )
        result: dict[str, set[str]] = {table: set() for table in MIRROR_TABLES}
        for row in rows:
            table_name = row[0]
            column_name = row[1]
            result.setdefault(str(table_name), set()).add(str(column_name))
        missing = [table for table in MIRROR_TABLES if table not in OPTIONAL_MIRROR_TABLES and (table not in result or not result[table])]
        if missing:
            raise DatabaseImportError(f"Target schema {self.config.schema!r} is missing required tables: {missing}")
        return result

    def _delete_rebuilt_children(self, conn: Any, run_id: str, columns_by_table: dict[str, set[str]]) -> None:
        if "track_id" in columns_by_table.get("plate_readings", set()):
            self._execute(
                conn,
                f"""
                delete from {self._table('plate_readings')} child
                using {self._table('vehicle_tracks')} track
                where child.track_id = track.id
                  and track.run_id = %s
                """,
                (run_id,),
            )
        elif "plate_detection_id" in columns_by_table.get("plate_readings", set()):
            self._execute(
                conn,
                f"""
                delete from {self._table('plate_readings')} reading
                using {self._table('plate_detections')} detection, {self._table('vehicle_tracks')} track
                where reading.plate_detection_id = detection.id
                  and detection.track_id = track.id
                  and track.run_id = %s
                """,
                (run_id,),
            )
        for table in ("plate_detections", "vehicle_attribute_predictions", "colour_predictions"):
            self._execute(
                conn,
                f"""
                delete from {self._table(table)} child
                using {self._table('vehicle_tracks')} track
                where child.track_id = track.id
                  and track.run_id = %s
                """,
                (run_id,),
            )
        self._execute(conn, f"delete from {self._table('pipeline_artifacts')} where run_id = %s", (run_id,))
        for table in ("identity_decisions", "physical_vehicle_tracks", "physical_vehicles"):
            if table in columns_by_table and columns_by_table[table]:
                self._execute(conn, f"delete from {self._table(table)} where run_id = %s" if table != "physical_vehicle_tracks" else f"delete from {self._table(table)} child using {self._table('physical_vehicles')} vehicle where child.physical_vehicle_id = vehicle.id and vehicle.run_id = %s", (run_id,))

    def _upsert_table(
        self,
        conn: Any,
        table: str,
        rows: list[dict[str, Any]],
        conflict_columns: list[str],
        columns_by_table: dict[str, set[str]],
    ) -> None:
        if not rows:
            return
        columns = _existing_columns(rows, columns_by_table[table])
        assignments = [column for column in columns if column not in set(conflict_columns) and column not in {"id", "created_at"}]
        if assignments:
            update_sql = ", ".join(f"{_quote_identifier(column)} = excluded.{_quote_identifier(column)}" for column in assignments)
            conflict_sql = f"do update set {update_sql}"
        else:
            conflict_sql = "do nothing"
        sql = _insert_sql(self._table(table), columns, f"on conflict ({_column_list(conflict_columns)}) {conflict_sql}")
        self._executemany(conn, sql, [_row_values(row, columns) for row in rows])

    def _upsert_batched(
        self,
        conn: Any,
        table: str,
        rows: list[dict[str, Any]],
        conflict_columns: list[str],
        columns_by_table: dict[str, set[str]],
        *,
        batch_size: int,
    ) -> None:
        for start in range(0, len(rows), batch_size):
            self._upsert_table(conn, table, rows[start : start + batch_size], conflict_columns, columns_by_table)

    def _insert_rows(self, conn: Any, table: str, rows: list[dict[str, Any]], columns_by_table: dict[str, set[str]]) -> None:
        if not rows:
            return
        columns = _existing_columns(rows, columns_by_table[table])
        sql = _insert_sql(self._table(table), columns, "on conflict (id) do nothing")
        self._executemany(conn, sql, [_row_values(row, columns) for row in rows])

    def _table(self, table: str) -> str:
        return f"{self.schema_sql}.{_quote_identifier(table)}"

    def _execute(self, conn: Any, sql: str, params: tuple[Any, ...]) -> Any:
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor

    def _fetchall(self, conn: Any, sql: str, params: tuple[Any, ...]) -> list[Any]:
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            return list(cursor.fetchall())

    def _executemany(self, conn: Any, sql: str, values: list[tuple[Any, ...]]) -> None:
        if not values:
            return
        with conn.cursor() as cursor:
            cursor.executemany(sql, values)


def build_payload(report: DryRunReport) -> CanonicalRunPayload:
    run_id = _id("processing_runs", report.run_key)
    camera_ids = {row.camera_key: _id("run_cameras", report.run_key, row.camera_key) for row in report.rows.run_cameras}
    track_ids = {row.ref.key: _id("vehicle_tracks", row.ref.key) for row in report.rows.vehicle_tracks}
    physical_vehicle_ids = {row.vehicle_key: _id("physical_vehicles", report.run_key, row.vehicle_key) for row in report.rows.physical_vehicles}
    media_ids = {
        _media_key(row.media_type, row.relative_path): _id("media_assets", report.run_key, row.media_type, row.relative_path or row.original_path or str(index))
        for index, row in enumerate(report.rows.media_assets)
    }

    tables: dict[str, list[dict[str, Any]]] = {table: [] for table in MIRROR_TABLES}
    run = report.rows.processing_runs[0]
    tables["processing_runs"].append(
        {
            "id": run_id,
            "run_key": run.run_key,
            "status": run.status,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
            "output_directory": run.output_directory,
            "project_name": run.project_name,
            "detection_backend": run.detection_backend,
            "tracking_backend": run.tracking_backend,
            "enrichment_enabled": run.enrichment_enabled,
            "processed_frames": run.processed_frames,
            "total_detections": _sum_metric_map(run.summary.get("detections_by_camera")),
            "raw_yolo_detections": run.raw_yolo_detections,
            "roi_filtered_detections": run.roi_filtered_detections,
            "completed_tracks": run.completed_tracks,
            "discarded_tracks": run.discarded_tracks,
            "config_path": run.config_path,
            "config_snapshot": run.config_snapshot,
            "summary": run.summary,
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
                "source": camera.source,
                "source_uri": camera.source,
                "source_type": camera.source_type,
                "enabled": camera.enabled,
                "fps": camera.fps,
                "total_frames": camera.total_frames,
                "processed_frames": camera.frames_processed,
                "frames_processed": camera.frames_processed,
                "detections_count": camera.detections_count,
                "frame_width": camera.width,
                "frame_height": camera.height,
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
                "tracker_namespace": track.tracker_namespace,
                "native_tracker_id": track.native_tracker_id,
                "track_status": track.track_status,
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
                "final_class_reason": track.final_class_reason,
                "completion_reason": track.completion_reason,
                "class_counts": track.class_counts,
                "class_confidence_sums": track.class_confidence_sums,
                "enrichment_summary": track.enrichment_summary,
                "evidence_record_count": track.evidence_record_count,
                "raw_track": track.raw_track,
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
                "relative_path": media.relative_path,
                "frame_number": media.frame_number,
                "timestamp_seconds": media.timestamp_seconds,
                "width": media.width,
                "height": media.height,
                "storage_provider": "local",
                "bucket": None,
                "object_key": None,
                "metadata": media.metadata,
            }
        )
    for evidence in report.rows.track_evidence:
        evidence_id = _id("track_evidence", evidence.ref.key, evidence.evidence_role, evidence.frame_number)
        crop_media_id = media_ids.get(_media_key("crop", evidence.crop_relative_path))
        tables["track_evidence"].append(
            {
                "id": evidence_id,
                "track_id": track_ids[evidence.ref.key],
                "run_id": run_id,
                "camera_id": camera_ids[evidence.ref.camera_key],
                "media_asset_id": crop_media_id,
                "crop_media_id": crop_media_id,
                "source_frame_media_id": media_ids.get(_media_key("source_full_frame", evidence.source_frame_relative_path)),
                "annotated_frame_media_id": media_ids.get(_media_key("annotated_frame", evidence.annotated_frame_relative_path)),
                "evidence_role": evidence.evidence_role,
                "frame_number": evidence.frame_number,
                "timestamp_seconds": evidence.timestamp_seconds,
                **_bbox_columns("bbox", evidence.bbox_xyxy, aliases=("bbox",)),
                "original_bbox": evidence.original_bbox_xyxy,
                "expanded_crop_bbox": evidence.expanded_crop_bbox_xyxy,
                **_bbox_columns("original_bbox", evidence.original_bbox_xyxy),
                **_bbox_columns("expanded_crop_bbox", evidence.expanded_crop_bbox_xyxy),
                "detection_confidence": evidence.detection_confidence,
                "quality_score": evidence.quality_score,
                "best_overall_score": evidence.quality_score,
                "sharpness_score": evidence.sharpness_score,
                "centeredness_score": evidence.centeredness_score,
                "edge_visibility_score": evidence.edge_visibility_score,
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
                "x1": observation.bbox_x1,
                "y1": observation.bbox_y1,
                "x2": observation.bbox_x2,
                "y2": observation.bbox_y2,
                "bbox_x1": observation.bbox_x1,
                "bbox_y1": observation.bbox_y1,
                "bbox_x2": observation.bbox_x2,
                "bbox_y2": observation.bbox_y2,
                "confidence": observation.detection_confidence,
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
                "media_asset_id": media_ids.get(_media_key("selected_colour_crop", prediction.evidence_relative_path)),
                "media_id": media_ids.get(_media_key("selected_colour_crop", prediction.evidence_relative_path)),
                "predicted_colour": prediction.predicted_colour,
                "normalized_colour": prediction.normalized_colour,
                "confidence": prediction.confidence,
                "source_model": prediction.source_model,
                "model_name": prediction.source_model,
                "status": prediction.status,
                "raw_response": prediction.raw_response,
                "prompt": prediction.prompt,
                "inference_time_ms": prediction.inference_duration_ms,
                "evidence_frame_number": prediction.evidence_frame_number,
                "evidence_timestamp_seconds": prediction.evidence_timestamp_seconds,
                "metadata": prediction.metadata,
            }
        )
    for index, prediction in enumerate(report.rows.vehicle_attribute_predictions):
        tables["vehicle_attribute_predictions"].append(
            {
                "id": _id("vehicle_attribute_predictions", prediction.ref.key, index, prediction.attribute_type, prediction.attribute_value, prediction.status),
                "track_id": track_ids[prediction.ref.key],
                "media_asset_id": media_ids.get(_media_key(f"selected_{prediction.attribute_type}_crop", prediction.evidence_relative_path)),
                "media_id": media_ids.get(_media_key(f"selected_{prediction.attribute_type}_crop", prediction.evidence_relative_path)),
                "attribute_type": prediction.attribute_type,
                "attribute_value": prediction.attribute_value,
                "label": prediction.attribute_value,
                "normalized_label": prediction.attribute_value.upper() if prediction.attribute_value else None,
                "status": prediction.status,
                "confidence": prediction.confidence,
                "source_backend": prediction.source_backend,
                "source_model": prediction.source_model,
                "raw_response": prediction.raw_response,
                "evidence_frame_number": prediction.evidence_frame_number,
                "evidence_timestamp_seconds": prediction.evidence_timestamp_seconds,
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
                "media_asset_id": media_ids.get(_media_key("plate_crop", detection.crop_relative_path)),
                "media_id": media_ids.get(_media_key("plate_crop", detection.crop_relative_path)),
                "frame_number": detection.frame_number,
                "timestamp_seconds": detection.timestamp_seconds,
                "bbox": detection.plate_bbox,
                "detection_confidence": detection.confidence,
                "confidence": detection.confidence,
                "source_model": detection.source_model,
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
                "source_model": reading.ocr_backend,
                "model_name": reading.ocr_backend,
                "raw_response": reading.raw_response,
                "status": reading.reason,
                "plate_colour": reading.plate_colour,
                "registration_category": reading.registration_category,
                "is_selected": reading.is_selected,
                "verified": False,
                "metadata": reading.metadata,
            }
        )
    for vehicle in report.rows.physical_vehicles:
        tables["physical_vehicles"].append(
            {
                "id": physical_vehicle_ids[vehicle.vehicle_key],
                "run_id": run_id,
                "vehicle_key": vehicle.vehicle_key,
                "vehicle_class": vehicle.vehicle_class,
                "vehicle_colour": vehicle.vehicle_colour,
                "first_timestamp_seconds": vehicle.first_timestamp_seconds,
                "last_timestamp_seconds": vehicle.last_timestamp_seconds,
                "identity_confidence": vehicle.identity_confidence,
                "identity_method": vehicle.identity_method,
                "identity_status": vehicle.identity_status,
                "consensus_plate_text": vehicle.consensus_plate_text,
                "plate_confidence": vehicle.plate_confidence,
                "metadata": vehicle.metadata,
            }
        )
    for membership in report.rows.physical_vehicle_tracks:
        physical_vehicle_id = physical_vehicle_ids.get(membership.vehicle_key)
        vehicle_track_id = track_ids.get(membership.ref.key)
        if physical_vehicle_id is None or vehicle_track_id is None:
            continue
        tables["physical_vehicle_tracks"].append(
            {
                "physical_vehicle_id": physical_vehicle_id,
                "vehicle_track_id": vehicle_track_id,
                "association_score": membership.association_score,
                "association_method": membership.association_method,
                "association_reason": membership.association_reason,
                "metadata": membership.metadata,
            }
        )
    for index, decision in enumerate(report.rows.identity_decisions):
        tables["identity_decisions"].append(
            {
                "id": _id(
                    "identity_decisions",
                    report.run_key,
                    index,
                    decision.source_ref.key if decision.source_ref else "",
                    decision.target_ref.key if decision.target_ref else "",
                    decision.decision,
                ),
                "run_id": run_id,
                "source_track_id": track_ids.get(decision.source_ref.key) if decision.source_ref else None,
                "target_track_id": track_ids.get(decision.target_ref.key) if decision.target_ref else None,
                "decision": decision.decision,
                "final_score": decision.final_score,
                "plate_score": decision.plate_score,
                "spatial_score": decision.spatial_score,
                "temporal_score": decision.temporal_score,
                "motion_score": decision.motion_score,
                "appearance_score": decision.appearance_score,
                "colour_score": decision.colour_score,
                "reason": decision.reason,
                "metadata": decision.metadata,
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


def _insert_sql(table_sql: str, columns: list[str], suffix: str) -> str:
    placeholders = ", ".join(["%s"] * len(columns))
    return f"insert into {table_sql} ({_column_list(columns)}) values ({placeholders}) {suffix}"


def _column_list(columns: list[str]) -> str:
    return ", ".join(_quote_identifier(column) for column in columns)


def _existing_columns(rows: list[dict[str, Any]], table_columns: set[str]) -> list[str]:
    wanted = [column for column in rows[0] if column in table_columns]
    if not wanted:
        raise DatabaseImportError("No payload columns match target table columns.")
    return wanted


def _row_values(row: dict[str, Any], columns: list[str]) -> tuple[Any, ...]:
    return tuple(_db_value(row[column]) for column in columns)


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


def _bbox_columns(prefix: str, bbox: list[Any] | None, aliases: tuple[str, ...] = ()) -> dict[str, Any]:
    names = ["x1", "y1", "x2", "y2"]
    values = list(bbox or [])
    values = values[:4] + [None] * max(0, 4 - len(values))
    payload = {f"{prefix}_{name}": values[index] for index, name in enumerate(names)}
    for alias in aliases:
        payload.update({f"{alias}_{name}": values[index] for index, name in enumerate(names)})
    return payload


def _sum_metric_map(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None
    total = 0
    for item in value.values():
        if item is None:
            continue
        total += int(item)
    return total


def _batch_count(row_count: int, batch_size: int) -> int:
    if row_count <= 0:
        return 0
    return ((row_count - 1) // batch_size) + 1


def _quote_identifier(identifier: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(identifier)):
        raise DatabaseWriteConfigurationError(f"Unsafe SQL identifier: {identifier!r}")
    return f'"{identifier}"'
