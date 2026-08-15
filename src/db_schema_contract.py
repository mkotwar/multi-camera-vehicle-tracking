from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .importers.db_writer import DEFAULT_DB_SCHEMA, MIRROR_TABLES, DatabaseWriteConfig


TYPE_ALIASES = {
    "text": {"text", "character varying"},
    "uuid": {"uuid"},
    "jsonb": {"jsonb"},
    "boolean": {"boolean"},
    "integer": {"integer", "bigint"},
    "bigint": {"bigint", "integer"},
    "double precision": {"double precision", "real"},
    "timestamp with time zone": {"timestamp with time zone"},
}


CANONICAL_SCHEMA: dict[str, dict[str, str]] = {
    "processing_runs": {
        "id": "uuid",
        "run_key": "text",
        "status": "text",
        "started_at": "timestamp with time zone",
        "completed_at": "timestamp with time zone",
        "output_directory": "text",
        "project_name": "text",
        "detection_backend": "text",
        "tracking_backend": "text",
        "enrichment_enabled": "boolean",
        "processed_frames": "bigint",
        "total_detections": "bigint",
        "raw_yolo_detections": "bigint",
        "roi_filtered_detections": "bigint",
        "completed_tracks": "integer",
        "discarded_tracks": "integer",
        "config_path": "text",
        "config_snapshot": "jsonb",
        "summary": "jsonb",
        "metrics": "jsonb",
        "metadata": "jsonb",
    },
    "run_cameras": {
        "id": "uuid",
        "run_id": "uuid",
        "camera_key": "text",
        "source": "text",
        "source_uri": "text",
        "source_type": "text",
        "enabled": "boolean",
        "fps": "double precision",
        "total_frames": "bigint",
        "processed_frames": "bigint",
        "frames_processed": "bigint",
        "detections_count": "bigint",
        "frame_width": "integer",
        "frame_height": "integer",
        "metadata": "jsonb",
    },
    "vehicle_tracks": {
        "id": "uuid",
        "run_id": "uuid",
        "camera_id": "uuid",
        "camera_key": "text",
        "local_track_id": "text",
        "tracker_namespace": "text",
        "native_tracker_id": "text",
        "track_status": "text",
        "first_frame": "bigint",
        "last_frame": "bigint",
        "first_seen_seconds": "double precision",
        "last_seen_seconds": "double precision",
        "observation_count": "integer",
        "lost_frames": "integer",
        "vehicle_class": "text",
        "vehicle_class_confidence": "double precision",
        "vehicle_colour": "text",
        "vehicle_colour_status": "text",
        "body_type": "text",
        "body_type_status": "text",
        "plate_text": "text",
        "plate_detected": "boolean",
        "plate_colour": "text",
        "registration_category": "text",
        "final_class_reason": "text",
        "completion_reason": "text",
        "class_counts": "jsonb",
        "class_confidence_sums": "jsonb",
        "enrichment_summary": "jsonb",
        "evidence_record_count": "integer",
        "raw_track": "jsonb",
    },
    "media_assets": {
        "id": "uuid",
        "run_id": "uuid",
        "camera_id": "uuid",
        "track_id": "uuid",
        "media_type": "text",
        "relative_path": "text",
        "frame_number": "bigint",
        "timestamp_seconds": "double precision",
        "width": "integer",
        "height": "integer",
        "storage_provider": "text",
        "bucket": "text",
        "object_key": "text",
        "metadata": "jsonb",
    },
    "track_observations": {
        "track_id": "uuid",
        "run_id": "uuid",
        "camera_id": "uuid",
        "frame_number": "bigint",
        "timestamp_seconds": "double precision",
        "x1": "double precision",
        "y1": "double precision",
        "x2": "double precision",
        "y2": "double precision",
        "bbox_x1": "double precision",
        "bbox_y1": "double precision",
        "bbox_x2": "double precision",
        "bbox_y2": "double precision",
        "confidence": "double precision",
        "detection_confidence": "double precision",
        "raw_class_id": "integer",
        "raw_class_name": "text",
        "tracker_namespace": "text",
        "native_tracker_id": "text",
        "metadata": "jsonb",
    },
    "track_evidence": {
        "id": "uuid",
        "track_id": "uuid",
        "run_id": "uuid",
        "camera_id": "uuid",
        "media_asset_id": "uuid",
        "crop_media_id": "uuid",
        "source_frame_media_id": "uuid",
        "annotated_frame_media_id": "uuid",
        "evidence_role": "text",
        "frame_number": "bigint",
        "timestamp_seconds": "double precision",
        "bbox_x1": "double precision",
        "bbox_y1": "double precision",
        "bbox_x2": "double precision",
        "bbox_y2": "double precision",
        "original_bbox": "jsonb",
        "expanded_crop_bbox": "jsonb",
        "original_bbox_x1": "double precision",
        "original_bbox_y1": "double precision",
        "original_bbox_x2": "double precision",
        "original_bbox_y2": "double precision",
        "expanded_crop_bbox_x1": "double precision",
        "expanded_crop_bbox_y1": "double precision",
        "expanded_crop_bbox_x2": "double precision",
        "expanded_crop_bbox_y2": "double precision",
        "detection_confidence": "double precision",
        "quality_score": "double precision",
        "best_overall_score": "double precision",
        "sharpness_score": "double precision",
        "centeredness_score": "double precision",
        "edge_visibility_score": "double precision",
        "brightness_score": "double precision",
        "crop_width": "integer",
        "crop_height": "integer",
        "resolution_tier": "text",
        "selected_for_colour": "boolean",
        "selected_for_body_type": "boolean",
        "evidence_source": "text",
        "candidate_rank": "integer",
        "metadata": "jsonb",
    },
    "colour_predictions": {
        "id": "uuid",
        "track_id": "uuid",
        "media_asset_id": "uuid",
        "media_id": "uuid",
        "predicted_colour": "text",
        "normalized_colour": "text",
        "confidence": "double precision",
        "source_model": "text",
        "model_name": "text",
        "status": "text",
        "raw_response": "text",
        "prompt": "text",
        "inference_time_ms": "double precision",
        "evidence_frame_number": "bigint",
        "evidence_timestamp_seconds": "double precision",
        "metadata": "jsonb",
    },
    "vehicle_attribute_predictions": {
        "id": "uuid",
        "track_id": "uuid",
        "media_asset_id": "uuid",
        "media_id": "uuid",
        "attribute_type": "text",
        "attribute_value": "text",
        "label": "text",
        "normalized_label": "text",
        "status": "text",
        "confidence": "double precision",
        "source_backend": "text",
        "source_model": "text",
        "raw_response": "text",
        "evidence_frame_number": "bigint",
        "evidence_timestamp_seconds": "double precision",
        "metadata": "jsonb",
    },
    "plate_detections": {
        "id": "uuid",
        "track_id": "uuid",
        "media_asset_id": "uuid",
        "media_id": "uuid",
        "frame_number": "bigint",
        "timestamp_seconds": "double precision",
        "bbox": "jsonb",
        "detection_confidence": "double precision",
        "confidence": "double precision",
        "crop_media_id": "uuid",
        "source_model": "text",
        "status": "text",
        "metadata": "jsonb",
    },
    "plate_readings": {
        "id": "uuid",
        "plate_detection_id": "uuid",
        "track_id": "uuid",
        "raw_text": "text",
        "normalized_text": "text",
        "confidence": "double precision",
        "source_model": "text",
        "model_name": "text",
        "raw_response": "text",
        "status": "text",
        "plate_colour": "text",
        "registration_category": "text",
        "is_selected": "boolean",
        "metadata": "jsonb",
    },
    "physical_vehicles": {
        "id": "uuid",
        "run_id": "uuid",
        "vehicle_key": "text",
        "vehicle_class": "text",
        "vehicle_colour": "text",
        "first_timestamp_seconds": "double precision",
        "last_timestamp_seconds": "double precision",
        "identity_confidence": "double precision",
        "identity_method": "text",
        "identity_status": "text",
        "consensus_plate_text": "text",
        "plate_confidence": "double precision",
        "metadata": "jsonb",
    },
    "physical_vehicle_tracks": {
        "physical_vehicle_id": "uuid",
        "vehicle_track_id": "uuid",
        "association_score": "double precision",
        "association_method": "text",
        "association_reason": "text",
        "metadata": "jsonb",
    },
    "identity_decisions": {
        "id": "uuid",
        "run_id": "uuid",
        "source_track_id": "uuid",
        "target_track_id": "uuid",
        "decision": "text",
        "final_score": "double precision",
        "plate_score": "double precision",
        "spatial_score": "double precision",
        "temporal_score": "double precision",
        "motion_score": "double precision",
        "appearance_score": "double precision",
        "colour_score": "double precision",
        "reason": "text",
        "metadata": "jsonb",
    },
    "pipeline_artifacts": {
        "id": "uuid",
        "run_id": "uuid",
        "artifact_type": "text",
        "relative_path": "text",
        "format": "text",
        "metadata": "jsonb",
    },
    "pipeline_errors": {
        "id": "uuid",
        "run_id": "uuid",
        "camera_id": "uuid",
        "track_id": "uuid",
        "stage": "text",
        "severity": "text",
        "error_code": "text",
        "message": "text",
        "details": "jsonb",
    },
}


@dataclass(frozen=True, slots=True)
class SchemaMismatch:
    kind: str
    table: str
    column: str | None
    expected: str
    actual: str


def check_database_schema(*, config: DatabaseWriteConfig | None = None) -> list[SchemaMismatch]:
    config = config or DatabaseWriteConfig.from_env()
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("DATABASE SCHEMA CONTRACT requires psycopg.") from exc

    with psycopg.connect(config.dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                select table_name, column_name, data_type
                from information_schema.columns
                where table_schema = %s
                """,
                (config.schema,),
            )
            columns: dict[str, dict[str, str]] = {}
            for row in cursor.fetchall():
                columns.setdefault(str(row["table_name"]), {})[str(row["column_name"])] = str(row["data_type"])
            cursor.execute(
                """
                select table_schema, table_name
                from information_schema.tables
                where table_schema = 'public'
                  and table_type = 'BASE TABLE'
                  and table_name = any(%s)
                """,
                (MIRROR_TABLES,),
            )
            public_tables = [str(row["table_name"]) for row in cursor.fetchall()]
            cursor.execute(
                """
                select constraint_name
                from information_schema.table_constraints
                where table_schema = %s
                  and table_name = 'vehicle_tracks'
                  and constraint_type = 'UNIQUE'
                  and constraint_name = 'vehicle_tracks_run_id_camera_id_tracker_namespace_native_tr_key'
                """,
                (config.schema,),
            )
            obsolete_native_tracker_unique = cursor.fetchone() is not None

    mismatches: list[SchemaMismatch] = []
    if config.schema != DEFAULT_DB_SCHEMA:
        mismatches.append(SchemaMismatch("wrong_schema", "*", None, DEFAULT_DB_SCHEMA, config.schema))
    for table, expected_columns in CANONICAL_SCHEMA.items():
        actual_columns = columns.get(table)
        if actual_columns is None:
            mismatches.append(SchemaMismatch("missing_table", table, None, "present", "missing"))
            continue
        for column, expected_type in expected_columns.items():
            actual_type = actual_columns.get(column)
            if actual_type is None:
                mismatches.append(SchemaMismatch("missing_column", table, column, expected_type, "missing"))
            elif actual_type not in TYPE_ALIASES.get(expected_type, {expected_type}):
                mismatches.append(SchemaMismatch("wrong_type", table, column, expected_type, actual_type))
    for table in public_tables:
        mismatches.append(SchemaMismatch("wrong_schema_table", table, None, config.schema, "public"))
    if obsolete_native_tracker_unique:
        mismatches.append(
            SchemaMismatch(
                "obsolete_constraint",
                "vehicle_tracks",
                None,
                "unique(run_id,camera_id,local_track_id) only",
                "unique(run_id,camera_id,tracker_namespace,native_tracker_id)",
            )
        )
    return mismatches


def main() -> int:
    mismatches = check_database_schema()
    if not mismatches:
        print("DATABASE SCHEMA CONTRACT: PASS")
        return 0
    print("DATABASE SCHEMA CONTRACT: FAIL")
    for index, mismatch in enumerate(mismatches, start=1):
        column = f".{mismatch.column}" if mismatch.column else ""
        print(f"[{index}] {mismatch.kind}: {mismatch.table}{column} expected={mismatch.expected} actual={mismatch.actual}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
