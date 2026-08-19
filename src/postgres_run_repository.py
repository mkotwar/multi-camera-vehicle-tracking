from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

from .env_loader import load_env_file
from .plate_text import normalize_plate_text
from .vehicle_analytics import vehicle_records_from_physical_vehicles, vehicle_records_from_repository_tracks
from .vehicle_enrichment.taxonomy import SUPPORTED_VEHICLE_CLASSES, SUPPORTED_VEHICLE_COLOUR_LABELS


DEFAULT_DB_SCHEMA = "vehicle_analytics"
SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PostgresRepositoryConfigurationError(RuntimeError):
    pass


class PostgresRepositoryError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PostgresRunRepositoryConfig:
    dsn: str
    schema: str = DEFAULT_DB_SCHEMA

    def __post_init__(self) -> None:
        _quote_identifier(self.schema)

    @classmethod
    def from_env(cls, env_path: str | Path = ".env") -> "PostgresRunRepositoryConfig":
        values = load_env_file(env_path)
        values.update(os.environ)
        dsn = values.get("DATABASE_URL")
        if not dsn:
            raise PostgresRepositoryConfigurationError("DATA_SOURCE=postgres requires DATABASE_URL.")
        return cls(dsn=dsn, schema=values.get("DB_SCHEMA") or DEFAULT_DB_SCHEMA)


class PostgresRunRepository:
    """Read-only PostgreSQL repository that mirrors the file repository response shape."""

    def __init__(self, config: PostgresRunRepositoryConfig, *, outputs_root: str | Path = "outputs/runs") -> None:
        self.config = config
        self.outputs_root = Path(outputs_root).expanduser().resolve()
        self.schema_sql = _quote_identifier(config.schema)

    def latest_run_id(self) -> str | None:
        runs = self.list_runs()
        if not runs:
            return None
        return str(runs[0]["run_id"])

    def resolve_run_id(self, run_id: str | None = None) -> str | None:
        if run_id is None or str(run_id).strip() == "" or str(run_id).strip().lower() == "latest":
            return self.latest_run_id()
        candidate = str(run_id).strip()
        return candidate if self.get_run(candidate) is not None else None

    def tracks_json_path(self, run_id: str) -> Path | None:
        return None

    def list_runs(self) -> list[dict[str, Any]]:
        sql = f"""
            select
                r.run_key,
                r.status,
                r.started_at,
                r.completed_at,
                r.output_directory,
                r.summary,
                r.metrics,
                (select count(*) from {self._table('run_cameras')} c where c.run_id = r.id) as camera_count,
                (select count(*) from {self._table('vehicle_tracks')} t where t.run_id = r.id) as raw_track_count,
                (select count(*) from {self._table('vehicle_tracks')} t where t.run_id = r.id and upper(t.track_status) = 'COMPLETED') as completed_track_count,
                (select count(*) from {self._table('physical_vehicles')} v where v.run_id = r.id) as physical_vehicle_count
            from {self._table('processing_runs')} r
            order by coalesce(r.completed_at, r.started_at, r.created_at) desc, r.run_key desc
        """
        rows = self._fetchall(sql, ())
        return [self._run_summary(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        sql = f"""
            select
                r.id,
                r.run_key,
                r.status,
                r.started_at,
                r.completed_at,
                r.output_directory,
                r.config_snapshot,
                r.summary,
                r.metrics,
                r.metadata,
                (select count(*) from {self._table('vehicle_tracks')} t where t.run_id = r.id) as raw_track_count,
                (select count(*) from {self._table('vehicle_tracks')} t where t.run_id = r.id and upper(t.track_status) = 'COMPLETED') as completed_track_count,
                (select count(*) from {self._table('physical_vehicles')} v where v.run_id = r.id) as physical_vehicle_count
            from {self._table('processing_runs')} r
            where r.run_key = %s
        """
        rows = self._fetchall(sql, (run_id,))
        if not rows:
            return None
        row = rows[0]
        run_dir = self._run_directory(row)
        metrics = dict(row.get("metrics") or {})
        return {
            "run_id": str(row["run_key"]),
            "summary": dict(row.get("summary") or {}),
            "metadata": dict(row.get("metadata") or {}),
            "detection_tracking_metrics": dict(metrics.get("detection_tracking_metrics") or {}),
            "ingestion_metrics": dict(metrics.get("ingestion_metrics") or {}),
            "evidence_metrics": dict(metrics.get("evidence_metrics") or {}),
            "vehicle_enrichment_metrics": dict(metrics.get("vehicle_enrichment_metrics") or {}),
            "track_count": int(row.get("physical_vehicle_count") or row.get("raw_track_count") or row.get("track_count") or 0),
            "physical_vehicle_count": int(row.get("physical_vehicle_count") or 0),
            "raw_track_count": int(row.get("raw_track_count") or row.get("track_count") or 0),
            "completed_track_count": int(row.get("completed_track_count") or 0),
            "paths": {
                "tracks": str(run_dir / "tracks.json") if run_dir else None,
                "vehicle_enrichment": str(run_dir / "vehicle_enrichment.json") if run_dir else None,
                "evidence_index": str(run_dir / "evidence_index.json") if run_dir else None,
                "track_crop_manifest": str(run_dir / "04_track_crops" / "track_crop_manifest.csv") if run_dir else None,
                "detected_frames": str(run_dir / "detected_frames") if run_dir else None,
                "tracked_frames": str(run_dir / "tracked_frames") if run_dir else None,
            },
        }

    def list_tracks(
        self,
        *,
        run_id: str | None = None,
        camera_id: str | None = None,
        vehicle_class: str | None = None,
        colour: str | None = None,
        track_id: str | None = None,
        status: str | None = None,
        from_time: float | None = None,
        to_time: float | None = None,
    ) -> list[dict[str, Any]]:
        rows = self._track_rows(
            run_id=run_id,
            camera_id=camera_id,
            vehicle_class=vehicle_class,
            colour=colour,
            track_id=track_id,
            status=status,
            from_time=from_time,
            to_time=to_time,
        )
        return [self._track_from_row(row, include_evidence=False) for row in rows]

    def get_track(self, *, camera_id: str, track_id: str, run_id: str | None = None) -> dict[str, Any] | None:
        rows = self._track_rows(run_id=run_id, camera_id=camera_id, track_id=track_id)
        if not rows:
            return None
        return self._track_from_row(rows[0], include_evidence=True)

    def get_track_evidence(self, *, camera_id: str, track_id: str, run_id: str | None = None) -> list[dict[str, Any]]:
        track = self.get_track(camera_id=camera_id, track_id=track_id, run_id=run_id)
        return list(track.get("evidence", []) or []) if track else []

    def list_vehicle_records(self, *, run_id: str | None = None) -> list[Any]:
        try:
            vehicles = self.list_physical_vehicles(run_id=run_id)
        except PostgresRepositoryError:
            vehicles = []
        if vehicles:
            return vehicle_records_from_physical_vehicles(vehicles)
        return vehicle_records_from_repository_tracks(self.list_tracks(run_id=run_id, status="COMPLETED"))

    def list_physical_vehicles(
        self,
        *,
        run_id: str | None = None,
        vehicle_class: str | None = None,
        colour: str | None = None,
        plate_text: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        resolved_run_id = self._resolve_single_run_id(run_id)
        if run_id is not None and resolved_run_id is None:
            return []
        if resolved_run_id is not None:
            clauses.append("r.run_key = %s")
            params.append(resolved_run_id)
        if vehicle_class:
            clauses.append("upper(coalesce(v.vehicle_class, 'UNKNOWN')) = upper(%s)")
            params.append(vehicle_class)
        if colour:
            clauses.append("upper(coalesce(v.vehicle_colour, 'UNKNOWN')) = upper(%s)")
            params.append(colour)
        normalized_plate_text = normalize_plate_text(plate_text)
        if normalized_plate_text:
            clauses.append("regexp_replace(upper(coalesce(v.consensus_plate_text, '')), '[^A-Z0-9]+', '', 'g') = %s")
            params.append(normalized_plate_text)
        where = f"where {' and '.join(clauses)}" if clauses else ""
        sql = f"""
            select
                r.run_key,
                v.id,
                v.vehicle_key,
                v.vehicle_class,
                v.vehicle_colour,
                v.first_timestamp_seconds,
                v.last_timestamp_seconds,
                v.identity_confidence,
                v.identity_method,
                v.identity_status,
                v.consensus_plate_text,
                v.plate_confidence,
                v.metadata,
                array_remove(array_agg(t.local_track_id order by t.local_track_id), null) as member_track_ids,
                array_remove(array_agg(distinct c.camera_key), null) as camera_ids
            from {self._table('physical_vehicles')} v
            join {self._table('processing_runs')} r on r.id = v.run_id
            left join {self._table('physical_vehicle_tracks')} pvt on pvt.physical_vehicle_id = v.id
            left join {self._table('vehicle_tracks')} t on t.id = pvt.vehicle_track_id
            left join {self._table('run_cameras')} c on c.id = t.camera_id
            {where}
            group by r.run_key, v.id
            order by r.run_key desc, v.vehicle_key desc
        """
        rows = self._fetchall(sql, tuple(params))
        vehicles = []
        for row in rows:
            metadata = dict(row.get("metadata") or {})
            member_tracks = [str(item) for item in list(row.get("member_track_ids") or []) if item]
            camera_ids = [str(item) for item in list(row.get("camera_ids") or []) if item]
            vehicles.append(
                {
                    "run_id": str(row["run_key"]),
                    "vehicle_id": str(row.get("vehicle_key") or ""),
                    "vehicle_key": str(row.get("vehicle_key") or ""),
                    "vehicle_class": row.get("vehicle_class"),
                    "vehicle_colour": row.get("vehicle_colour"),
                    "first_seen_seconds": _coerce_float(row.get("first_timestamp_seconds")),
                    "last_seen_seconds": _coerce_float(row.get("last_timestamp_seconds")),
                    "identity_confidence": _coerce_float(row.get("identity_confidence")),
                    "identity_method": row.get("identity_method"),
                    "identity_status": row.get("identity_status"),
                    "consensus_plate_text": row.get("consensus_plate_text"),
                    "plate_confidence": _coerce_float(row.get("plate_confidence")),
                    "member_track_ids": member_tracks,
                    "member_track_count": len(member_tracks),
                    "camera_ids": camera_ids,
                    "primary_camera_id": camera_ids[0] if camera_ids else None,
                    **metadata,
                }
            )
        return vehicles

    def get_physical_vehicle(self, *, vehicle_id: str, run_id: str | None = None) -> dict[str, Any] | None:
        for vehicle in self.list_physical_vehicles(run_id=run_id):
            if str(vehicle.get("vehicle_id")) == vehicle_id or str(vehicle.get("vehicle_key")) == vehicle_id:
                return vehicle
        return None

    def list_cameras(self, *, run_id: str | None = None) -> list[dict[str, Any]]:
        resolved = self._resolve_single_run_id(run_id)
        if resolved is None:
            return []
        sql = f"""
            select
                r.run_key,
                c.camera_key,
                c.source,
                c.source_type,
                c.fps,
                c.total_frames,
                c.processed_frames,
                max(t.last_seen_seconds) as timestamp_seconds,
                max(t.last_frame) as frame_number,
                count(t.id) as active_vehicle_count
            from {self._table('run_cameras')} c
            join {self._table('processing_runs')} r on r.id = c.run_id
            left join {self._table('vehicle_tracks')} t on t.camera_id = c.id
            where r.run_key = %s
            group by r.run_key, c.id, c.camera_key, c.source, c.source_type, c.fps, c.total_frames, c.processed_frames
            order by c.camera_key
        """
        cameras = []
        for row in self._fetchall(sql, (resolved,)):
            camera_id = str(row.get("camera_key") or "")
            frame_number = row.get("frame_number")
            cameras.append(
                {
                    "camera_id": camera_id,
                    "name": camera_id,
                    "status": "completed",
                    "frame_number": int(frame_number) if frame_number is not None else None,
                    "timestamp_seconds": _coerce_float(row.get("timestamp_seconds")),
                    "processed_fps": 0.0,
                    "input_fps": _coerce_float(row.get("fps")),
                    "active_vehicle_count": int(row.get("active_vehicle_count") or 0),
                    "active_track_ids": [],
                    "detections": [],
                    "last_update": _coerce_float(row.get("timestamp_seconds")),
                    "run_id": resolved,
                    "latest_frame_url_parts": self._tracked_frame_relative_parts(run_id=resolved, camera_id=camera_id, frame_number=frame_number),
                    "source_type": row.get("source_type") or "saved_run",
                    "source": row.get("source"),
                }
            )
        return cameras

    def get_filter_options(self, *, run_id: str | None = None) -> dict[str, list[str]]:
        tracks = self.list_tracks(run_id=run_id)
        return {
            "cameras": sorted({str(item.get("camera_id")) for item in tracks if item.get("camera_id")}),
            "vehicle_classes": list(SUPPORTED_VEHICLE_CLASSES),
            "colours": list(SUPPORTED_VEHICLE_COLOUR_LABELS),
            "runs": [str(item["run_id"]) for item in self.list_runs()],
        }

    def resolve_media_path(self, *, run_id: str, category: str, relative_parts: list[str]) -> Path | None:
        run_dir = self._resolve_run_directory(run_id)
        if run_dir is None:
            return None
        base = self._category_base_directory(run_dir, category)
        if base is None:
            return None
        candidate = base
        for part in relative_parts:
            cleaned = str(part).strip()
            if cleaned in {"", ".", ".."} or "/" in cleaned or "\\" in cleaned:
                return None
            candidate = candidate / cleaned
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            resolved = candidate
        try:
            resolved.relative_to(base.resolve())
        except ValueError:
            return None
        return resolved if resolved.exists() and resolved.is_file() else None

    def get_track_reconciliation(self, run_id: str) -> dict[str, Any] | None:
        if self.resolve_run_id(run_id) is None:
            return None
        return {"run_id": run_id, "available": False, "message": "Reconciliation file artifacts are only available in file mode.", "metrics": {}, "config": {}, "tracks": [], "accepted_associations": [], "manual_validation": [], "visual_evidence": [], "paths": {}}

    def get_vehicle_identity_experiment(self, run_id: str) -> dict[str, Any] | None:
        if self.resolve_run_id(run_id) is None:
            return None
        return {"run_id": run_id, "experimental": True, "available": False, "message": "Persistent vehicle identity artifacts are only available in file mode.", "metrics": {}, "analytics_simulation": {}, "config": {}, "calibration": {}, "vehicles": [], "vehicle_id_map": {}, "association_decisions": [], "paths": {}}

    def get_vehicle_identity_summary(self, run_id: str) -> dict[str, Any] | None:
        payload = self.get_vehicle_identity_experiment(run_id)
        if payload is None:
            return None
        return {key: payload[key] for key in ("run_id", "experimental", "available", "message", "metrics", "analytics_simulation", "calibration")}

    def get_stationary_recovery_experiment(self, run_id: str) -> dict[str, Any] | None:
        if self.resolve_run_id(run_id) is None:
            return None
        return {"run_id": run_id, "experimental": True, "stage": "stationary_recovery", "available": False, "message": "Stationary recovery artifacts are only available in file mode.", "metrics": {}, "analytics_simulation": {}, "config": {}, "calibration": {}, "persistent_vehicles": [], "persistent_vehicle_id_map": {}, "recovery_decisions": [], "recovery_scores": [], "paths": {}}

    def get_plate_assisted_identity_experiment(self, run_id: str) -> dict[str, Any] | None:
        if self.resolve_run_id(run_id) is None:
            return None
        return {"run_id": run_id, "experimental": True, "stage": "plate_assisted_identity", "available": False, "message": "Plate-assisted identity artifacts are only available in file mode.", "verification": {}, "plate_coverage": {}, "baseline_without_plate": {}, "plate_assisted": {}, "vehicles": [], "vehicle_id_map": {}, "track_plate_consensus": [], "association_decisions": [], "plate_pair_scores": [], "identity_scores": [], "paths": {}}

    def _track_rows(
        self,
        *,
        run_id: str | None = None,
        camera_id: str | None = None,
        vehicle_class: str | None = None,
        colour: str | None = None,
        track_id: str | None = None,
        status: str | None = None,
        from_time: float | None = None,
        to_time: float | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        resolved_run_id = self._resolve_single_run_id(run_id)
        if run_id is not None and resolved_run_id is None:
            return []
        if resolved_run_id is not None:
            clauses.append("r.run_key = %s")
            params.append(resolved_run_id)
        if camera_id:
            clauses.append("c.camera_key = %s")
            params.append(camera_id)
        if vehicle_class:
            clauses.append("upper(coalesce(t.vehicle_class, 'UNKNOWN')) = upper(%s)")
            params.append(vehicle_class)
        if colour:
            clauses.append("upper(coalesce(t.vehicle_colour, 'UNKNOWN')) = upper(%s)")
            params.append(colour)
        if status:
            clauses.append("upper(coalesce(t.track_status, '')) = upper(%s)")
            params.append(status)
        if track_id:
            clauses.append("(t.local_track_id = %s or split_part(t.local_track_id, ':', 2) = %s)")
            params.extend([track_id, track_id])
        if from_time is not None:
            clauses.append("(t.last_seen_seconds is null or t.last_seen_seconds >= %s)")
            params.append(float(from_time))
        if to_time is not None:
            clauses.append("(t.first_seen_seconds is null or t.first_seen_seconds <= %s)")
            params.append(float(to_time))
        where = f"where {' and '.join(clauses)}" if clauses else ""
        sql = f"""
            select
                r.run_key,
                r.output_directory,
                c.camera_key,
                t.id,
                t.local_track_id,
                t.track_status,
                t.first_frame,
                t.last_frame,
                t.first_seen_seconds,
                t.last_seen_seconds,
                t.observation_count,
                t.completion_reason,
                t.vehicle_class,
                t.vehicle_colour,
                t.vehicle_colour_status,
                t.plate_text,
                t.plate_detected,
                t.plate_colour,
                t.registration_category,
                t.enrichment_summary,
                t.raw_track,
                (
                    select d.detection_confidence
                    from {self._table('plate_detections')} d
                    where d.track_id = t.id
                    order by d.detection_confidence desc nulls last, d.id
                    limit 1
                ) as plate_detection_confidence,
                (
                    select crop.relative_path
                    from {self._table('plate_detections')} d
                    left join {self._table('media_assets')} crop on crop.id = d.media_asset_id
                    where d.track_id = t.id
                    order by d.detection_confidence desc nulls last, d.id
                    limit 1
                ) as plate_crop_path,
                (
                    select d.status
                    from {self._table('plate_detections')} d
                    where d.track_id = t.id
                    order by d.detection_confidence desc nulls last, d.id
                    limit 1
                ) as plate_quality_status,
                (
                    select pr.confidence
                    from {self._table('plate_readings')} pr
                    where pr.track_id = t.id and coalesce(pr.normalized_text, pr.raw_text, '') <> ''
                    order by pr.is_selected desc nulls last, pr.confidence desc nulls last, pr.id
                    limit 1
                ) as plate_text_confidence,
                (
                    select pr.status
                    from {self._table('plate_readings')} pr
                    where pr.track_id = t.id
                    order by pr.is_selected desc nulls last, pr.confidence desc nulls last, pr.id
                    limit 1
                ) as plate_ocr_reason,
                (
                    select crop.relative_path
                    from {self._table('track_evidence')} e
                    left join {self._table('media_assets')} crop on crop.id = e.media_asset_id
                    where e.track_id = t.id
                    order by
                        case e.evidence_role
                            when 'BEST_OVERALL' then 0
                            when 'MIDDLE' then 1
                            when 'LARGEST' then 2
                            when 'FIRST' then 3
                            when 'LAST' then 4
                            else 5
                        end,
                        e.frame_number nulls last
                    limit 1
                ) as best_crop_path
            from {self._table('vehicle_tracks')} t
            join {self._table('processing_runs')} r on r.id = t.run_id
            join {self._table('run_cameras')} c on c.id = t.camera_id
            {where}
            order by r.run_key desc, c.camera_key desc, t.local_track_id desc
        """
        return self._fetchall(sql, tuple(params))

    def _track_from_row(self, row: dict[str, Any], *, include_evidence: bool) -> dict[str, Any]:
        run_id = str(row["run_key"])
        local_track_id = str(row.get("local_track_id") or "")
        first_seen = _coerce_float(row.get("first_seen_seconds"))
        last_seen = _coerce_float(row.get("last_seen_seconds"))
        evidence = self._evidence_for_track(str(row["id"]), run_id=run_id) if include_evidence else []
        best_crop = self._best_crop_from_evidence(evidence) if evidence else row.get("best_crop_path")
        return {
            "run_id": run_id,
            "camera_id": row.get("camera_key"),
            "track_id": self._short_track_id(local_track_id),
            "local_track_id": local_track_id,
            "status": row.get("track_status"),
            "vehicle_class": row.get("vehicle_class"),
            "colour": row.get("vehicle_colour"),
            "colour_status": row.get("vehicle_colour_status"),
            "plate_text": row.get("plate_text"),
            "plate_detected": row.get("plate_detected"),
            "plate_colour": row.get("plate_colour"),
            "registration_category": row.get("registration_category"),
            "plate_detection_confidence": _coerce_float(row.get("plate_detection_confidence")),
            "plate_text_confidence": _coerce_float(row.get("plate_text_confidence")),
            "plate_quality_status": row.get("plate_quality_status"),
            "plate_ocr_reason": row.get("plate_ocr_reason"),
            "plate_crop_path": row.get("plate_crop_path"),
            "plate_crop_parts": self._media_reference_from_relative(run_id=run_id, relative_path=row.get("plate_crop_path")),
            "first_seen": first_seen,
            "last_seen": last_seen,
            "first_seen_seconds": first_seen,
            "last_seen_seconds": last_seen,
            "duration_seconds": self._derive_duration_seconds(first_seen, last_seen),
            "first_frame": row.get("first_frame"),
            "last_frame": row.get("last_frame"),
            "observation_count": row.get("observation_count"),
            "completion_reason": row.get("completion_reason"),
            "vehicle_enrichment_status": (dict(row.get("enrichment_summary") or {}).get("status") if isinstance(row.get("enrichment_summary"), dict) else None),
            "evidence": evidence,
            "best_crop": best_crop,
            "best_crop_parts": self._media_reference_from_relative(run_id=run_id, relative_path=best_crop),
            "available_crop_paths": [item.get("vehicle_crop_path") for item in evidence if item.get("vehicle_crop_path")],
            "colour_resolution": self._colour_resolution(str(row["id"])),
        }

    def _evidence_for_track(self, track_pk: str, *, run_id: str) -> list[dict[str, Any]]:
        sql = f"""
            select
                e.frame_number,
                e.timestamp_seconds,
                e.evidence_role,
                e.detection_confidence,
                e.quality_score,
                e.sharpness_score,
                e.brightness_score,
                e.crop_width,
                e.crop_height,
                e.selected_for_colour,
                e.evidence_source,
                array[e.bbox_x1, e.bbox_y1, e.bbox_x2, e.bbox_y2] as bbox,
                crop.relative_path as crop_path,
                coalesce(annotated.relative_path, source_frame.relative_path) as full_frame_path
            from {self._table('track_evidence')} e
            left join {self._table('media_assets')} crop on crop.id = e.media_asset_id
            left join {self._table('media_assets')} source_frame on source_frame.id = e.source_frame_media_id
            left join {self._table('media_assets')} annotated on annotated.id = e.annotated_frame_media_id
            where e.track_id = %s
            order by e.frame_number
        """
        rows = self._fetchall(sql, (track_pk,))
        evidence: list[dict[str, Any]] = []
        for row in rows:
            crop_path = row.get("crop_path")
            full_frame_path = row.get("full_frame_path")
            evidence.append(
                {
                    "frame_number": row.get("frame_number"),
                    "timestamp_seconds": _coerce_float(row.get("timestamp_seconds")),
                    "vehicle_crop_path": crop_path,
                    "annotated_frame_path": full_frame_path,
                    "bbox_xyxy": row.get("bbox"),
                    "evidence_role": row.get("evidence_role"),
                    "detection_confidence": _coerce_float(row.get("detection_confidence")),
                    "crop_width": row.get("crop_width"),
                    "crop_height": row.get("crop_height"),
                    "sharpness_score": _coerce_float(row.get("sharpness_score")),
                    "brightness_score": _coerce_float(row.get("brightness_score")),
                    "quality_score": _coerce_float(row.get("quality_score")),
                    "evidence_source": row.get("evidence_source"),
                    "selected_for_colour": row.get("selected_for_colour"),
                    "crop_media": self._media_reference_from_relative(run_id=run_id, relative_path=crop_path),
                    "full_frame_media": self._media_reference_from_relative(run_id=run_id, relative_path=full_frame_path),
                }
            )
        return evidence

    def _colour_resolution(self, track_pk: str) -> list[dict[str, Any]]:
        sql = f"""
            select predicted_colour, predicted_colour as normalized_colour, confidence, null as status, evidence_frame_number, metadata
            from {self._table('colour_predictions')}
            where track_id = %s
            order by evidence_frame_number nulls last, id
        """
        rows = self._fetchall(sql, (track_pk,))
        result = []
        for index, row in enumerate(rows, start=1):
            metadata = dict(row.get("metadata") or {})
            result.append(
                {
                    "index": index,
                    "label": row.get("normalized_colour") or row.get("predicted_colour"),
                    "frame_number": row.get("evidence_frame_number"),
                    "evidence_role": metadata.get("evidence_role"),
                    "quality_weight": metadata.get("quality_weight"),
                    "status": row.get("status"),
                    "reason": metadata.get("reason"),
                    "crop_path": metadata.get("source_crop_path"),
                }
            )
        return result

    def _fetchall(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise PostgresRepositoryConfigurationError("DATA_SOURCE=postgres requires psycopg.") from exc
        try:
            with psycopg.connect(self.config.dsn, row_factory=dict_row) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("set transaction read only")
                    cursor.execute(sql, params)
                    return list(cursor.fetchall())
        except Exception as exc:
            raise PostgresRepositoryError(f"PostgreSQL read failed: {exc.__class__.__name__}: {exc}") from exc

    def _table(self, table: str) -> str:
        return f"{self.schema_sql}.{_quote_identifier(table)}"

    def _run_summary(self, row: dict[str, Any]) -> dict[str, Any]:
        summary = dict(row.get("summary") or {})
        metrics = dict(row.get("metrics") or {})
        return {
            "run_id": str(row["run_key"]),
            "status": row.get("status") or "UNKNOWN",
            "start_time": _iso(row.get("started_at")),
            "completed_at": _iso(row.get("completed_at")),
            "camera_count": int(row.get("camera_count") or summary.get("configured_camera_count") or 0),
            "processed_frames": summary.get("processed_frames"),
            "overall_pipeline_runtime_ms": summary.get("overall_pipeline_runtime_ms"),
            "duration_seconds": self._duration_seconds_from_summary(summary),
            "track_count": int(row.get("physical_vehicle_count") or row.get("raw_track_count") or row.get("track_count") or 0),
            "physical_vehicle_count": int(row.get("physical_vehicle_count") or 0),
            "raw_track_count": int(row.get("raw_track_count") or row.get("track_count") or 0),
            "completed_track_count": int(row.get("completed_track_count") or 0),
            "frames_by_camera": summary.get("frames_by_camera", {}),
            "run_directory": row.get("output_directory"),
            "has_run_config": bool(metrics.get("run_config") or summary),
        }

    def _resolve_single_run_id(self, run_id: str | None) -> str | None:
        if run_id is None or str(run_id).strip() == "" or str(run_id).strip().lower() == "all":
            return None
        if str(run_id).strip().lower() == "latest":
            return self.latest_run_id()
        return str(run_id).strip() if self.get_run(str(run_id).strip()) is not None else None

    def _run_directory(self, row: dict[str, Any]) -> Path | None:
        output_directory = row.get("output_directory")
        if output_directory:
            return Path(str(output_directory)).expanduser()
        run_key = row.get("run_key")
        return self.outputs_root / str(run_key) if run_key else None

    def _resolve_run_directory(self, run_id: str) -> Path | None:
        row = self._fetchone_run(run_id)
        if row is None:
            return None
        run_dir = self._run_directory(row)
        if run_dir is None:
            return None
        try:
            resolved = run_dir.resolve()
        except FileNotFoundError:
            resolved = run_dir
        return resolved if resolved.exists() and resolved.is_dir() else None

    def _fetchone_run(self, run_id: str) -> dict[str, Any] | None:
        rows = self._fetchall(
            f"select run_key, output_directory from {self._table('processing_runs')} where run_key = %s",
            (run_id,),
        )
        return rows[0] if rows else None

    def _media_reference_from_relative(self, *, run_id: str, relative_path: Any) -> dict[str, Any] | None:
        if not relative_path:
            return None
        parts = [part for part in str(relative_path).replace("\\", "/").split("/") if part]
        if not parts:
            return None
        category = self._category_from_relative_parts(parts)
        if category is None:
            return None
        category_prefixes = {
            "evidence": ["evidence"],
            "florence_selected_crops": ["05_florence_selected_crops"],
            "track_crops": ["04_track_crops"],
            "body_type_selected_crops": ["07_body_type_selected_crops"],
            "tracked_frames": ["tracked_frames"],
            "detected_frames": ["detected_frames"],
            "raw_frames": ["raw_frames"],
        }
        prefix = category_prefixes.get(category, [])
        media_parts = parts[len(prefix) :] if prefix and parts[: len(prefix)] == prefix else parts
        return {"category": category, "run_id": run_id, "parts": media_parts, "filename": media_parts[-1] if media_parts else None}

    def _category_from_relative_parts(self, parts: list[str]) -> str | None:
        first = parts[0]
        return {
            "evidence": "evidence",
            "05_florence_selected_crops": "florence_selected_crops",
            "04_track_crops": "track_crops",
            "07_body_type_selected_crops": "body_type_selected_crops",
            "tracked_frames": "tracked_frames",
            "detected_frames": "detected_frames",
            "raw_frames": "raw_frames",
        }.get(first)

    def _category_base_directory(self, run_dir: Path, category: str) -> Path | None:
        mapping = {
            "evidence": run_dir / "evidence",
            "florence_selected_crops": run_dir / "05_florence_selected_crops",
            "track_crops": run_dir / "04_track_crops",
            "body_type_selected_crops": run_dir / "07_body_type_selected_crops",
            "tracked_frames": run_dir / "tracked_frames",
            "detected_frames": run_dir / "detected_frames",
            "raw_frames": run_dir / "raw_frames",
        }
        return mapping.get(category)

    def _tracked_frame_relative_parts(self, *, run_id: str, camera_id: str, frame_number: Any) -> list[str] | None:
        if frame_number is None:
            return None
        try:
            frame_index = int(frame_number)
        except (TypeError, ValueError):
            return None
        run_dir = self._resolve_run_directory(run_id)
        if run_dir is None:
            return None
        candidate = run_dir / "tracked_frames" / camera_id / f"frame_{frame_index:06d}.jpg"
        return [camera_id, candidate.name] if candidate.exists() else None

    def _best_crop_from_evidence(self, evidence: list[dict[str, Any]]) -> str | None:
        for item in evidence:
            if item.get("selected_for_colour") and item.get("vehicle_crop_path"):
                return str(item["vehicle_crop_path"])
        return str(evidence[0]["vehicle_crop_path"]) if evidence and evidence[0].get("vehicle_crop_path") else None

    def _short_track_id(self, local_track_id: str) -> str:
        parts = str(local_track_id).split(":")
        return parts[-1] if parts else local_track_id

    def _duration_seconds_from_summary(self, summary: dict[str, Any]) -> float | None:
        runtime_ms = summary.get("overall_pipeline_runtime_ms")
        if runtime_ms is None:
            return None
        try:
            return float(runtime_ms) / 1000.0
        except (TypeError, ValueError):
            return None

    def _derive_duration_seconds(self, first_seen: float | None, last_seen: float | None) -> float | None:
        if first_seen is None or last_seen is None:
            return None
        return max(0.0, float(last_seen) - float(first_seen))


def _quote_identifier(identifier: str) -> str:
    if not SAFE_IDENTIFIER_RE.fullmatch(identifier):
        raise PostgresRepositoryConfigurationError(f"Unsafe PostgreSQL identifier: {identifier!r}")
    return f'"{identifier}"'


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
