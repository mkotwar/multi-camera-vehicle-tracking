from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .vehicle_enrichment.taxonomy import SUPPORTED_VEHICLE_CLASSES, SUPPORTED_VEHICLE_COLOUR_LABELS


class RunRepository:
    def __init__(self, outputs_root: str | Path) -> None:
        self.outputs_root = Path(outputs_root).expanduser().resolve()

    def latest_run_id(self) -> str | None:
        runs = self.list_runs()
        if not runs:
            return None
        return str(runs[0]["run_id"])

    def list_runs(self) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for run_dir in self._iter_run_directories():
            summary = self._read_json(run_dir / "summary.json", default={})
            metadata = self._read_json(run_dir / "run_metadata.json", default={})
            run_config = self._read_yaml_text(run_dir / "run_config.yaml")
            tracks = self._read_json(run_dir / "tracks.json", default=[])
            camera_count = summary.get("configured_camera_count")
            if camera_count is None:
                camera_count = metadata.get("camera_count")
            runs.append(
                {
                    "run_id": run_dir.name,
                    "status": summary.get("status") or metadata.get("status") or "UNKNOWN",
                    "start_time": metadata.get("started_at"),
                    "completed_at": metadata.get("completed_at"),
                    "camera_count": camera_count,
                    "processed_frames": summary.get("processed_frames"),
                    "overall_pipeline_runtime_ms": summary.get("overall_pipeline_runtime_ms"),
                    "duration_seconds": self._duration_seconds_from_summary(summary),
                    "track_count": len([item for item in tracks if isinstance(item, dict)]),
                    "frames_by_camera": summary.get("frames_by_camera", {}),
                    "run_directory": str(run_dir),
                    "has_run_config": run_config is not None,
                }
            )
        runs.sort(key=lambda item: str(item["run_id"]), reverse=True)
        return runs

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        run_dir = self._resolve_run_directory(run_id)
        if run_dir is None:
            return None
        summary = self._read_json(run_dir / "summary.json", default={})
        metadata = self._read_json(run_dir / "run_metadata.json", default={})
        detection = self._read_json(run_dir / "detection_tracking_metrics.json", default={})
        ingestion = self._read_json(run_dir / "ingestion_metrics.json", default={})
        evidence = self._read_json(run_dir / "evidence_metrics.json", default={})
        enrichment = self._read_json(run_dir / "vehicle_enrichment_metrics.json", default={})
        tracks = self._read_json(run_dir / "tracks.json", default=[])
        return {
            "run_id": run_id,
            "summary": summary,
            "metadata": metadata,
            "detection_tracking_metrics": detection,
            "ingestion_metrics": ingestion,
            "evidence_metrics": evidence,
            "vehicle_enrichment_metrics": enrichment,
            "track_count": len([item for item in tracks if isinstance(item, dict)]),
            "paths": {
                "tracks": str(run_dir / "tracks.json"),
                "vehicle_enrichment": str(run_dir / "vehicle_enrichment.json"),
                "evidence_index": str(run_dir / "evidence_index.json"),
                "track_crop_manifest": str(run_dir / "04_track_crops" / "track_crop_manifest.csv"),
                "detected_frames": str(run_dir / "detected_frames"),
                "tracked_frames": str(run_dir / "tracked_frames"),
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
        from_time: float | None = None,
        to_time: float | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        run_ids = self._resolve_run_ids(run_id)
        for candidate_run_id in run_ids:
            rows.extend(
                self._load_run_tracks(
                    run_id=candidate_run_id,
                    camera_id=camera_id,
                    vehicle_class=vehicle_class,
                    colour=colour,
                    track_id=track_id,
                    from_time=from_time,
                    to_time=to_time,
                )
            )
        rows.sort(key=lambda item: (str(item.get("run_id", "")), str(item.get("camera_id", "")), str(item.get("track_id", ""))), reverse=True)
        return rows

    def get_track(self, *, camera_id: str, track_id: str, run_id: str | None = None) -> dict[str, Any] | None:
        run_ids = self._resolve_run_ids(run_id)
        for candidate_run_id in run_ids:
            for item in self._load_run_tracks(run_id=candidate_run_id):
                if str(item.get("camera_id")) != camera_id:
                    continue
                if str(item.get("track_id")) == track_id or str(item.get("local_track_id")) == track_id:
                    return item
        return None

    def get_track_evidence(self, *, camera_id: str, track_id: str, run_id: str | None = None) -> list[dict[str, Any]]:
        track = self.get_track(camera_id=camera_id, track_id=track_id, run_id=run_id)
        if track is None:
            return []
        return list(track.get("evidence", []) or [])

    def list_cameras(self, *, run_id: str | None = None) -> list[dict[str, Any]]:
        resolved_run_id = self._resolve_single_run_id(run_id)
        if resolved_run_id is None:
            return []
        tracks = self._load_run_tracks(run_id=resolved_run_id)
        by_camera: dict[str, dict[str, Any]] = {}
        for item in tracks:
            camera_id = str(item.get("camera_id", "") or "")
            if not camera_id:
                continue
            camera = by_camera.setdefault(
                camera_id,
                {
                    "camera_id": camera_id,
                    "name": camera_id,
                    "status": "completed",
                    "frame_number": None,
                    "timestamp_seconds": None,
                    "processed_fps": 0.0,
                    "input_fps": None,
                    "active_vehicle_count": 0,
                    "active_track_ids": [],
                    "detections": [],
                    "last_update": None,
                    "run_id": resolved_run_id,
                    "latest_frame_url_parts": None,
                    "source_type": "saved_run",
                    "source": str(self._resolve_run_directory(resolved_run_id) or ""),
                },
            )
            camera["active_vehicle_count"] = int(camera["active_vehicle_count"]) + 1
            camera["active_track_ids"].append(str(item.get("track_id") or item.get("local_track_id") or ""))
            last_seen = item.get("last_seen_seconds")
            last_frame = item.get("last_frame")
            if last_seen is not None and (camera["timestamp_seconds"] is None or float(last_seen) >= float(camera["timestamp_seconds"])):
                camera["timestamp_seconds"] = float(last_seen)
                camera["frame_number"] = int(last_frame) if last_frame is not None else None
                camera["last_update"] = float(last_seen)
                camera["latest_frame_url_parts"] = self._tracked_frame_relative_parts(
                    run_id=resolved_run_id,
                    camera_id=camera_id,
                    frame_number=last_frame,
                )
        return sorted(by_camera.values(), key=lambda item: str(item["camera_id"]))

    def get_filter_options(self, *, run_id: str | None = None) -> dict[str, list[str]]:
        tracks = self.list_tracks(run_id=run_id)
        cameras = sorted({str(item.get("camera_id")) for item in tracks if item.get("camera_id")})
        runs = [str(item["run_id"]) for item in self.list_runs()]
        return {
            "cameras": cameras,
            "vehicle_classes": list(SUPPORTED_VEHICLE_CLASSES),
            "colours": list(SUPPORTED_VEHICLE_COLOUR_LABELS),
            "runs": runs,
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
        if not self._is_relative_to(resolved, base.resolve()):
            return None
        if not resolved.exists() or not resolved.is_file():
            return None
        return resolved

    def _load_run_tracks(
        self,
        *,
        run_id: str,
        camera_id: str | None = None,
        vehicle_class: str | None = None,
        colour: str | None = None,
        track_id: str | None = None,
        from_time: float | None = None,
        to_time: float | None = None,
    ) -> list[dict[str, Any]]:
        run_dir = self._resolve_run_directory(run_id)
        if run_dir is None:
            return []
        tracks = self._read_json(run_dir / "tracks.json", default=[])
        enrichments = self._read_json(run_dir / "vehicle_enrichment.json", default=[])
        enrichment_by_track = {
            str(item.get("local_track_id")): item
            for item in enrichments
            if isinstance(item, dict) and item.get("local_track_id")
        }
        results: list[dict[str, Any]] = []
        for item in tracks:
            if not isinstance(item, dict):
                continue
            local_track_id = str(item.get("local_track_id", ""))
            if not local_track_id:
                continue
            short_track_id = self._short_track_id(local_track_id)
            enrichment = enrichment_by_track.get(local_track_id, {})
            colour_payload = dict(enrichment.get("vehicle_colour", {}) or {})
            colour_label = colour_payload.get("label")
            vehicle_class_value = enrichment.get("vehicle_class") or item.get("final_class")
            first_seen_seconds = self._coerce_float(item.get("first_timestamp_seconds"))
            last_seen_seconds = self._coerce_float(item.get("last_timestamp_seconds"))
            evidence_rows = [self._normalize_evidence_item(run_id=run_id, payload=row) for row in list(enrichment.get("evidence_used", []) or [])]
            record = {
                "run_id": run_id,
                "camera_id": item.get("camera_id"),
                "track_id": short_track_id,
                "local_track_id": local_track_id,
                "status": item.get("status"),
                "vehicle_class": vehicle_class_value,
                "colour": colour_label,
                "colour_status": colour_payload.get("status"),
                "first_seen": first_seen_seconds,
                "last_seen": last_seen_seconds,
                "first_seen_seconds": first_seen_seconds,
                "last_seen_seconds": last_seen_seconds,
                "duration_seconds": self._derive_duration_seconds(first_seen_seconds, last_seen_seconds),
                "first_frame": item.get("first_frame"),
                "last_frame": item.get("last_frame"),
                "observation_count": item.get("observation_count"),
                "completion_reason": item.get("completion_reason"),
                "vehicle_enrichment_status": enrichment.get("status"),
                "evidence": evidence_rows,
                "best_crop": self._best_crop_from_enrichment(enrichment),
                "best_crop_parts": self._path_to_media_reference(run_id=run_id, path_value=self._best_crop_from_enrichment(enrichment)),
                "available_crop_paths": list(enrichment.get("selected_crop_paths", []) or []),
                "colour_resolution": self._build_colour_resolution(colour_payload),
                "raw_track": item,
                "raw_enrichment": enrichment,
            }
            if camera_id and str(record["camera_id"]) != camera_id:
                continue
            if vehicle_class and str(record["vehicle_class"]).upper() != str(vehicle_class).upper():
                continue
            if colour and str(record["colour"]).upper() != str(colour).upper():
                continue
            if track_id and str(record["track_id"]) != track_id and str(record["local_track_id"]) != track_id:
                continue
            if from_time is not None and record["last_seen_seconds"] is not None and float(record["last_seen_seconds"]) < float(from_time):
                continue
            if to_time is not None and record["first_seen_seconds"] is not None and float(record["first_seen_seconds"]) > float(to_time):
                continue
            results.append(record)
        return results

    def _best_crop_from_enrichment(self, enrichment: dict[str, Any]) -> str | None:
        evidence = list(enrichment.get("evidence_used", []) or [])
        for item in evidence:
            if item.get("selected_for_colour") and item.get("vehicle_crop_path"):
                return str(item["vehicle_crop_path"])
        crop_paths = list(enrichment.get("selected_crop_paths", []) or [])
        return str(crop_paths[0]) if crop_paths else None

    def _iter_run_directories(self) -> list[Path]:
        if not self.outputs_root.exists():
            return []
        return [item for item in self.outputs_root.iterdir() if item.is_dir()]

    def _resolve_run_ids(self, run_id: str | None) -> list[str]:
        if run_id is None or str(run_id).strip() == "" or str(run_id).strip().lower() == "all":
            return [str(item["run_id"]) for item in self.list_runs()]
        if str(run_id).strip().lower() == "latest":
            latest = self.latest_run_id()
            return [latest] if latest else []
        return [str(run_id)]

    def _resolve_single_run_id(self, run_id: str | None) -> str | None:
        run_ids = self._resolve_run_ids(run_id)
        return run_ids[0] if run_ids else None

    def _resolve_run_directory(self, run_id: str) -> Path | None:
        candidate = (self.outputs_root / str(run_id)).resolve()
        if not candidate.exists() or not candidate.is_dir():
            return None
        if not self._is_relative_to(candidate, self.outputs_root.resolve()):
            return None
        return candidate

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

    def _read_json(self, path: Path, *, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    def _read_yaml_text(self, path: Path) -> str | None:
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return None

    def _is_relative_to(self, path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

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

    def _coerce_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _derive_duration_seconds(self, first_seen: float | None, last_seen: float | None) -> float | None:
        if first_seen is None or last_seen is None:
            return None
        return max(0.0, float(last_seen) - float(first_seen))

    def _build_colour_resolution(self, colour_payload: dict[str, Any]) -> list[dict[str, Any]]:
        predictions = list(colour_payload.get("predictions", []) or [])
        rows: list[dict[str, Any]] = []
        for index, item in enumerate(predictions, start=1):
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "index": index,
                    "label": item.get("label"),
                    "frame_number": item.get("source_frame_number"),
                    "evidence_role": item.get("evidence_role"),
                    "quality_weight": item.get("quality_weight"),
                    "status": item.get("status"),
                    "reason": item.get("reason"),
                    "crop_path": item.get("source_crop_path"),
                }
            )
        return rows

    def _normalize_evidence_item(self, *, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        item = dict(payload)
        item["timestamp_seconds"] = self._coerce_float(item.get("timestamp_seconds"))
        item["crop_media"] = self._path_to_media_reference(run_id=run_id, path_value=item.get("vehicle_crop_path"))
        item["full_frame_media"] = self._path_to_media_reference(run_id=run_id, path_value=item.get("annotated_frame_path") or item.get("source_image_path"))
        return item

    def _path_to_media_reference(self, *, run_id: str, path_value: Any) -> dict[str, Any] | None:
        if not path_value:
            return None
        run_dir = self._resolve_run_directory(run_id)
        if run_dir is None:
            return None
        try:
            path = Path(str(path_value)).resolve()
        except Exception:
            return None
        category_roots = {
            "evidence": run_dir / "evidence",
            "florence_selected_crops": run_dir / "05_florence_selected_crops",
            "track_crops": run_dir / "04_track_crops",
            "body_type_selected_crops": run_dir / "07_body_type_selected_crops",
            "tracked_frames": run_dir / "tracked_frames",
            "detected_frames": run_dir / "detected_frames",
            "raw_frames": run_dir / "raw_frames",
        }
        for category, root in category_roots.items():
            try:
                relative = path.relative_to(root.resolve())
            except ValueError:
                continue
            parts = [item for item in relative.parts if item]
            return {
                "category": category,
                "run_id": run_id,
                "parts": parts,
                "filename": parts[-1] if parts else None,
            }
        return None

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
        if candidate.exists():
            return [camera_id, candidate.name]
        return None


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
