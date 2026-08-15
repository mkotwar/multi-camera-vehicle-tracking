from __future__ import annotations

import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .models import ConfigurationError
from .pipeline import _validate_config


@dataclass(slots=True)
class ConfigValidationError:
    rule: str
    path: str
    message: str
    expected: Any | None = None
    actual: Any | None = None


class ConfigServiceError(Exception):
    def __init__(self, message: str, *, status_code: int = 400, errors: list[ConfigValidationError] | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.errors = errors or []


class ConfigService:
    def __init__(self, config_dir: str | Path = "config") -> None:
        self.config_dir = Path(config_dir).expanduser().resolve()

    def list_configs(self) -> dict[str, Any]:
        self._ensure_config_dir()
        configs = []
        preferred = {"default.yaml", "production.yaml", "validation_rectangle_roi.yaml", "validation_rectangle_roi_plate.yaml"}
        for path in sorted(self.config_dir.glob("*.y*ml")):
            configs.append(
                {
                    "config_name": path.name,
                    "path": str(path),
                    "production": path.name == "production.yaml",
                    "preferred": path.name in preferred,
                }
            )
        return {"configs": configs}

    def load_config(self, config_name: str) -> dict[str, Any]:
        path = self._resolve_config_path(config_name, must_exist=True)
        config = normalize_config_for_ui(self._read_yaml(path))
        validation = self.validate_config(config_name, config)
        return {
            "config_name": path.name,
            "path": str(path),
            "config": config,
            "yaml_text": self._dump_yaml(config),
            "validation": validation,
            "inventory": build_config_inventory(config),
        }

    def validate_config(self, config_name: str, config: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_config_path(config_name, must_exist=False)
        config = normalize_config_for_ui(config)
        errors = self._validate_common_fields(config)
        if not errors:
            try:
                _validate_config(config, path)
            except ConfigurationError as exc:
                errors.append(self._error_from_exception(exc))
        warnings = self._build_warnings(path.name, config)
        return {
            "valid": not errors,
            "errors": [asdict(error) for error in errors],
            "warnings": warnings,
        }

    def save_config(self, config_name: str, config: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_config_path(config_name, must_exist=True)
        config = normalize_config_for_ui(config)
        validation = self.validate_config(path.name, config)
        if not validation["valid"]:
            errors = [
                ConfigValidationError(
                    rule=str(error.get("rule") or "config.invalid"),
                    path=str(error.get("path") or "config"),
                    message=str(error.get("message") or "Invalid config."),
                    expected=error.get("expected"),
                    actual=error.get("actual"),
                )
                for error in validation["errors"]
            ]
            raise ConfigServiceError("Configuration validation failed.", status_code=422, errors=errors)
        self._atomic_write_yaml(path, config)
        return {
            "valid": True,
            "errors": [],
            "warnings": validation["warnings"],
            "saved_path": str(path),
            "config_name": path.name,
            "yaml_text": self._dump_yaml(config),
        }

    def clone_config(self, config_name: str, new_name: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        source_path = self._resolve_config_path(config_name, must_exist=True)
        target_path = self._resolve_config_path(new_name, must_exist=False)
        if target_path.exists():
            raise ConfigServiceError(f"Config already exists: {target_path.name}", status_code=409)
        payload = normalize_config_for_ui(config if config is not None else self._read_yaml(source_path))
        validation = self.validate_config(target_path.name, payload)
        if not validation["valid"]:
            errors = [
                ConfigValidationError(
                    rule=str(error.get("rule") or "config.invalid"),
                    path=str(error.get("path") or "config"),
                    message=str(error.get("message") or "Invalid config."),
                    expected=error.get("expected"),
                    actual=error.get("actual"),
                )
                for error in validation["errors"]
            ]
            raise ConfigServiceError("Configuration validation failed.", status_code=422, errors=errors)
        self._atomic_write_yaml(target_path, payload)
        return {
            "valid": True,
            "errors": [],
            "warnings": validation["warnings"],
            "saved_path": str(target_path),
            "config_name": target_path.name,
            "yaml_text": self._dump_yaml(payload),
        }

    def read_roi_preview_frame(
        self,
        config_name: str,
        camera_id: str,
        *,
        frame_number: int | None = None,
    ) -> tuple[bytes, dict[str, str]]:
        path = self._resolve_config_path(config_name, must_exist=True)
        config = normalize_config_for_ui(self._read_yaml(path))
        camera = self._camera_for_config(config, camera_id)
        if not bool(camera.get("enabled", False)):
            raise ConfigServiceError(f"Camera is disabled: {camera_id}", status_code=400)
        source_type = str(camera.get("source_type") or "").strip().lower()
        if source_type != "video":
            raise ConfigServiceError(f"ROI preview supports video sources only; {camera_id} is {source_type or '<empty>'}.", status_code=400)
        source_path = Path(clean_config_string(camera.get("source"))).expanduser()
        if not source_path.is_absolute():
            source_path = (path.parent / source_path).resolve()
        else:
            source_path = source_path.resolve()
        if not source_path.exists():
            raise ConfigServiceError(f"Video source does not exist for {camera_id}: {source_path}", status_code=404)

        try:
            import cv2  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ConfigServiceError("OpenCV is required for ROI preview frames.", status_code=500) from exc

        capture = cv2.VideoCapture(str(source_path))
        try:
            if not capture.isOpened():
                raise ConfigServiceError(f"Unable to open video source for {camera_id}: {source_path}", status_code=400)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            requested_frame = int(frame_number) if frame_number is not None else (min(max(int(frame_count * 0.25), 0), frame_count - 1) if frame_count > 0 else 0)
            if requested_frame > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, requested_frame)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ConfigServiceError(f"Unable to read frame {requested_frame} from {camera_id}.", status_code=400)
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                raise ConfigServiceError("Unable to encode ROI preview frame.", status_code=500)
            height, width = frame.shape[:2]
            headers = {
                "X-Frame-Width": str(width),
                "X-Frame-Height": str(height),
                "X-Frame-Number": str(requested_frame),
                "Cache-Control": "no-store",
            }
            return encoded.tobytes(), headers
        finally:
            capture.release()

    def _ensure_config_dir(self) -> None:
        if not self.config_dir.exists() or not self.config_dir.is_dir():
            raise ConfigServiceError(f"Config directory does not exist: {self.config_dir}", status_code=500)

    def _resolve_config_path(self, config_name: str, *, must_exist: bool) -> Path:
        self._ensure_config_dir()
        raw = str(config_name or "").strip()
        if not raw or raw != Path(raw).name or "\\" in raw or "/" in raw or ".." in raw:
            raise ConfigServiceError("Config name must be a file name inside the config directory.", status_code=400)
        if not raw.lower().endswith((".yaml", ".yml")):
            raise ConfigServiceError("Config name must end with .yaml or .yml.", status_code=400)
        path = (self.config_dir / raw).resolve()
        if path.parent != self.config_dir:
            raise ConfigServiceError("Config path escapes the approved config directory.", status_code=400)
        if must_exist and not path.exists():
            raise ConfigServiceError(f"Config not found: {raw}", status_code=404)
        return path

    def _read_yaml(self, path: Path) -> dict[str, Any]:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ConfigServiceError("Configuration root must be a mapping.", status_code=422)
        return payload

    def _dump_yaml(self, config: dict[str, Any]) -> str:
        return yaml.safe_dump(config, sort_keys=False, allow_unicode=False)

    def _atomic_write_yaml(self, path: Path, config: dict[str, Any]) -> None:
        tmp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=self.config_dir, delete=False, suffix=".tmp") as handle:
                tmp_name = handle.name
                handle.write(self._dump_yaml(config))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        finally:
            if tmp_name and Path(tmp_name).exists():
                Path(tmp_name).unlink()

    def _camera_for_config(self, config: dict[str, Any], camera_id: str) -> dict[str, Any]:
        cameras = list(dict(config.get("input", {}) or {}).get("cameras", []) or [])
        for camera in cameras:
            if isinstance(camera, dict) and str(camera.get("camera_id")) == camera_id:
                return dict(camera)
        raise ConfigServiceError(f"Camera not found in config: {camera_id}", status_code=404)

    def _validate_common_fields(self, config: dict[str, Any]) -> list[ConfigValidationError]:
        errors: list[ConfigValidationError] = []
        for path in (
            "detection.confidence_threshold",
            "detection.iou_threshold",
            "tracking.track_activation_threshold",
            "tracking.minimum_matching_threshold",
            "vehicle_identity.conservative.acceptance_threshold",
            "vehicle_identity.conservative.ambiguity_margin",
            "vehicle_identity.conservative.vehicle_consistency_floor",
        ):
            value = _get_path(config, path)
            if value is not None and not _is_number_between(value, 0.0, 1.0):
                errors.append(ConfigValidationError("range.0_to_1", path, "Must be between 0.0 and 1.0.", "0.0 <= value <= 1.0", value))
        for path, minimum in (
            ("tracking.lost_track_buffer", 0),
            ("tracking.minimum_consecutive_frames", 1),
            ("lifecycle.minimum_observations", 1),
            ("lifecycle.maximum_lost_frames", 0),
            ("evidence.maximum_candidates_per_track", 1),
            ("ingestion.worker_count", 1),
            ("ingestion.frame_queue_size", 1),
        ):
            value = _get_path(config, path)
            if value is not None and not _is_int_at_least(value, minimum):
                errors.append(ConfigValidationError("integer.minimum", path, f"Must be an integer >= {minimum}.", f">= {minimum}", value))
        errors.extend(self._validate_roi(config))
        errors.extend(self._validate_plate_readiness(config))
        return errors

    def _validate_plate_readiness(self, config: dict[str, Any]) -> list[ConfigValidationError]:
        plate, base_path = _active_plate_config(config)
        if not plate:
            return []
        detector = dict(plate.get("detector", {}) or {})
        ocr = dict(plate.get("ocr", {}) or {})
        detector_enabled = bool(detector.get("enabled", plate.get("detection_enabled", False)))
        ocr_enabled = bool(ocr.get("enabled", plate.get("recognition_enabled", False)))
        plate_enabled = bool(plate.get("enabled", detector_enabled or ocr_enabled))
        if not plate_enabled and not detector_enabled and not ocr_enabled:
            return []
        errors: list[ConfigValidationError] = []
        model_path = clean_config_string(detector.get("model_path"))
        if ocr_enabled and not detector_enabled:
            errors.append(
                ConfigValidationError(
                    "plate.detector.required",
                    f"{base_path}.detector.enabled",
                    "Plate OCR requires the plate detector to be enabled.",
                    True,
                    detector.get("enabled"),
                )
            )
        if detector_enabled and not model_path:
            errors.append(
                ConfigValidationError(
                    "plate.detector.model_path_required",
                    f"{base_path}.detector.model_path",
                    "Plate detector model path is required when plate detection/OCR is enabled.",
                    "path to license_plate_weights.pt",
                    model_path,
                )
            )
        elif detector_enabled:
            candidate = Path(model_path).expanduser()
            if not candidate.is_absolute():
                candidate = (self.config_dir / candidate).resolve()
            if not candidate.exists():
                errors.append(
                    ConfigValidationError(
                        "plate.detector.model_path_exists",
                        f"{base_path}.detector.model_path",
                        "Plate detector model path does not exist.",
                        "existing .pt file",
                        str(candidate),
                    )
                )
        return errors

    def _validate_roi(self, config: dict[str, Any]) -> list[ConfigValidationError]:
        roi = config.get("tracking_roi")
        if roi is None:
            return []
        if not isinstance(roi, dict):
            return [ConfigValidationError("mapping", "tracking_roi", "tracking_roi must be a mapping.", "mapping", type(roi).__name__)]
        mode = str(roi.get("mode", "horizontal")).strip().lower()
        errors: list[ConfigValidationError] = []
        if mode not in {"horizontal", "rectangle"}:
            errors.append(ConfigValidationError("enum", "tracking_roi.mode", "Must be horizontal or rectangle.", "horizontal|rectangle", roi.get("mode")))
        if str(roi.get("anchor", "bottom_center")).strip() != "bottom_center":
            errors.append(ConfigValidationError("enum", "tracking_roi.anchor", "Must be bottom_center.", "bottom_center", roi.get("anchor")))
        if mode != "rectangle":
            return errors
        rectangle = roi.get("rectangle")
        if not isinstance(rectangle, dict):
            return [*errors, ConfigValidationError("mapping", "tracking_roi.rectangle", "Rectangle ROI settings are required.", "mapping", type(rectangle).__name__)]
        names = ("x_min_fraction", "y_min_fraction", "x_max_fraction", "y_max_fraction")
        values: dict[str, float] = {}
        for name in names:
            value = rectangle.get(name)
            if not _is_number_between(value, 0.0, 1.0):
                errors.append(
                    ConfigValidationError(
                        "roi.fraction",
                        f"tracking_roi.rectangle.{name}",
                        "Must be a number between 0.0 and 1.0.",
                        "0.0 <= value <= 1.0",
                        value,
                    )
                )
            else:
                values[name] = float(value)
        if len(values) == 4:
            if values["x_min_fraction"] >= values["x_max_fraction"]:
                errors.append(
                    ConfigValidationError(
                        "roi.x_order",
                        "tracking_roi.rectangle.x_min_fraction",
                        "Must be less than x_max_fraction.",
                        "< x_max_fraction",
                        values["x_min_fraction"],
                    )
                )
            if values["y_min_fraction"] >= values["y_max_fraction"]:
                errors.append(
                    ConfigValidationError(
                        "roi.y_order",
                        "tracking_roi.rectangle.y_min_fraction",
                        "Must be less than y_max_fraction.",
                        "< y_max_fraction",
                        values["y_min_fraction"],
                    )
                )
        return errors

    def _error_from_exception(self, exc: ConfigurationError) -> ConfigValidationError:
        message = str(exc)
        path = _extract_error_path(message)
        return ConfigValidationError("pipeline.validation", path, message)

    def _build_warnings(self, config_name: str, config: dict[str, Any]) -> list[dict[str, str]]:
        warnings: list[dict[str, str]] = []
        if config_name == "production.yaml":
            warnings.append({"path": "config_name", "message": "production.yaml changes require explicit operator confirmation."})
        if bool(_get_path(config, "vehicle_identity.stationary_recovery.enabled")):
            warnings.append({"path": "vehicle_identity.stationary_recovery.enabled", "message": "Stationary recovery is experimental and should be validated on a new run."})
        warnings.append({"path": ".env", "message": "Secrets such as DATABASE_URL, passwords, tokens, and API keys are intentionally not editable here."})
        return warnings


def build_config_inventory(config: dict[str, Any]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(f"{prefix}.{key}" if prefix else str(key), child)
            return
        if isinstance(value, list):
            inventory.append(_inventory_row(prefix, "list", value))
            return
        inventory.append(_inventory_row(prefix, _value_type(value), value))

    walk("", config)
    return inventory


def normalize_config_for_ui(config: dict[str, Any]) -> dict[str, Any]:
    payload = _clone_mapping(config)
    cameras = dict(payload.get("input", {}) or {}).get("cameras", [])
    if isinstance(cameras, list):
        for camera in cameras:
            if isinstance(camera, dict) and "source" in camera:
                camera["source"] = clean_config_string(camera.get("source"))
    for path in (
        ("detection", "model_path"),
        ("vehicle_enrichment", "plate", "detector", "model_path"),
        ("vehicle_enrichment", "enrichment", "plate", "detector", "model_path"),
    ):
        current: Any = payload
        for key in path[:-1]:
            current = current.get(key) if isinstance(current, dict) else None
        if isinstance(current, dict) and path[-1] in current:
            current[path[-1]] = clean_config_string(current.get(path[-1]))
    return payload


def clean_config_string(value: Any) -> str:
    text = str(value or "").strip()
    for _ in range(2):
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
            text = text[1:-1].strip()
        else:
            break
    return text


def _active_plate_config(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    vehicle_enrichment = dict(config.get("vehicle_enrichment", {}) or {})
    enrichment = dict(vehicle_enrichment.get("enrichment", {}) or {})
    if isinstance(enrichment.get("plate"), dict):
        return dict(enrichment["plate"]), "vehicle_enrichment.enrichment.plate"
    if isinstance(vehicle_enrichment.get("plate"), dict):
        return dict(vehicle_enrichment["plate"]), "vehicle_enrichment.plate"
    return {}, "vehicle_enrichment.enrichment.plate"


def _clone_mapping(config: dict[str, Any]) -> dict[str, Any]:
    return yaml.safe_load(yaml.safe_dump(config, sort_keys=False, allow_unicode=False)) or {}


def _inventory_row(path: str, value_type: str, value: Any) -> dict[str, Any]:
    return {
        "path": path,
        "type": value_type,
        "default": value,
        "required": _is_required_path(path),
        "restart_required": True,
        "operator_level": _operator_level(path),
        "runtime_effect": _runtime_effect(path),
    }


def _get_path(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if value is None:
        return "null"
    return type(value).__name__


def _is_number_between(value: Any, minimum: float, maximum: float) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return minimum <= numeric <= maximum


def _is_int_at_least(value: Any, minimum: int) -> bool:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return False
    return numeric >= minimum and str(value).strip() == str(numeric)


def _extract_error_path(message: str) -> str:
    match = re.search(r"([a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*)+)", message)
    if match:
        return match.group(1)
    quoted = re.search(r"'([^']+)'", message)
    if quoted:
        return quoted.group(1)
    return "config"


def _is_required_path(path: str) -> bool:
    return path.split(".")[0] in {"project", "input", "ingestion", "detection", "tracking", "visualization", "output"}


def _operator_level(path: str) -> str:
    if any(token in path for token in ("stationary_recovery", "tracking_fix_experiment")):
        return "experimental"
    if any(token in path for token in ("backend", "model_path", "adapter", "worker_count", "queue", "dtype", "device")):
        return "advanced"
    return "safe"


def _runtime_effect(path: str) -> str:
    if path.startswith("tracking_roi"):
        return "Filters detections by bottom-center anchor during new pipeline runs."
    if path.startswith("detection"):
        return "Controls detector loading and filtering for new pipeline runs."
    if path.startswith("tracking"):
        return "Controls ByteTrack association behavior for new pipeline runs."
    if path.startswith("vehicle_identity"):
        return "Controls physical vehicle identity reconciliation for new pipeline runs."
    if "plate" in path or "ocr" in path:
        return "Controls plate detection/OCR enrichment for new pipeline runs."
    return "Applies when the next pipeline run loads this YAML config."
