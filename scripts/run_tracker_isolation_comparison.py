from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.pipeline import _load_raw_config, _validate_config, run_pipeline


MODE_PER_CAMERA = "per_camera"
MODE_PER_CAMERA_CLASS = "per_camera_class"
FOCUS_FRAME_START = 150
FOCUS_FRAME_END = 190


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the per-camera vs per-camera-class ByteTrack comparison.")
    parser.add_argument("--config", default="config.yaml", help="Path to the base YAML configuration file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    raw_config = _load_raw_config(config_path)
    validated_config = _validate_config(raw_config, config_path)
    comparison_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_root = Path("outputs") / "tracker_isolation_comparison" / comparison_id
    comparison_root.mkdir(parents=True, exist_ok=False)

    per_camera_result = _run_mode(config_path, validated_config, comparison_root, MODE_PER_CAMERA)
    per_camera_class_result = _run_mode(config_path, validated_config, comparison_root, MODE_PER_CAMERA_CLASS)

    comparison_json = _build_comparison_json(per_camera_result, per_camera_class_result, comparison_root)
    (comparison_root / "comparison.json").write_text(json.dumps(comparison_json, indent=2), encoding="utf-8")
    (comparison_root / "comparison.md").write_text(_build_comparison_markdown(comparison_json), encoding="utf-8")
    (comparison_root / "run_command.txt").write_text(
        '.\\.venv\\Scripts\\python.exe -m scripts.run_tracker_isolation_comparison --config config.yaml\n',
        encoding="utf-8",
    )
    (comparison_root / "review_checklist.md").write_text(_build_review_checklist(), encoding="utf-8")
    print(json.dumps({"comparison_id": comparison_id, "output_directory": str(comparison_root.resolve())}, indent=2))
    return 0


def _run_mode(
    base_config_path: Path,
    validated_config: dict[str, Any],
    comparison_root: Path,
    isolation_mode: str,
) -> dict[str, Any]:
    mode_directory = comparison_root / isolation_mode
    mode_directory.mkdir(parents=True, exist_ok=True)
    config_override = _build_mode_config(validated_config, isolation_mode)
    with tempfile.TemporaryDirectory(prefix=f"tracker-isolation-{isolation_mode}-") as temp_dir:
        temp_config_path = Path(temp_dir) / f"{isolation_mode}.yaml"
        temp_config_path.write_text(yaml.safe_dump(config_override, sort_keys=False), encoding="utf-8")
        exit_code, run_id, run_directory = run_pipeline(str(temp_config_path))
        if exit_code != 0:
            raise RuntimeError(f"Comparison run failed for isolation_mode={isolation_mode}. See {run_directory}")
    run_dir = Path(run_directory)
    _copy_mode_outputs(run_dir, mode_directory)
    _copy_focus_frames(run_dir / "tracked_frames", mode_directory / "focus_frames")
    return {
        "mode": isolation_mode,
        "run_id": run_id,
        "run_directory": str(run_dir),
        "mode_directory": str(mode_directory.resolve()),
        "tracks": _read_json(run_dir / "tracks.json"),
        "track_lifecycle_metrics": _read_json(run_dir / "track_lifecycle_metrics.json"),
        "detection_tracking_metrics": _read_json(run_dir / "detection_tracking_metrics.json"),
        "observations": _read_observations(run_dir / "observations.csv"),
        "timeline_rows": _extract_timeline_rows(run_dir / "observations.csv"),
    }


def _build_mode_config(validated_config: dict[str, Any], isolation_mode: str) -> dict[str, Any]:
    config = deepcopy(validated_config)
    config["tracking"]["isolation_mode"] = isolation_mode
    config["tracking"]["supported_isolation_modes"] = [MODE_PER_CAMERA, MODE_PER_CAMERA_CLASS]
    config["input"]["max_frames_per_camera"] = max(int(config["input"]["max_frames_per_camera"]), FOCUS_FRAME_END + 1)
    config["visualization"]["tracked_frames"]["enabled"] = True
    config["visualization"]["tracked_frames"]["save_every_n_frames"] = 1
    config["visualization"]["tracked_frames"]["max_saved_frames_per_camera"] = max(
        int(config["visualization"]["tracked_frames"]["max_saved_frames_per_camera"]),
        FOCUS_FRAME_END + 5,
    )
    return config


def _copy_mode_outputs(run_dir: Path, mode_directory: Path) -> None:
    for file_name in (
        "tracks.json",
        "observations.csv",
        "track_lifecycle_metrics.json",
        "detection_tracking_metrics.json",
        "pipeline.log",
    ):
        source = run_dir / file_name
        if source.exists():
            shutil.copy2(source, mode_directory / file_name)
    tracked_frames_source = run_dir / "tracked_frames"
    if tracked_frames_source.exists():
        tracked_frames_target = mode_directory / "tracked_frames"
        if tracked_frames_target.exists():
            shutil.rmtree(tracked_frames_target)
        shutil.copytree(tracked_frames_source, tracked_frames_target)


def _copy_focus_frames(tracked_frames_dir: Path, focus_frames_dir: Path) -> None:
    focus_frames_dir.mkdir(parents=True, exist_ok=True)
    for camera_dir in sorted(path for path in tracked_frames_dir.iterdir() if path.is_dir()):
        target_camera_dir = focus_frames_dir / camera_dir.name
        target_camera_dir.mkdir(parents=True, exist_ok=True)
        for frame_path in sorted(camera_dir.glob("frame_*.jpg")):
            frame_number = int(frame_path.stem.split("_")[-1])
            if FOCUS_FRAME_START <= frame_number <= FOCUS_FRAME_END:
                shutil.copy2(frame_path, target_camera_dir / frame_path.name)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_observations(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _extract_timeline_rows(observations_path: Path) -> list[dict[str, Any]]:
    rows = _read_observations(observations_path)
    timeline = []
    for row in rows:
        frame_number = int(row["frame_number"])
        if FOCUS_FRAME_START <= frame_number <= FOCUS_FRAME_END:
            timeline.append(
                {
                    "camera_id": row["camera_id"],
                    "frame_number": frame_number,
                    "raw_class": row["raw_class_name"],
                    "tracker_namespace": row["tracker_namespace"],
                    "native_tracker_id": int(row["native_tracker_id"]),
                    "local_track_id": row["local_track_id"],
                    "bbox": [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])],
                    "confidence": float(row["confidence"]),
                }
            )
    return timeline


def _build_comparison_json(per_camera: dict[str, Any], per_camera_class: dict[str, Any], comparison_root: Path) -> dict[str, Any]:
    return {
        "comparison_id": comparison_root.name,
        "focus_frame_range": [FOCUS_FRAME_START, FOCUS_FRAME_END],
        "per_camera": _summarize_mode(per_camera),
        "per_camera_class": _summarize_mode(per_camera_class),
        "metric_table": _build_metric_table(per_camera, per_camera_class),
    }


def _summarize_mode(mode_result: dict[str, Any]) -> dict[str, Any]:
    tracks = mode_result["tracks"]
    observations = mode_result["observations"]
    lifecycle_metrics = mode_result["track_lifecycle_metrics"]
    detection_tracking_metrics = mode_result["detection_tracking_metrics"]
    mixed_tracks = [
        track
        for track in tracks
        if "car" in track.get("class_counts", {}) and "motorcycle" in track.get("class_counts", {})
    ]
    unknown_tracks = [track for track in tracks if track.get("final_class") == "UNKNOWN"]
    return {
        "mode": mode_result["mode"],
        "run_id": mode_result["run_id"],
        "run_directory": mode_result["run_directory"],
        "mode_directory": mode_result["mode_directory"],
        "tracker_instances_created": detection_tracking_metrics.get("tracker_instances_created_total", 0),
        "trackers_created_by_camera": detection_tracking_metrics.get("trackers_created_by_camera", {}),
        "trackers_created_by_camera_namespace": detection_tracking_metrics.get("trackers_created_by_camera_namespace", {}),
        "local_tracks": len(tracks),
        "mixed_car_motorcycle_tracks": len(mixed_tracks),
        "unknown_final_classes": len(unknown_tracks),
        "car_observations": len([row for row in observations if row["raw_class_name"] == "car"]),
        "motorcycle_observations": len([row for row in observations if row["raw_class_name"] == "motorcycle"]),
        "lost_observations": max(0, sum(detection_tracking_metrics.get("detections_by_camera", {}).values()) - len(observations)),
        "duplicate_observations": lifecycle_metrics.get("duplicate_observation_count", 0),
        "average_tracking_time_frame_ms": detection_tracking_metrics.get("average_inference_time_ms", 0.0),
        "timeline_rows": mode_result["timeline_rows"],
        "tracks": tracks,
    }


def _build_metric_table(per_camera: dict[str, Any], per_camera_class: dict[str, Any]) -> dict[str, dict[str, Any]]:
    a = _summarize_mode(per_camera)
    b = _summarize_mode(per_camera_class)
    return {
        "Tracker instances": {"per_camera": a["tracker_instances_created"], "per_camera_class": b["tracker_instances_created"]},
        "Local tracks": {"per_camera": a["local_tracks"], "per_camera_class": b["local_tracks"]},
        "Mixed CAR/MOTORCYCLE tracks": {"per_camera": a["mixed_car_motorcycle_tracks"], "per_camera_class": b["mixed_car_motorcycle_tracks"]},
        "UNKNOWN final classes": {"per_camera": a["unknown_final_classes"], "per_camera_class": b["unknown_final_classes"]},
        "CAR observations": {"per_camera": a["car_observations"], "per_camera_class": b["car_observations"]},
        "MOTORCYCLE observations": {"per_camera": a["motorcycle_observations"], "per_camera_class": b["motorcycle_observations"]},
        "Lost observations": {"per_camera": a["lost_observations"], "per_camera_class": b["lost_observations"]},
        "Duplicate observations": {"per_camera": a["duplicate_observations"], "per_camera_class": b["duplicate_observations"]},
        "Average tracking time/frame": {
            "per_camera": a["average_tracking_time_frame_ms"],
            "per_camera_class": b["average_tracking_time_frame_ms"],
        },
    }


def _build_comparison_markdown(comparison_json: dict[str, Any]) -> str:
    lines = ["# Tracker Isolation Comparison", ""]
    lines.append(f"Focus frames: `{FOCUS_FRAME_START}-{FOCUS_FRAME_END}`")
    lines.append("")
    lines.append("| Metric | Per camera | Per camera + class |")
    lines.append("| --- | ---: | ---: |")
    for metric_name, values in comparison_json["metric_table"].items():
        lines.append(f"| {metric_name} | {values['per_camera']} | {values['per_camera_class']} |")
    lines.append("")
    for mode_name in ("per_camera", "per_camera_class"):
        mode = comparison_json[mode_name]
        lines.append(f"## {mode_name}")
        lines.append("")
        lines.append(f"- Run ID: `{mode['run_id']}`")
        lines.append(f"- Tracker instances created: `{mode['tracker_instances_created']}`")
        lines.append(f"- Mixed CAR/MOTORCYCLE tracks: `{mode['mixed_car_motorcycle_tracks']}`")
        lines.append(f"- UNKNOWN final classes: `{mode['unknown_final_classes']}`")
        lines.append("")
        lines.append("### Timeline 150-190")
        lines.append("")
        lines.append("| Camera | Frame | Raw class | Namespace | Native ID | Local track ID | Confidence |")
        lines.append("| --- | ---: | --- | --- | ---: | --- | ---: |")
        for row in mode["timeline_rows"]:
            lines.append(
                f"| {row['camera_id']} | {row['frame_number']} | {row['raw_class']} | {row['tracker_namespace']} | {row['native_tracker_id']} | {row['local_track_id']} | {row['confidence']:.2f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _build_review_checklist() -> str:
    return (
        "# Review Checklist\n\n"
        "- Open `comparison.md`\n"
        "- Inspect `per_camera/focus_frames/`\n"
        "- Inspect `per_camera_class/focus_frames/`\n"
        "- Verify the white car and motorcycle do not share one local track in `per_camera_class`\n"
        "- Verify mixed CAR/MOTORCYCLE tracks are reduced or removed\n"
        "- Verify no observations are lost or duplicated\n"
        "- Verify tracker namespaces appear in `tracks.json` and `observations.csv`\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
