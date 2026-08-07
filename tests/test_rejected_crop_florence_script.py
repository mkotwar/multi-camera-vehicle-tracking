from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "test_rejected_crops_with_florence.py"
_SPEC = importlib.util.spec_from_file_location("test_rejected_crops_with_florence_module", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class _FakeBackend:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = list(responses)
        self.adapter_active = False
        self.resolved_device = "cuda:0"
        self.resolved_dtype = "float16"
        self.load_calls = 0
        self.close_calls = 0
        self.run_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def close(self) -> None:
        self.close_calls += 1

    def run_task(self, image, task_prompt, text_input=None, *, adapter_active=None, generation_overrides=None):
        response = self.responses[self.run_calls]
        self.run_calls += 1
        assert adapter_active is False
        assert task_prompt == "<VQA>"
        assert text_input == "What colour is the vehicle?"
        return response


def _write_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "vehicle_enrichment:",
                "  shared_florence:",
                "    enabled: true",
                "    backend: florence2",
                "    base_model_id: D:/project/models/Florence-2-base-ft",
                "    processor_path: D:/project/models/Florence-2-base-ft",
                "    adapter_path: D:/project/models/OCR_MUKUL/OCR_MUKUL/adaptor_florance_baseFT",
                "    adapter_enabled: true",
                "    device: auto",
                "    dtype: auto",
                "    trust_remote_code: true",
                "    attention_implementation: eager",
                "    max_new_tokens: 64",
                "    num_beams: 1",
                "    use_cache: false",
                "    local_files_only: true",
                "    lazy_load: true",
                "  vehicle_attributes:",
                "    enabled: true",
                "    task_token: \"<VQA>\"",
                "    prompt: \"What colour is the vehicle?\"",
                "    florence:",
                "      adapter_enabled: false",
                "      adapter_path: \"\"",
                "      max_new_tokens: 16",
                "      num_beams: 1",
                "      use_cache: true",
                "    colour:",
                "      enabled: true",
                "      task_token: \"<VQA>\"",
                "      prompt: \"What colour is the vehicle?\"",
                "      generation:",
                "        max_new_tokens: 16",
                "        num_beams: 1",
                "        do_sample: false",
                "        use_cache: true",
                "        early_stopping: false",
            ]
        ),
        encoding="utf-8",
    )


def _write_crop(path: Path, width: int, height: int) -> None:
    image = np.full((height, width, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
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
            writer.writerow(row)


def test_rejected_crop_diagnostic_preserves_flags_and_saves_raw_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "runs" / "20260807_134127"
    manifest_dir = run_dir / "04_track_crops"
    track_dir = manifest_dir / "CAM_001" / "TRACK_192"
    track_dir.mkdir(parents=True)
    crop1 = track_dir / "frame_001152.jpg"
    crop2 = track_dir / "frame_001155.jpg"
    _write_crop(crop1, 79, 101)
    _write_crop(crop2, 56, 123)
    manifest_path = manifest_dir / "track_crop_manifest.csv"
    _write_manifest(
        manifest_path,
        [
            {
                "camera_id": "CAM_001",
                "local_track_id": "CAM_001:TRACK_192",
                "frame_number": 1152,
                "timestamp_seconds": 38.4,
                "vehicle_class": "motorcycle",
                "confidence": 0.91,
                "bbox_x1": 0,
                "bbox_y1": 0,
                "bbox_x2": 10,
                "bbox_y2": 10,
                "crop_width": 79,
                "crop_height": 101,
                "crop_path": str(crop1),
                "trigger_y": 500,
                "inside_capture_zone": True,
                "capture_zone_top": 432,
                "capture_zone_bottom": 590,
                "evidence_eligible": False,
                "evidence_rejection_reason": "width_below_motorcycle_minimum",
                "florence_eligible": False,
                "florence_rejection_reason": "width_below_motorcycle_florence_minimum",
            },
            {
                "camera_id": "CAM_001",
                "local_track_id": "CAM_001:TRACK_192",
                "frame_number": 1155,
                "timestamp_seconds": 38.5,
                "vehicle_class": "motorcycle",
                "confidence": 0.92,
                "bbox_x1": 0,
                "bbox_y1": 0,
                "bbox_x2": 10,
                "bbox_y2": 10,
                "crop_width": 56,
                "crop_height": 123,
                "crop_path": str(crop2),
                "trigger_y": 520,
                "inside_capture_zone": True,
                "capture_zone_top": 432,
                "capture_zone_bottom": 590,
                "evidence_eligible": False,
                "evidence_rejection_reason": "width_below_motorcycle_minimum",
                "florence_eligible": False,
                "florence_rejection_reason": "width_below_motorcycle_florence_minimum",
            },
        ],
    )
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    backend = _FakeBackend(
        [
            {
                "status": "completed",
                "reason": None,
                "payload": {"generated_text": "black", "parsed_answer": {"<VQA>": "black"}, "inference_duration_ms": 7.0},
            },
            {
                "status": "completed",
                "reason": None,
                "payload": {"generated_text": "The vehicle appears blue.", "parsed_answer": {"<VQA>": "The vehicle appears blue."}, "inference_duration_ms": 8.0},
            },
        ]
    )

    result = _MODULE.run_rejected_crop_diagnostic(
        run_dir=run_dir,
        config_path=config_path,
        track_id="CAM_001:TRACK_192",
        sample_other_motorcycles_count=0,
        backend=backend,
        command_text="python scripts/test_rejected_crops_with_florence.py --track-id CAM_001:TRACK_192",
    )

    assert backend.load_calls == 1
    assert backend.run_calls == 2
    assert result["adapter_loaded"] is False
    assert result["track_192_results"][0]["normal_evidence_eligible"] is False
    assert result["track_192_results"][0]["normal_florence_eligible"] is False
    assert result["track_192_results"][0]["eligibility_bypassed"] is True
    assert result["track_192_results"][0]["raw_response"] == "black"
    assert result["track_192_results"][0]["parsed_colour"] == "BLACK"
    assert result["track_192_results"][1]["raw_response"] == "The vehicle appears blue."
    assert result["track_192_results"][1]["parsed_colour"] == "BLUE"
    assert Path(result["csv_path"]).exists()
    assert Path(result["json_path"]).exists()
    assert (run_dir / "diagnostics" / "rejected_crop_florence_test" / "CAM_001" / "TRACK_192" / "frame_001152.jpg").exists()

    saved_json = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    assert saved_json["results"][0]["raw_response"] == "black"


def test_inference_failure_does_not_abort_remaining_crops(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "runs" / "20260807_134127"
    manifest_dir = run_dir / "04_track_crops"
    track_dir = manifest_dir / "CAM_001" / "TRACK_192"
    track_dir.mkdir(parents=True)
    crop1 = track_dir / "frame_001152.jpg"
    crop2 = track_dir / "frame_001155.jpg"
    _write_crop(crop1, 79, 101)
    _write_crop(crop2, 56, 123)
    _write_manifest(
        manifest_dir / "track_crop_manifest.csv",
        [
            {
                "camera_id": "CAM_001",
                "local_track_id": "CAM_001:TRACK_192",
                "frame_number": 1152,
                "timestamp_seconds": 38.4,
                "vehicle_class": "motorcycle",
                "confidence": 0.91,
                "bbox_x1": 0,
                "bbox_y1": 0,
                "bbox_x2": 10,
                "bbox_y2": 10,
                "crop_width": 79,
                "crop_height": 101,
                "crop_path": str(crop1),
                "trigger_y": 500,
                "inside_capture_zone": True,
                "capture_zone_top": 432,
                "capture_zone_bottom": 590,
                "evidence_eligible": False,
                "evidence_rejection_reason": "width_below_motorcycle_minimum",
                "florence_eligible": False,
                "florence_rejection_reason": "width_below_motorcycle_florence_minimum",
            },
            {
                "camera_id": "CAM_001",
                "local_track_id": "CAM_001:TRACK_192",
                "frame_number": 1155,
                "timestamp_seconds": 38.5,
                "vehicle_class": "motorcycle",
                "confidence": 0.92,
                "bbox_x1": 0,
                "bbox_y1": 0,
                "bbox_x2": 10,
                "bbox_y2": 10,
                "crop_width": 56,
                "crop_height": 123,
                "crop_path": str(crop2),
                "trigger_y": 520,
                "inside_capture_zone": True,
                "capture_zone_top": 432,
                "capture_zone_bottom": 590,
                "evidence_eligible": False,
                "evidence_rejection_reason": "width_below_motorcycle_minimum",
                "florence_eligible": False,
                "florence_rejection_reason": "width_below_motorcycle_florence_minimum",
            },
        ],
    )
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    backend = _FakeBackend(
        [
            {"status": "error", "reason": "mock failure", "payload": None},
            {
                "status": "completed",
                "reason": None,
                "payload": {"generated_text": "white", "parsed_answer": {"<VQA>": "white"}, "inference_duration_ms": 9.0},
            },
        ]
    )

    result = _MODULE.run_rejected_crop_diagnostic(
        run_dir=run_dir,
        config_path=config_path,
        track_id="CAM_001:TRACK_192",
        sample_other_motorcycles_count=0,
        backend=backend,
        command_text="python script.py",
    )

    assert backend.run_calls == 2
    assert result["track_192_results"][0]["status"] == "error"
    assert result["track_192_results"][0]["error"] == "mock failure"
    assert result["track_192_results"][1]["status"] == "completed"
    assert result["track_192_results"][1]["parsed_colour"] == "WHITE"


def test_sampling_other_motorcycles_prefers_distinct_tracks(tmp_path: Path) -> None:
    crop_base = tmp_path / "crops"
    crop_base.mkdir()
    rows: list[dict[str, object]] = []
    for index, (track, width) in enumerate(
        [
            ("CAM_001:TRACK_8", 78),
            ("CAM_001:TRACK_15", 92),
            ("CAM_001:TRACK_33", 110),
            ("CAM_001:TRACK_55", 126),
        ],
        start=1,
    ):
        crop_path = crop_base / f"{track.split(':')[-1]}_{index}.jpg"
        _write_crop(crop_path, width, 120)
        rows.append(
            {
                "camera_id": "CAM_001",
                "local_track_id": track,
                "frame_number": 1000 + index,
                "vehicle_class": "motorcycle",
                "crop_width": width,
                "crop_height": 120,
                "crop_path": str(crop_path),
                "inside_capture_zone": index % 2 == 0,
                "evidence_eligible": False,
                "florence_eligible": False,
            }
        )

    sampled = _MODULE.sample_other_motorcycle_rows(rows, exclude_track_ids={"CAM_001:TRACK_192"}, max_tracks=3)
    assert len(sampled) == 3
    assert len({row["local_track_id"] for row in sampled}) == 3
    assert { _MODULE.bucket_for_width(int(row["crop_width"])) for row in sampled } <= {"<80", "80-99", "100-119", ">=120"}
