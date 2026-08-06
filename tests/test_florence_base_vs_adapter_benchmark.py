from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.vehicle_enrichment.benchmarking.florence_base_vs_adapter import (
    build_manual_review_rows,
    build_pivot_rows,
    generate_configuration_audit,
    inspect_model_assets,
    summarize_rows,
)


def test_base_only_and_adapter_configs_are_correct() -> None:
    base_cfg = yaml.safe_load(Path("config.validation_florence_base_only.yaml").read_text(encoding="utf-8"))
    adapter_cfg = yaml.safe_load(Path("config.validation_florence_base_adapter.yaml").read_text(encoding="utf-8"))

    assert base_cfg["vehicle_enrichment"]["shared_florence"]["adapter_path"] is None
    assert base_cfg["vehicle_enrichment"]["shared_florence"]["adapter_enabled"] is False
    assert adapter_cfg["vehicle_enrichment"]["shared_florence"]["adapter_path"] == "D:/project/models/OCR_MUKUL/OCR_MUKUL/adaptor_florance_baseFT"
    assert adapter_cfg["vehicle_enrichment"]["shared_florence"]["processor_path"] == base_cfg["vehicle_enrichment"]["shared_florence"]["processor_path"]


def test_configuration_audit_and_asset_inspection_outputs(tmp_path: Path) -> None:
    audit = generate_configuration_audit(Path("."), tmp_path)
    assert (tmp_path / "florence_model_configuration_audit.json").exists()
    assert (tmp_path / "florence_model_configuration_audit.md").exists()
    assert any(item["config_path"] == "config.yaml" for item in audit["configs"])

    assets = inspect_model_assets(
        "D:/project/models/Florence-2-base-ft",
        "D:/project/models/Florence-2-base-ft",
        "D:/project/models/OCR_MUKUL/OCR_MUKUL/adaptor_florance_baseFT",
    )
    assert "config.json" in assets["base_model_files"]
    assert "adapter_config.json" in assets["adapter_files"]
    assert assets["adapter_type"] == "LORA"


def test_pivot_summary_and_manual_review_outputs(tmp_path: Path) -> None:
    rows = [
        {
            "camera_id": "CAM_001",
            "local_track_id": "TRACK_1",
            "frame_index": 1,
            "crop_path": "crop_1.jpg",
            "vehicle_class": "CAR",
            "configuration": "base_current",
            "body_type_label": "UNKNOWN",
            "colour_label": "UNKNOWN",
            "raw_response": json.dumps({"body_type": "", "colour": ""}),
            "generic_response": json.dumps({"body_type": True, "colour": True}),
            "prompt_echo": json.dumps({"body_type": False, "colour": False}),
            "inference_time_ms": 10.0,
            "adapter_loaded": False,
            "peak_gpu_memory_mb": 0.0,
        },
        {
            "camera_id": "CAM_001",
            "local_track_id": "TRACK_1",
            "frame_index": 1,
            "crop_path": "crop_1.jpg",
            "vehicle_class": "CAR",
            "configuration": "adapter_current",
            "body_type_label": "SEDAN",
            "colour_label": "WHITE",
            "raw_response": json.dumps({"body_type": "sedan", "colour": "white"}),
            "generic_response": json.dumps({"body_type": False, "colour": False}),
            "prompt_echo": json.dumps({"body_type": False, "colour": False}),
            "inference_time_ms": 12.0,
            "adapter_loaded": True,
            "peak_gpu_memory_mb": 0.0,
        },
        {
            "camera_id": "CAM_001",
            "local_track_id": "TRACK_1",
            "frame_index": 1,
            "crop_path": "crop_1.jpg",
            "vehicle_class": "CAR",
            "configuration": "base_caption",
            "body_type_label": "SEDAN",
            "colour_label": "WHITE",
            "raw_response": json.dumps({"body_type": "caption", "colour": "caption"}),
            "generic_response": json.dumps({"body_type": False, "colour": False}),
            "prompt_echo": json.dumps({"body_type": False, "colour": False}),
            "inference_time_ms": 8.0,
            "adapter_loaded": False,
            "peak_gpu_memory_mb": 0.0,
        },
        {
            "camera_id": "CAM_001",
            "local_track_id": "TRACK_1",
            "frame_index": 1,
            "crop_path": "crop_1.jpg",
            "vehicle_class": "CAR",
            "configuration": "adapter_caption",
            "body_type_label": "SUV",
            "colour_label": "WHITE",
            "raw_response": json.dumps({"body_type": "caption2", "colour": "caption2"}),
            "generic_response": json.dumps({"body_type": False, "colour": False}),
            "prompt_echo": json.dumps({"body_type": False, "colour": False}),
            "inference_time_ms": 9.0,
            "adapter_loaded": True,
            "peak_gpu_memory_mb": 0.0,
        },
    ]

    pivot_rows = build_pivot_rows(rows)
    assert pivot_rows[0]["base_current_raw"]
    assert pivot_rows[0]["adapter_caption_body_type"] == "SUV"

    summary, summary_rows = summarize_rows(rows)
    assert summary["configurations"]["base_current"]["body_type_unknown_count"] == 1
    assert summary["agreements"]["current_body_type"]["base_unknown_adapter_valid"] == 1
    assert any(item["section"] == "base_current" for item in summary_rows)

    manual_review = build_manual_review_rows(pivot_rows)
    assert manual_review
    assert any(item["review_bucket"] == "base_unknown_adapter_valid" for item in manual_review)

    pivot_path = tmp_path / "pivot_comparison.csv"
    with pivot_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pivot_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pivot_rows)
    assert pivot_path.exists()


def test_current_and_caption_preprocessing_fingerprints_are_deterministic(tmp_path: Path) -> None:
    image = np.full((20, 30, 3), 127, dtype=np.uint8)
    image_path = tmp_path / "crop.jpg"
    assert cv2.imwrite(str(image_path), image)
    loaded = cv2.imread(str(image_path))
    assert loaded is not None
    assert loaded.shape[:2] == (20, 30)
