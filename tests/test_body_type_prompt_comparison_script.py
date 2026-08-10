from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import cv2
import numpy as np


def _load_module():
    module_path = Path("scripts/benchmark_body_type_prompt_comparison.py").resolve()
    spec = importlib.util.spec_from_file_location("benchmark_body_type_prompt_comparison", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeBackend:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)

    def run_task(self, image, task_prompt, text_input=None, *, adapter_active=None):
        response = self.responses.pop(0)
        return {
            "status": "completed",
            "payload": {
                "generated_text": response,
                "parsed_answer": response,
                "inference_duration_ms": 9.5,
            },
        }


def test_body_type_prompt_comparison_writes_outputs(tmp_path: Path) -> None:
    module = _load_module()
    crop_dir = tmp_path / "crops"
    crop_dir.mkdir()
    review_csv = tmp_path / "manual_review.csv"
    image_paths: list[Path] = []
    rows = []
    for index in range(2):
        image = np.full((120 + index, 180 + index, 3), 127, dtype=np.uint8)
        image_path = crop_dir / f"frame_{index:06d}.jpg"
        assert cv2.imwrite(str(image_path), image)
        image_paths.append(image_path)
        rows.append(
            {
                "track_id": f"CAM_001:TRACK_{index + 1}",
                "frame_number": str(index),
                "crop_path": str(image_path),
                "raw_response": "",
                "predicted_body_type": "",
                "ground_truth_body_type": "",
                "prediction_correct": "",
                "notes": "",
            }
        )
    with review_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    output_dir = tmp_path / "comparison"
    backend = _FakeBackend(["car", "sedan", "hyundai", "suv"])
    result = module.run_prompt_comparison(
        manual_review_csv=review_csv,
        config_path=Path("config/archive/config.validation_car_body_type.yaml"),
        output_dir=output_dir,
        backend=backend,
    )

    assert Path(result["csv_path"]).exists()
    assert Path(result["summary_path"]).exists()
    assert result["summary"]["total_crops"] == 2
    assert result["summary"]["old"]["valid_parsed_crops"] == 0
    assert result["summary"]["new"]["valid_parsed_crops"] == 2
    assert result["summary"]["old"]["raw_response_distribution"]["car"] == 1
    assert result["summary"]["new"]["raw_response_distribution"]["sedan"] == 1
