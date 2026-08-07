from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import cv2
import numpy as np


def _load_module():
    module_path = Path("scripts/benchmark_body_type_prompt_experiment_v2.py").resolve()
    spec = importlib.util.spec_from_file_location("benchmark_body_type_prompt_experiment_v2", module_path)
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
                "inference_duration_ms": 7.0,
            },
        }


def test_body_type_prompt_experiment_v2_writes_outputs(tmp_path: Path) -> None:
    module = _load_module()
    image_path = tmp_path / "crop.jpg"
    image = np.full((120, 180, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    source_csv = tmp_path / "source.csv"
    with source_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["track_id", "frame_number", "crop_path", "crop_width", "crop_height"])
        writer.writeheader()
        writer.writerow(
            {
                "track_id": "CAM_001:TRACK_1",
                "frame_number": "1",
                "crop_path": str(image_path),
                "crop_width": "180",
                "crop_height": "120",
            }
        )

    backend = _FakeBackend(["sedan", "suv", "hatchback", "mpv", "A white sport utility vehicle is on the road."])
    output_dir = tmp_path / "out"
    result = module.run_experiment(
        source_csv=source_csv,
        config_path=Path("config.validation_car_body_type.yaml"),
        output_dir=output_dir,
        backend=backend,
    )

    assert Path(result["csv_path"]).exists()
    assert Path(result["summary_path"]).exists()
    assert result["summary"]["baseline"]["valid_parsed_crops"] == 1
    assert result["summary"]["prompt_a"]["parsed_distribution"]["SUV"] == 1
    assert result["summary"]["prompt_b"]["parsed_distribution"]["HATCHBACK"] == 1
    assert result["summary"]["prompt_c"]["parsed_distribution"]["MPV"] == 1
    assert result["summary"]["caption"]["parsed_distribution"]["SUV"] == 1
