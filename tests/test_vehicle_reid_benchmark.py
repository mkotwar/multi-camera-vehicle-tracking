from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.vehicle_reid_benchmark import VehicleReIDModel, run_vehicle_reid_benchmark
from src.vehicle_reid_experiment import l2_normalize


class TinyDeterministicReID(VehicleReIDModel):
    def __init__(self) -> None:
        self.name = "tiny"
        self.architecture = "test adapter"
        self.checkpoint_path = "test.pth"
        self.training_dataset = "synthetic"
        self.input_size = (16, 16)

    def embed_image(self, image) -> np.ndarray:
        array = np.asarray(Image.open(image).convert("RGB"), dtype=np.float32) / 255.0
        return l2_normalize(array.mean(axis=(0, 1)))


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 16), color).save(path)


def test_adapter_embedding_normalization_determinism_and_dimension(tmp_path: Path) -> None:
    image = tmp_path / "crop.jpg"
    _write_image(image, (80, 90, 100))
    model = TinyDeterministicReID()
    first = model.embed_image(image)
    second = model.embed_image(image)
    assert first.shape == (3,)
    assert np.linalg.norm(first) == pytest.approx(1.0)
    assert np.allclose(first, second)


def test_missing_checkpoint_metadata_and_corrupt_image_handling(tmp_path: Path) -> None:
    model = TinyDeterministicReID()
    assert model.metadata["checkpoint_path"] == "test.pth"
    corrupt = tmp_path / "bad.jpg"
    corrupt.write_text("bad", encoding="utf-8")
    with pytest.raises(Exception):
        model.embed_image(corrupt)


def test_benchmark_does_not_mutate_tracks_json_and_writes_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_synthetic_reid_benchmark_run(run_dir)
    before = (run_dir / "tracks.json").read_bytes()
    result = run_vehicle_reid_benchmark(run_dir, models=[TinyDeterministicReID()])
    after = (run_dir / "tracks.json").read_bytes()
    assert before == after
    assert result["tracks_json_unchanged"] is True
    assert (run_dir / "vehicle_reid_benchmark" / "tiny" / "similarity_matrix.json").exists()
    matrix = json.loads((run_dir / "vehicle_reid_benchmark" / "tiny" / "similarity_matrix.json").read_text(encoding="utf-8"))
    assert matrix["matrix"][0][0] == pytest.approx(1.0)


def _write_synthetic_reid_benchmark_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    tracks = []
    observations = ["local_track_id,frame_number,timestamp_seconds,x1,y1,x2,y2,confidence,raw_class_name"]
    colors = {
        "CAM_001:TRACK_4": (120, 120, 120),
        "CAM_001:TRACK_5": (125, 125, 125),
        "CAM_001:TRACK_8": (130, 130, 130),
        "CAM_001:TRACK_16": (200, 200, 200),
        "CAM_001:TRACK_18": (210, 210, 210),
        "CAM_001:TRACK_19": (20, 30, 40),
        "CAM_001:TRACK_21": (30, 40, 50),
        "CAM_001:TRACK_30": (40, 50, 60),
    }
    selected = {}
    for index, (track_id, color) in enumerate(colors.items(), start=1):
        tracks.append(
            {
                "local_track_id": track_id,
                "camera_id": "CAM_001",
                "status": "COMPLETED",
                "first_frame": index * 10,
                "last_frame": index * 10 + 2,
                "observation_count": 3,
                "final_class": "car",
                "vehicle_enrichment": {"vehicle_colour": {"label": "GREY"}},
            }
        )
        crop = run_dir / "crops" / f"{track_id.replace(':', '_')}.jpg"
        _write_image(crop, color)
        selected[track_id] = [
            {
                "local_track_id": track_id,
                "role": "BEST_OVERALL",
                "frame_number": index * 10,
                "crop_path": str(crop),
                "source": "test",
                "reason": "synthetic",
            }
        ]
        observations.append(f"{track_id},{index * 10},{index / 3},1,2,10,12,0.9,car")
    (run_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")
    (run_dir / "observations.csv").write_text("\n".join(observations), encoding="utf-8")
    vehicle_test = run_dir / "vehicle_reid_test"
    vehicle_test.mkdir()
    (vehicle_test / "selected_crops.json").write_text(json.dumps(selected), encoding="utf-8")
    recon_dir = run_dir / "track_reconciliation_test"
    recon_dir.mkdir()
    (recon_dir / "track_reconciliation_test.json").write_text(json.dumps({"attempts": [], "accepted_associations": []}), encoding="utf-8")
