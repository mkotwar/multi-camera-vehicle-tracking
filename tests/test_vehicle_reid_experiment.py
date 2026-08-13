from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.vehicle_reid_experiment import (
    VehicleAppearanceEmbedder,
    build_reid_groups,
    build_similarity_matrix,
    cosine_similarity,
    l2_normalize,
    run_vehicle_reid_experiment,
)


class MeanColorEmbedder(VehicleAppearanceEmbedder):
    def embed_image(self, image) -> np.ndarray:
        array = np.asarray(Image.open(image).convert("RGB"), dtype=np.float32) / 255.0
        return l2_normalize(array.mean(axis=(0, 1)))


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color).save(path)


def test_l2_normalization_and_cosine_similarity() -> None:
    normalized = l2_normalize(np.array([3.0, 4.0], dtype=np.float32))
    assert np.linalg.norm(normalized) == pytest.approx(1.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_multiple_crop_aggregation_single_crop_and_determinism(tmp_path: Path) -> None:
    crop_a = tmp_path / "a.jpg"
    crop_b = tmp_path / "b.jpg"
    _write_image(crop_a, (120, 120, 120))
    _write_image(crop_b, (130, 130, 130))
    embedder = MeanColorEmbedder()
    single = embedder.embed_track([crop_a])
    multi_1 = embedder.embed_track([crop_a, crop_b])
    multi_2 = embedder.embed_track([crop_a, crop_b])
    assert np.linalg.norm(single) == pytest.approx(1.0)
    assert np.linalg.norm(multi_1) == pytest.approx(1.0)
    assert np.allclose(multi_1, multi_2)


def test_missing_crop_and_corrupt_crop_handling(tmp_path: Path) -> None:
    embedder = MeanColorEmbedder()
    with pytest.raises(FileNotFoundError):
        embedder.embed_track([tmp_path / "missing.jpg"])
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_text("not an image", encoding="utf-8")
    with pytest.raises(Exception):
        embedder.embed_track([corrupt])


def test_similarity_matrix_symmetry_and_diagonal() -> None:
    embeddings = {
        "A": l2_normalize(np.array([1.0, 0.0, 0.0])),
        "B": l2_normalize(np.array([0.5, 0.5, 0.0])),
        "C": l2_normalize(np.array([0.0, 1.0, 0.0])),
    }
    matrix = build_similarity_matrix(["A", "B", "C"], embeddings)
    assert matrix[0][0] == pytest.approx(1.0)
    assert matrix[1][1] == pytest.approx(1.0)
    assert matrix[2][2] == pytest.approx(1.0)
    assert matrix[0][1] == pytest.approx(matrix[1][0])
    assert matrix[0][2] == pytest.approx(matrix[2][0])


def test_group_consistency_records_all_pairwise_edges() -> None:
    embeddings = {
        "T4": l2_normalize(np.array([1.0, 0.0])),
        "T5": l2_normalize(np.array([0.9, 0.1])),
        "T8": l2_normalize(np.array([0.8, 0.2])),
    }
    groups = build_reid_groups(["T4", "T5", "T8"], embeddings)
    group = groups["groups"][0]
    assert group["pairwise_consistency_checked"] is True
    assert len(group["pairwise_similarities"]) == 3


def test_experiment_does_not_mutate_tracks_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_synthetic_run(run_dir)
    before = (run_dir / "tracks.json").read_bytes()
    result = run_vehicle_reid_experiment(
        run_dir,
        embedder=MeanColorEmbedder(),
        positive_track_ids=["CAM_001:TRACK_4", "CAM_001:TRACK_5", "CAM_001:TRACK_8"],
        comparison_track_ids=["CAM_001:TRACK_2", "CAM_001:TRACK_12"],
    )
    after = (run_dir / "tracks.json").read_bytes()
    assert before == after
    assert result["tracks_json_unchanged"] is True
    matrix = json.loads((run_dir / "vehicle_reid_test" / "similarity_matrix.json").read_text(encoding="utf-8"))
    assert matrix["matrix"][0][0] == pytest.approx(1.0)


def _write_synthetic_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    tracks = []
    observations = ["local_track_id,frame_number,timestamp_seconds,x1,y1,x2,y2,confidence,raw_class_name"]
    specs = [
        ("CAM_001:TRACK_4", (120, 120, 120), "SILVER", 1, 3),
        ("CAM_001:TRACK_5", (125, 125, 125), "GREY", 6, 8),
        ("CAM_001:TRACK_8", (130, 130, 130), "GREY", 12, 14),
        ("CAM_001:TRACK_2", (20, 20, 20), "BLACK", 1, 3),
        ("CAM_001:TRACK_12", (200, 200, 200), "WHITE", 15, 17),
    ]
    summary_rows = [
        "camera_id,local_track_id,vehicle_class,track_start_frame,track_end_frame,track_duration_frames,candidate_crops_seen,candidate_crops_retained,acceptable_crops,preferred_crops,selected_body_type_crops,selected_colour_crops,largest_original_crop_width,largest_original_crop_height,best_quality_score,body_type_status,body_type_label,colour_status,colour_label"
    ]
    for track_id, color, colour_label, first, last in specs:
        tracks.append(
            {
                "local_track_id": track_id,
                "camera_id": "CAM_001",
                "status": "COMPLETED",
                "first_frame": first,
                "last_frame": last,
                "first_timestamp_seconds": first / 30.0,
                "last_timestamp_seconds": last / 30.0,
                "observation_count": last - first + 1,
                "final_class": "car",
                "vehicle_enrichment": {"vehicle_colour": {"label": colour_label}},
            }
        )
        evidence_dir = run_dir / "evidence" / "CAM_001" / track_id.replace(":", "_")
        crop_dir = evidence_dir / "crops"
        evidence = []
        for frame in range(first, last + 1):
            crop_path = crop_dir / f"frame_{frame:06d}.jpg"
            _write_image(crop_path, color)
            role = "BEST_OVERALL" if frame == first else "MIDDLE" if frame < last else "LAST"
            evidence.append(
                {
                    "local_track_id": track_id,
                    "role": role,
                    "frame_number": frame,
                    "confidence": 0.9,
                    "original_crop_width": 32,
                    "original_crop_height": 24,
                    "sharpness_score": 100.0,
                    "best_overall_score": 0.9,
                    "crop_path": str(crop_path),
                }
            )
            observations.append(f"{track_id},{frame},{frame / 30.0},1,2,31,22,0.9,car")
        (evidence_dir / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
        summary_rows.append(
            f"CAM_001,{track_id},CAR,{first},{last},{last - first + 1},3,3,3,0,,selected,32,24,0.9,disabled,UNKNOWN,completed,{colour_label}"
        )
    (run_dir / "tracks.json").write_text(json.dumps(tracks, indent=2), encoding="utf-8")
    (run_dir / "observations.csv").write_text("\n".join(observations), encoding="utf-8")
    (run_dir / "vehicle_enrichment_track_evidence_summary.csv").write_text("\n".join(summary_rows), encoding="utf-8")
