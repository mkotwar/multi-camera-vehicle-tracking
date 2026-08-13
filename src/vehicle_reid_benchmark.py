from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image

from .track_reconciliation import load_tracklets
from .vehicle_reid_experiment import (
    POSITIVE_TRACK_GROUP,
    build_reconciliation_lookup,
    build_similarity_matrix,
    cosine_similarity,
    distribution,
    frame_gap,
    l2_normalize,
    load_pil_image,
    make_contact_sheet,
    sha256_file,
    write_csv,
    write_similarity_matrix,
)


FROZEN_HARD_NEGATIVES = [
    ("CAM_001:TRACK_8", "CAM_001:TRACK_21"),
    ("CAM_001:TRACK_5", "CAM_001:TRACK_21"),
    ("CAM_001:TRACK_19", "CAM_001:TRACK_30"),
    ("CAM_001:TRACK_21", "CAM_001:TRACK_30"),
    ("CAM_001:TRACK_16", "CAM_001:TRACK_18"),
    ("CAM_001:TRACK_5", "CAM_001:TRACK_19"),
]
FLORENCE_BASELINE = {
    "model": "Florence baseline",
    "positive_min": 0.892562,
    "positive_mean": 0.929230,
    "negative_max": 0.987592,
    "negative_mean": 0.910843,
    "separation_gap": -0.095030,
}


@dataclass(frozen=True, slots=True)
class BenchmarkCrop:
    local_track_id: str
    role: str
    frame_number: int
    crop_path: str


class VehicleReIDModel:
    name: str
    architecture: str
    checkpoint_path: str
    training_dataset: str
    input_size: tuple[int, int]

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "architecture": self.architecture,
            "checkpoint_path": self.checkpoint_path,
            "training_dataset": self.training_dataset,
            "input_size": list(self.input_size),
        }

    def embed_image(self, image: Image.Image | np.ndarray | str | Path) -> np.ndarray:
        raise NotImplementedError

    def embed_track(self, crop_paths: Iterable[str | Path]) -> np.ndarray:
        embeddings = [l2_normalize(self.embed_image(path)) for path in crop_paths]
        if not embeddings:
            raise ValueError("At least one crop is required.")
        return l2_normalize(np.mean(np.stack(embeddings, axis=0), axis=0))


class TransReIDAdapter(VehicleReIDModel):
    def __init__(self, *, repo_dir: Path, checkpoint_path: Path, device: str = "auto") -> None:
        self.name = "transreid"
        self.architecture = "TransReID ViT-B/16 with JPM, stride [12,12]"
        self.checkpoint_path = str(checkpoint_path)
        self.training_dataset = "VeRi-776"
        self.input_size = (256, 256)
        self.repo_dir = repo_dir
        self.device_request = device
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device = "cpu"
        self._latencies_ms: list[float] = []
        self._model_size_bytes = checkpoint_path.stat().st_size if checkpoint_path.exists() else 0
        self._peak_vram_mb: float | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        payload = super().metadata
        payload.update(
            {
                "checkpoint_source": "https://github.com/damo-cv/TransReID README trained models, VeRi TransReID*(ViT)",
                "normalization": "resize 256x256, RGB, mean/std 0.5",
                "feature_procedure": "official model eval forward; output L2-normalized",
                "embedding_dimension": None if self._model is None else 3840,
                "average_embedding_latency_ms": average(self._latencies_ms),
                "peak_vram_mb": self._peak_vram_mb,
                "model_size_bytes": self._model_size_bytes,
                "device": self._device,
            }
        )
        return payload

    def load(self) -> None:
        if self._model is not None:
            return
        import torch

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() and self.device_request != "cpu" else "cpu"
        if self._device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        transreid_root = str(self.repo_dir)
        if transreid_root not in sys.path:
            sys.path.insert(0, transreid_root)
        from config import cfg
        from model.make_model import make_model

        cfg.defrost()
        cfg.merge_from_file(str(self.repo_dir / "configs" / "VeRi" / "vit_transreid_stride.yml"))
        cfg.MODEL.PRETRAIN_CHOICE = "none"
        cfg.MODEL.DEVICE = self._device
        cfg.TEST.NECK_FEAT = "before"
        cfg.freeze()
        model = make_model(cfg, num_class=576, camera_num=20, view_num=8)
        state = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        cleaned = {str(k).replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        critical_missing = [key for key in missing if not key.startswith("classifier")]
        if critical_missing:
            raise RuntimeError(f"TransReID missing non-classifier keys: {critical_missing[:8]}")
        model.to(self._device)
        model.eval()
        self._model = model

    def embed_image(self, image: Image.Image | np.ndarray | str | Path) -> np.ndarray:
        self.load()
        assert self._torch is not None
        assert self._model is not None
        tensor = preprocess_rgb(load_pil_image(image), self.input_size, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        tensor = tensor.to(self._device)
        started = time.perf_counter()
        with self._torch.inference_mode():
            zeros = self._torch.zeros((1,), dtype=self._torch.long, device=self._device)
            feat = self._model(tensor, cam_label=zeros, view_label=zeros)
        self._latencies_ms.append((time.perf_counter() - started) * 1000.0)
        if self._device == "cuda":
            self._peak_vram_mb = round(float(self._torch.cuda.max_memory_allocated() / (1024 * 1024)), 3)
        return l2_normalize(feat[0].detach().float().cpu().numpy())


class FastReIDAdapter(VehicleReIDModel):
    def __init__(self, *, repo_dir: Path, checkpoint_path: Path, device: str = "auto") -> None:
        self.name = "fastreid"
        self.architecture = "FastReID SBS R50-IBN"
        self.checkpoint_path = str(checkpoint_path)
        self.training_dataset = "VeRi-776"
        self.input_size = (256, 256)
        self.repo_dir = repo_dir
        self.device_request = device
        self._predictor: Any | None = None
        self._torch: Any | None = None
        self._device = "cpu"
        self._latencies_ms: list[float] = []
        self._model_size_bytes = checkpoint_path.stat().st_size if checkpoint_path.exists() else 0
        self._embedding_dimension: int | None = None
        self._peak_vram_mb: float | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        payload = super().metadata
        payload.update(
            {
                "checkpoint_source": "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth",
                "normalization": "FastReID config pixel mean/std after resize 256x256",
                "feature_procedure": "official DefaultPredictor eval forward; output L2-normalized",
                "embedding_dimension": self._embedding_dimension,
                "average_embedding_latency_ms": average(self._latencies_ms),
                "peak_vram_mb": self._peak_vram_mb,
                "model_size_bytes": self._model_size_bytes,
                "device": self._device,
            }
        )
        return payload

    def load(self) -> None:
        if self._predictor is not None:
            return
        import torch

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() and self.device_request != "cpu" else "cpu"
        if self._device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        fastreid_root = str(self.repo_dir)
        if fastreid_root not in sys.path:
            sys.path.insert(0, fastreid_root)
        from fastreid.config import get_cfg
        from fastreid.modeling.meta_arch import build_model
        from fastreid.utils.checkpoint import Checkpointer

        cfg = get_cfg()
        cfg.merge_from_file(str(self.repo_dir / "configs" / "VeRi" / "sbs_R50-ibn.yml"))
        cfg.defrost()
        cfg.MODEL.WEIGHTS = self.checkpoint_path
        cfg.MODEL.DEVICE = self._device
        cfg.MODEL.BACKBONE.PRETRAIN = False
        cfg.freeze()
        model = build_model(cfg)
        model.eval()
        Checkpointer(model).load(cfg.MODEL.WEIGHTS)
        self._predictor = model

    def embed_image(self, image: Image.Image | np.ndarray | str | Path) -> np.ndarray:
        self.load()
        assert self._torch is not None
        assert self._predictor is not None
        pil = load_pil_image(image).resize(self.input_size, Image.BILINEAR)
        rgb = np.asarray(pil, dtype=np.float32)
        bgr = rgb[:, :, ::-1].copy()
        tensor = self._torch.from_numpy(bgr.transpose(2, 0, 1)).unsqueeze(0).float()
        started = time.perf_counter()
        inputs = {"images": tensor.to(self._predictor.device)}
        with self._torch.inference_mode():
            feat = self._predictor(inputs).cpu()
        self._latencies_ms.append((time.perf_counter() - started) * 1000.0)
        if self._device == "cuda":
            self._peak_vram_mb = round(float(self._torch.cuda.max_memory_allocated() / (1024 * 1024)), 3)
        vector = feat[0].detach().float().cpu().numpy()
        self._embedding_dimension = int(vector.shape[0])
        return l2_normalize(vector)


def run_vehicle_reid_benchmark(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    models: list[VehicleReIDModel] | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    benchmark_dir = Path(output_dir).expanduser().resolve() if output_dir else run_path / "vehicle_reid_benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    tracks_hash_before = sha256_file(run_path / "tracks.json")
    selected = load_selected_crops(run_path / "vehicle_reid_test" / "selected_crops.json")
    benchmark_track_ids = sorted({track for pair in FROZEN_HARD_NEGATIVES for track in pair} | set(POSITIVE_TRACK_GROUP))
    selected = {track_id: selected[track_id] for track_id in benchmark_track_ids if track_id in selected}
    tracklets = {track.local_track_id: track for track in load_tracklets(run_path)}
    reconciliation_lookup = build_reconciliation_lookup(load_json(run_path / "track_reconciliation_test" / "track_reconciliation_test.json"))
    if models is None:
        vendor = benchmark_dir / "vendor"
        checkpoints = benchmark_dir / "checkpoints"
        models = [
            TransReIDAdapter(repo_dir=vendor / "TransReID", checkpoint_path=checkpoints / "transreid_vit_transreid_veri.pth"),
            FastReIDAdapter(repo_dir=vendor / "fast-reid", checkpoint_path=checkpoints / "fastreid_veri_sbs_R50-ibn.pth"),
        ]
    manifest = build_manifest(run_path, benchmark_dir, selected, models)
    (benchmark_dir / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    comparison_rows = [dict(FLORENCE_BASELINE)]
    model_results: dict[str, Any] = {}
    for model in models:
        result = run_single_model(model, benchmark_dir / model.name, selected, tracklets, reconciliation_lookup)
        model_results[model.name] = result
        comparison_rows.append(result["comparison_row"])
    write_csv(benchmark_dir / "model_comparison.csv", comparison_rows)
    (benchmark_dir / "model_comparison.json").write_text(json.dumps(comparison_rows, indent=2), encoding="utf-8")
    tracks_hash_after = sha256_file(run_path / "tracks.json")
    result = {
        "source_run_directory": str(run_path),
        "output_directory": str(benchmark_dir),
        "tracks_json_sha256_before": tracks_hash_before,
        "tracks_json_sha256_after": tracks_hash_after,
        "tracks_json_unchanged": tracks_hash_before == tracks_hash_after,
        "model_results": model_results,
        "model_comparison": comparison_rows,
    }
    (benchmark_dir / "vehicle_reid_benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_single_model(
    model: VehicleReIDModel,
    output_dir: Path,
    selected: dict[str, list[BenchmarkCrop]],
    tracklets: dict[str, Any],
    reconciliation_lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "embeddings").mkdir(exist_ok=True)
    (output_dir / "visual_evidence").mkdir(exist_ok=True)
    track_embeddings = {}
    for track_id, crops in selected.items():
        track_embeddings[track_id] = model.embed_track([crop.crop_path for crop in crops])
        np.save(output_dir / "embeddings" / f"{track_id.replace(':', '_')}.npy", track_embeddings[track_id])
    track_ids = list(selected.keys())
    pair_rows = build_benchmark_pair_rows(track_ids, track_embeddings, tracklets, reconciliation_lookup)
    matrix = build_similarity_matrix(track_ids, track_embeddings)
    write_csv(output_dir / "pairwise_similarity.csv", pair_rows)
    write_similarity_matrix(output_dir, track_ids, matrix)
    ranking_rows = build_ranking_rows(POSITIVE_TRACK_GROUP, track_ids, track_embeddings)
    write_csv(output_dir / "ranking.csv", ranking_rows)
    positives = [float(row["reid_similarity"]) for row in pair_rows if row["manual_label"] == "SAME_VEHICLE"]
    negatives = [float(row["reid_similarity"]) for row in pair_rows if row["manual_label"] == "UNCERTAIN_NEGATIVE"]
    positive_distribution = distribution(positives)
    negative_distribution = distribution(negatives)
    hardest_negative = max((row for row in pair_rows if row["manual_label"] == "UNCERTAIN_NEGATIVE"), key=lambda row: float(row["reid_similarity"]))
    weakest_positive = min((row for row in pair_rows if row["manual_label"] == "SAME_VEHICLE"), key=lambda row: float(row["reid_similarity"]))
    generate_model_visuals(output_dir / "visual_evidence", selected, [hardest_negative, weakest_positive])
    rank_metrics = rank_metrics_for_queries(ranking_rows)
    comparison_row = {
        "model": model.name,
        "positive_min": positive_distribution["minimum"],
        "positive_mean": positive_distribution["mean"],
        "negative_max": negative_distribution["maximum"],
        "negative_mean": negative_distribution["mean"],
        "separation_gap": round(float(positive_distribution["minimum"]) - float(negative_distribution["maximum"]), 6),
    }
    result = {
        "model": model.metadata,
        "positive_distribution": positive_distribution,
        "negative_distribution": negative_distribution,
        "comparison_row": comparison_row,
        "hardest_negative": hardest_negative,
        "weakest_positive": weakest_positive,
        "rank_metrics": rank_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_benchmark_pair_rows(
    track_ids: list[str],
    embeddings: dict[str, np.ndarray],
    tracklets: dict[str, Any],
    reconciliation_lookup: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    positive_pairs = {tuple(sorted(pair)) for pair in combinations(POSITIVE_TRACK_GROUP)}
    frozen_negative_pairs = {tuple(sorted(pair)) for pair in FROZEN_HARD_NEGATIVES}
    rows = []
    for left, right in combinations(track_ids):
        key = tuple(sorted((left, right)))
        if key not in positive_pairs and key not in frozen_negative_pairs:
            continue
        left_track = tracklets.get(left)
        right_track = tracklets.get(right)
        recon = reconciliation_lookup.get((left, right)) or reconciliation_lookup.get((right, left)) or {}
        rows.append(
            {
                "left_track": left,
                "right_track": right,
                "left_class": getattr(left_track, "final_class", ""),
                "right_class": getattr(right_track, "final_class", ""),
                "left_colour": getattr(left_track, "colour", ""),
                "right_colour": getattr(right_track, "colour", ""),
                "time_gap_frames": frame_gap(left_track, right_track),
                "existing_reconciliation_score": recon.get("score"),
                "position_score": (recon.get("components") or {}).get("position"),
                "motion_score": (recon.get("components") or {}).get("motion"),
                "direction_score": (recon.get("components") or {}).get("direction"),
                "existing_reconciliation_decision": recon.get("result") or "NOT_ATTEMPTED",
                "reid_similarity": round(cosine_similarity(embeddings[left], embeddings[right]), 6),
                "manual_label": "SAME_VEHICLE" if key in positive_pairs else "UNCERTAIN_NEGATIVE",
            }
        )
    return sorted(rows, key=lambda row: float(row["reid_similarity"]), reverse=True)


def build_ranking_rows(query_ids: list[str], track_ids: list[str], embeddings: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for query_id in query_ids:
        candidates = []
        for candidate_id in track_ids:
            if candidate_id == query_id:
                continue
            candidates.append(
                {
                    "query_track": query_id,
                    "candidate_track": candidate_id,
                    "reid_similarity": round(cosine_similarity(embeddings[query_id], embeddings[candidate_id]), 6),
                    "label": "SAME" if candidate_id in POSITIVE_TRACK_GROUP else "DIFFERENT_OR_UNCERTAIN",
                }
            )
        candidates.sort(key=lambda row: float(row["reid_similarity"]), reverse=True)
        for rank, row in enumerate(candidates, start=1):
            row["rank"] = rank
            rows.append(row)
    return rows


def rank_metrics_for_queries(ranking_rows: list[dict[str, Any]]) -> dict[str, float]:
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in ranking_rows:
        by_query.setdefault(str(row["query_track"]), []).append(row)
    rank1_hits = 0
    rank3_hits = 0
    for rows in by_query.values():
        ordered = sorted(rows, key=lambda row: int(row["rank"]))
        if ordered and ordered[0]["label"] == "SAME":
            rank1_hits += 1
        if any(row["label"] == "SAME" for row in ordered[:3]):
            rank3_hits += 1
    total = max(1, len(by_query))
    return {"rank1": rank1_hits / total, "rank3": rank3_hits / total}


def generate_model_visuals(output_dir: Path, selected: dict[str, list[BenchmarkCrop]], rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        left = str(row["left_track"])
        right = str(row["right_track"])
        make_contact_sheet(
            output_dir / f"{left.replace(':', '_')}__{right.replace(':', '_')}.jpg",
            selected[left],
            selected[right],
            title=f"{left} vs {right}",
            subtitle=f"similarity={row['reid_similarity']} label={row['manual_label']}",
        )


def load_selected_crops(path: Path) -> dict[str, list[BenchmarkCrop]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected: dict[str, list[BenchmarkCrop]] = {}
    for track_id, items in payload.items():
        selected[track_id] = [
            BenchmarkCrop(
                local_track_id=str(item["local_track_id"]),
                role=str(item["role"]),
                frame_number=int(item["frame_number"]),
                crop_path=str(item["crop_path"]),
            )
            for item in items
        ]
    return selected


def preprocess_rgb(image: Image.Image, input_size: tuple[int, int], *, mean: tuple[float, float, float], std: tuple[float, float, float]) -> Any:
    import torch

    resized = image.resize(input_size, Image.BILINEAR)
    array = np.asarray(resized.convert("RGB"), dtype=np.float32) / 255.0
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).float()


def combinations(items: Iterable[str]) -> list[tuple[str, str]]:
    values = list(items)
    return [(values[i], values[j]) for i in range(len(values)) for j in range(i + 1, len(values))]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(run_path: Path, benchmark_dir: Path, selected: dict[str, list[BenchmarkCrop]], models: list[VehicleReIDModel]) -> dict[str, Any]:
    return {
        "source_run_directory": str(run_path),
        "selected_crops_source": str(run_path / "vehicle_reid_test" / "selected_crops.json"),
        "positive_track_group": POSITIVE_TRACK_GROUP,
        "frozen_hard_negatives": [list(pair) for pair in FROZEN_HARD_NEGATIVES],
        "benchmark_tracks": list(selected.keys()),
        "models": [model.metadata for model in models],
        "output_directory": str(benchmark_dir),
    }


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(np.mean(np.asarray(values, dtype=np.float32))), 3)
