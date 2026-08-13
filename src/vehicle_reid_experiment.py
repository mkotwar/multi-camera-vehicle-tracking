from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .track_reconciliation import load_tracklets


POSITIVE_TRACK_GROUP = ["CAM_001:TRACK_4", "CAM_001:TRACK_5", "CAM_001:TRACK_8"]
DEFAULT_COMPARISON_TRACKS = [
    "CAM_001:TRACK_2",
    "CAM_001:TRACK_12",
    "CAM_001:TRACK_16",
    "CAM_001:TRACK_18",
    "CAM_001:TRACK_19",
    "CAM_001:TRACK_21",
    "CAM_001:TRACK_22",
    "CAM_001:TRACK_29",
    "CAM_001:TRACK_30",
]


@dataclass(frozen=True, slots=True)
class SelectedCrop:
    local_track_id: str
    role: str
    frame_number: int
    crop_path: str
    source: str
    reason: str
    confidence: float | None = None
    original_crop_width: int | None = None
    original_crop_height: int | None = None
    sharpness_score: float | None = None
    best_overall_score: float | None = None


class VehicleAppearanceEmbedder:
    def embed_image(self, image: Image.Image | np.ndarray | str | Path) -> np.ndarray:
        raise NotImplementedError

    def embed_track(self, crop_paths: Iterable[str | Path]) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        for crop_path in crop_paths:
            path = Path(crop_path)
            if not path.exists():
                raise FileNotFoundError(f"Crop does not exist: {path}")
            embeddings.append(l2_normalize(self.embed_image(path)))
        if not embeddings:
            raise ValueError("At least one crop is required for a track embedding.")
        return l2_normalize(np.mean(np.stack(embeddings, axis=0), axis=0))


class FlorenceVisionAppearanceEmbedder(VehicleAppearanceEmbedder):
    def __init__(
        self,
        model_path: str | Path = "D:/project/models/Florence-2-base-ft",
        *,
        device: str = "auto",
        dtype: str = "auto",
    ) -> None:
        self.model_path = str(model_path)
        self.device_request = device
        self.dtype_request = dtype
        self.model_name = "Florence-2-base-ft vision encoder pooled tokens"
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._device: str = "cpu"
        self._dtype: Any | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "GENERIC APPEARANCE EMBEDDING BASELINE",
            "model": self.model_name,
            "model_path": self.model_path,
            "device": self._device,
            "dtype": str(self._dtype),
            "embedding_pooling": "mean pool Florence _encode_image tokens, then L2 normalize",
        }

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._torch = torch
        use_cuda = torch.cuda.is_available() and self.device_request != "cpu"
        self._device = "cuda" if use_cuda else "cpu"
        if self.dtype_request == "float32" or not use_cuda:
            self._dtype = torch.float32
        else:
            self._dtype = torch.float16
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=self._dtype,
        ).to(self._device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

    def embed_image(self, image: Image.Image | np.ndarray | str | Path) -> np.ndarray:
        self.load()
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None
        pil_image = load_pil_image(image)
        inputs = self._processor(text="<CAPTION>", images=pil_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self._device, dtype=self._dtype)
        with self._torch.inference_mode():
            encoded = self._model._encode_image(pixel_values)
            pooled = encoded.mean(dim=1)
        return l2_normalize(pooled[0].detach().float().cpu().numpy())


class ColorTextureAppearanceEmbedder(VehicleAppearanceEmbedder):
    """Small deterministic fallback for tests and environments without model access."""

    def embed_image(self, image: Image.Image | np.ndarray | str | Path) -> np.ndarray:
        pil_image = load_pil_image(image).resize((96, 96))
        array = np.asarray(pil_image.convert("RGB"), dtype=np.float32) / 255.0
        hist_parts = []
        for channel in range(3):
            hist, _ = np.histogram(array[:, :, channel], bins=16, range=(0.0, 1.0), density=False)
            hist_parts.append(hist.astype(np.float32))
        gray = np.asarray(pil_image.convert("L"), dtype=np.float32) / 255.0
        texture = np.array(
            [
                float(gray.mean()),
                float(gray.std()),
                float(np.abs(np.diff(gray, axis=0)).mean()),
                float(np.abs(np.diff(gray, axis=1)).mean()),
            ],
            dtype=np.float32,
        )
        return l2_normalize(np.concatenate(hist_parts + [texture], axis=0))


def l2_normalize(vector: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= eps:
        raise ValueError("Cannot normalize a zero vector.")
    return array / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(l2_normalize(left), l2_normalize(right)))


def load_pil_image(image: Image.Image | np.ndarray | str | Path) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("Input image is empty.")
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGB")
    path = Path(image)
    with Image.open(path) as opened:
        return opened.convert("RGB")


def run_vehicle_reid_experiment(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    embedder: VehicleAppearanceEmbedder | None = None,
    positive_track_ids: list[str] | None = None,
    comparison_track_ids: list[str] | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_path}")
    tracks_hash_before = sha256_file(run_path / "tracks.json")
    experiment_dir = Path(output_dir).expanduser().resolve() if output_dir else run_path / "vehicle_reid_test"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "embeddings").mkdir(exist_ok=True)
    (experiment_dir / "visual_evidence").mkdir(exist_ok=True)

    positive_ids = positive_track_ids or list(POSITIVE_TRACK_GROUP)
    comparison_ids = comparison_track_ids or list(DEFAULT_COMPARISON_TRACKS)
    requested_track_ids = dedupe_preserve_order(positive_ids + comparison_ids)
    tracklets = {track.local_track_id: track for track in load_tracklets(run_path)}
    selected_by_track = select_crops_for_tracks(run_path, requested_track_ids)
    selected_track_ids = [track_id for track_id in requested_track_ids if selected_by_track.get(track_id)]
    if embedder is None:
        embedder = FlorenceVisionAppearanceEmbedder(_model_path_from_run_config(run_path))

    track_embeddings: dict[str, np.ndarray] = {}
    crop_embeddings: dict[str, np.ndarray] = {}
    for track_id in selected_track_ids:
        crop_paths = [item.crop_path for item in selected_by_track[track_id]]
        for crop in selected_by_track[track_id]:
            key = crop_key(crop)
            crop_embeddings[key] = l2_normalize(embedder.embed_image(crop.crop_path))
        track_embeddings[track_id] = embedder.embed_track(crop_paths)
        np.save(experiment_dir / "embeddings" / safe_track_filename(track_id), track_embeddings[track_id])

    pair_rows = build_pairwise_rows(
        selected_track_ids,
        positive_ids,
        track_embeddings,
        tracklets,
        reconciliation_result=load_reconciliation_result(run_path),
    )
    crop_rows = build_crop_similarity_rows(selected_by_track, crop_embeddings, positive_ids)
    matrix = build_similarity_matrix(selected_track_ids, track_embeddings)
    positives = [row["reid_similarity"] for row in pair_rows if row["manual_result"] == "SAME_VEHICLE"]
    negatives = [row["reid_similarity"] for row in pair_rows if row["manual_result"] == "KNOWN_NEGATIVE"]
    groups = build_reid_groups(positive_ids, track_embeddings)
    review_rows = build_review_rows(pair_rows)

    write_selected_crops(experiment_dir / "selected_crops.json", selected_by_track)
    write_csv(experiment_dir / "pairwise_similarity.csv", pair_rows)
    write_csv(experiment_dir / "crop_similarity.csv", crop_rows)
    write_similarity_matrix(experiment_dir, selected_track_ids, matrix)
    write_csv(experiment_dir / "manual_validation.csv", review_rows)
    (experiment_dir / "reid_groups.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
    generate_contact_sheets(experiment_dir / "visual_evidence", selected_by_track, pair_rows, tracklets)
    tracks_hash_after = sha256_file(run_path / "tracks.json")

    model_metadata = getattr(embedder, "metadata", {"kind": "GENERIC APPEARANCE EMBEDDING BASELINE", "model": type(embedder).__name__})
    result = {
        "source_run_directory": str(run_path),
        "output_directory": str(experiment_dir),
        "model": model_metadata,
        "positive_track_ids": positive_ids,
        "comparison_track_ids": [track_id for track_id in selected_track_ids if track_id not in positive_ids],
        "candidate_crop_counts": build_crop_count_summary(run_path, selected_track_ids, selected_by_track),
        "positive_distribution": distribution(positives),
        "negative_distribution": distribution(negatives),
        "minimum_positive_similarity": min(positives) if positives else None,
        "maximum_negative_similarity": max(negatives) if negatives else None,
        "separation_quality": classify_separation(positives, negatives),
        "candidate_threshold": candidate_threshold(positives, negatives),
        "tracks_json_sha256_before": tracks_hash_before,
        "tracks_json_sha256_after": tracks_hash_after,
        "tracks_json_unchanged": tracks_hash_before == tracks_hash_after,
        "pairwise_similarity_csv": str(experiment_dir / "pairwise_similarity.csv"),
        "similarity_matrix_csv": str(experiment_dir / "similarity_matrix.csv"),
        "crop_similarity_csv": str(experiment_dir / "crop_similarity.csv"),
        "reid_groups_json": str(experiment_dir / "reid_groups.json"),
    }
    (experiment_dir / "vehicle_reid_test.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(experiment_dir / "report.md", result, pair_rows, groups)
    if not result["tracks_json_unchanged"]:
        raise RuntimeError("tracks.json changed during ReID experiment.")
    return result


def select_crops_for_tracks(run_path: Path, track_ids: list[str], *, max_per_track: int = 5) -> dict[str, list[SelectedCrop]]:
    selected: dict[str, list[SelectedCrop]] = {}
    for track_id in track_ids:
        evidence_items = load_evidence_items(run_path, track_id)
        selected_from_evidence = choose_representative_evidence(evidence_items, max_per_track=max_per_track)
        if selected_from_evidence:
            selected[track_id] = selected_from_evidence
            continue
        selected_from_florence = load_florence_selected_crops(run_path, track_id)
        selected[track_id] = selected_from_florence[:max_per_track]
    return selected


def load_evidence_items(run_path: Path, track_id: str) -> list[SelectedCrop]:
    evidence_path = run_path / "evidence" / track_id.split(":")[0] / track_id.replace(":", "_") / "evidence.json"
    if not evidence_path.exists():
        return []
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    items: list[SelectedCrop] = []
    for item in payload if isinstance(payload, list) else []:
        crop_path = str(item.get("crop_path") or "")
        if not crop_path or not Path(crop_path).exists():
            continue
        role = str(item.get("role") or "UNKNOWN")
        frame = int(item.get("frame_number", -1))
        items.append(
            SelectedCrop(
                local_track_id=track_id,
                role=role,
                frame_number=frame,
                crop_path=crop_path,
                source="evidence",
                reason="selected from evidence.json using role diversity, quality score, crop size, confidence, and sharpness",
                confidence=to_optional_float(item.get("confidence")),
                original_crop_width=to_optional_int(item.get("original_crop_width")),
                original_crop_height=to_optional_int(item.get("original_crop_height")),
                sharpness_score=to_optional_float(item.get("sharpness_score")),
                best_overall_score=to_optional_float(item.get("best_overall_score")),
            )
        )
    return items


def choose_representative_evidence(items: list[SelectedCrop], *, max_per_track: int) -> list[SelectedCrop]:
    if not items:
        return []
    role_priority = ["BEST_OVERALL", "HIGHEST_CONFIDENCE", "LARGEST", "SHARPEST", "MIDDLE", "FIRST", "LAST"]
    by_role: dict[str, SelectedCrop] = {}
    for role in role_priority:
        candidates = [item for item in items if item.role == role]
        if candidates:
            by_role[role] = max(candidates, key=crop_quality_key)
    ranked = sorted(items, key=crop_quality_key, reverse=True)
    chosen: list[SelectedCrop] = []
    for item in list(by_role.values()) + ranked:
        if item.crop_path not in {existing.crop_path for existing in chosen}:
            chosen.append(item)
        if len(chosen) >= max_per_track:
            break
    return sorted(chosen, key=lambda item: (item.frame_number, item.role))


def crop_quality_key(item: SelectedCrop) -> tuple[float, int, float, float]:
    area = int(item.original_crop_width or 0) * int(item.original_crop_height or 0)
    return (
        float(item.best_overall_score if item.best_overall_score is not None else 0.0),
        area,
        float(item.confidence if item.confidence is not None else 0.0),
        float(item.sharpness_score if item.sharpness_score is not None else 0.0),
    )


def load_florence_selected_crops(run_path: Path, track_id: str) -> list[SelectedCrop]:
    camera_id = track_id.split(":")[0]
    short = track_id.split(":")[-1]
    directory = run_path / "05_florence_selected_crops" / camera_id / short
    if not directory.exists():
        return []
    items: list[SelectedCrop] = []
    for path in sorted(directory.glob("*.jpg")):
        frame, role = parse_crop_filename(path.name)
        items.append(
            SelectedCrop(
                local_track_id=track_id,
                role=role,
                frame_number=frame,
                crop_path=str(path),
                source="05_florence_selected_crops",
                reason="fallback to Florence-selected crop because evidence metadata was unavailable",
            )
        )
    return items


def build_pairwise_rows(
    track_ids: list[str],
    positive_track_ids: list[str],
    track_embeddings: dict[str, np.ndarray],
    tracklets: dict[str, Any],
    *,
    reconciliation_result: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    recon_lookup = build_reconciliation_lookup(reconciliation_result)
    positive_set = set(positive_track_ids)
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(track_ids):
        for right in track_ids[i + 1 :]:
            left_track = tracklets.get(left)
            right_track = tracklets.get(right)
            recon = recon_lookup.get((left, right)) or recon_lookup.get((right, left)) or {}
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
                    "position_score": component(recon, "position"),
                    "motion_score": component(recon, "motion"),
                    "direction_score": component(recon, "direction"),
                    "existing_reconciliation_decision": recon.get("result") or ("REJECTED" if recon else "NOT_ATTEMPTED"),
                    "reid_similarity": round(cosine_similarity(track_embeddings[left], track_embeddings[right]), 6),
                    "manual_result": "SAME_VEHICLE" if left in positive_set and right in positive_set else "KNOWN_NEGATIVE",
                }
            )
    return sorted(rows, key=lambda row: float(row["reid_similarity"]), reverse=True)


def build_crop_similarity_rows(
    selected_by_track: dict[str, list[SelectedCrop]],
    crop_embeddings: dict[str, np.ndarray],
    positive_track_ids: list[str],
) -> list[dict[str, Any]]:
    positive_set = set(positive_track_ids)
    crops = [crop for crops_for_track in selected_by_track.values() for crop in crops_for_track]
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(crops):
        for right in crops[i + 1 :]:
            if left.local_track_id == right.local_track_id:
                continue
            rows.append(
                {
                    "left_track": left.local_track_id,
                    "left_role": left.role,
                    "left_frame": left.frame_number,
                    "left_crop": left.crop_path,
                    "right_track": right.local_track_id,
                    "right_role": right.role,
                    "right_frame": right.frame_number,
                    "right_crop": right.crop_path,
                    "reid_similarity": round(cosine_similarity(crop_embeddings[crop_key(left)], crop_embeddings[crop_key(right)]), 6),
                    "manual_result": "SAME_VEHICLE"
                    if left.local_track_id in positive_set and right.local_track_id in positive_set
                    else "KNOWN_NEGATIVE",
                }
            )
    return sorted(rows, key=lambda row: float(row["reid_similarity"]), reverse=True)


def build_similarity_matrix(track_ids: list[str], embeddings: dict[str, np.ndarray]) -> list[list[float]]:
    matrix: list[list[float]] = []
    for left in track_ids:
        matrix.append([round(cosine_similarity(embeddings[left], embeddings[right]), 6) for right in track_ids])
    return matrix


def build_reid_groups(positive_track_ids: list[str], embeddings: dict[str, np.ndarray]) -> dict[str, Any]:
    pairwise = []
    all_consistent = True
    for i, left in enumerate(positive_track_ids):
        for right in positive_track_ids[i + 1 :]:
            score = cosine_similarity(embeddings[left], embeddings[right])
            pairwise.append({"left_track": left, "right_track": right, "reid_similarity": round(score, 6)})
            if not math.isfinite(score):
                all_consistent = False
    return {
        "groups": [
            {
                "vehicle_id": "VEHICLE_REID_TEST_001",
                "track_ids": list(positive_track_ids),
                "manual_ground_truth": "SAME_VEHICLE",
                "pairwise_consistency_checked": True,
                "pairwise_similarities": pairwise,
                "all_pairwise_edges_present": all_consistent,
            }
        ]
    }


def build_review_rows(pair_rows: list[dict[str, Any]], *, top_n: int = 12) -> list[dict[str, Any]]:
    rows = [row for row in pair_rows if row["manual_result"] == "KNOWN_NEGATIVE"]
    rows = sorted(rows, key=lambda row: float(row["reid_similarity"]), reverse=True)[:top_n]
    return [
        {
            "old_track": row["left_track"],
            "new_track": row["right_track"],
            "class": f"{row['left_class']}->{row['right_class']}",
            "colour": f"{row['left_colour']}->{row['right_colour']}",
            "time_gap_frames": row["time_gap_frames"],
            "reid_similarity": row["reid_similarity"],
            "existing_reconciliation_decision": row["existing_reconciliation_decision"],
            "existing_reconciliation_score": row["existing_reconciliation_score"],
            "manual_label": "UNCERTAIN",
        }
        for row in rows
    ]


def generate_contact_sheets(output_dir: Path, selected_by_track: dict[str, list[SelectedCrop]], pair_rows: list[dict[str, Any]], tracklets: dict[str, Any]) -> None:
    positive_pairs = [row for row in pair_rows if row["manual_result"] == "SAME_VEHICLE"]
    negative_pairs = [row for row in pair_rows if row["manual_result"] == "KNOWN_NEGATIVE"][:6]
    for row in positive_pairs + negative_pairs:
        left = row["left_track"]
        right = row["right_track"]
        filename = f"{safe_id(left)}__{safe_id(right)}.jpg"
        make_contact_sheet(
            output_dir / filename,
            selected_by_track[left],
            selected_by_track[right],
            title=f"{left} vs {right}",
            subtitle=(
                f"similarity={row['reid_similarity']} manual={row['manual_result']} "
                f"class={row['left_class']}->{row['right_class']} colour={row['left_colour']}->{row['right_colour']}"
            ),
        )


def make_contact_sheet(path: Path, left_crops: list[SelectedCrop], right_crops: list[SelectedCrop], *, title: str, subtitle: str) -> None:
    thumb_w, thumb_h = 180, 140
    pad = 12
    header_h = 70
    columns = max(len(left_crops), len(right_crops), 1)
    width = columns * (thumb_w + pad) + pad
    height = header_h + 2 * (thumb_h + 44 + pad) + pad
    sheet = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 8), title, fill=(20, 20, 20), font=ImageFont.load_default())
    draw.text((pad, 30), subtitle[:160], fill=(40, 40, 40), font=ImageFont.load_default())
    for row_index, crops in enumerate([left_crops, right_crops]):
        y = header_h + row_index * (thumb_h + 44 + pad)
        for col, crop in enumerate(crops):
            x = pad + col * (thumb_w + pad)
            thumb = Image.open(crop.crop_path).convert("RGB")
            thumb.thumbnail((thumb_w, thumb_h))
            tile = Image.new("RGB", (thumb_w, thumb_h), (255, 255, 255))
            tile.paste(thumb, ((thumb_w - thumb.width) // 2, (thumb_h - thumb.height) // 2))
            sheet.paste(tile, (x, y))
            label = f"{crop.local_track_id.split(':')[-1]} {crop.role} f{crop.frame_number}"
            draw.text((x, y + thumb_h + 4), label[:28], fill=(20, 20, 20), font=ImageFont.load_default())
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def write_selected_crops(path: Path, selected_by_track: dict[str, list[SelectedCrop]]) -> None:
    payload = {track_id: [asdict(item) for item in crops] for track_id, crops in selected_by_track.items()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_similarity_matrix(output_dir: Path, track_ids: list[str], matrix: list[list[float]]) -> None:
    with (output_dir / "similarity_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["track_id", *track_ids])
        for track_id, row in zip(track_ids, matrix):
            writer.writerow([track_id, *row])
    (output_dir / "similarity_matrix.json").write_text(
        json.dumps({"track_ids": track_ids, "matrix": matrix}, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, result: dict[str, Any], pair_rows: list[dict[str, Any]], groups: dict[str, Any]) -> None:
    positive_rows = [row for row in pair_rows if row["manual_result"] == "SAME_VEHICLE"]
    hardest_negative = next((row for row in pair_rows if row["manual_result"] == "KNOWN_NEGATIVE"), None)
    lines = [
        "# Vehicle ReID Appearance Test",
        "",
        f"- Model kind: `{result['model'].get('kind')}`",
        f"- Model: `{result['model'].get('model')}`",
        f"- Source run: `{result['source_run_directory']}`",
        f"- Tracks JSON unchanged: `{result['tracks_json_unchanged']}`",
        f"- Separation quality: `{result['separation_quality']}`",
        f"- Candidate threshold: `{result['candidate_threshold']}`",
        "",
        "## Known Positive Similarities",
        "",
    ]
    for row in positive_rows:
        lines.append(f"- `{row['left_track']}` vs `{row['right_track']}`: `{row['reid_similarity']}`")
    lines.extend(["", "## Hardest Negative", ""])
    if hardest_negative:
        lines.append(
            f"- `{hardest_negative['left_track']}` vs `{hardest_negative['right_track']}`: "
            f"`{hardest_negative['reid_similarity']}`"
        )
    lines.extend(["", "## Grouping", "", "```json", json.dumps(groups, indent=2), "```"])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_crop_count_summary(run_path: Path, track_ids: list[str], selected_by_track: dict[str, list[SelectedCrop]]) -> dict[str, dict[str, int]]:
    summary_rows = read_csv_dicts(run_path / "vehicle_enrichment_track_evidence_summary.csv")
    by_track = {row.get("local_track_id", ""): row for row in summary_rows}
    summary: dict[str, dict[str, int]] = {}
    for track_id in track_ids:
        row = by_track.get(track_id, {})
        usable_evidence_count = len({item.crop_path for item in load_evidence_items(run_path, track_id)})
        summary[track_id] = {
            "candidate_crop_count": to_int(row.get("candidate_crops_seen"), usable_evidence_count),
            "usable_crop_count": usable_evidence_count,
            "selected_crop_count": len(selected_by_track.get(track_id, [])),
        }
    return summary


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "maximum": None, "mean": None, "median": None}
    array = np.asarray(values, dtype=np.float32)
    return {
        "count": int(array.size),
        "minimum": round(float(array.min()), 6),
        "maximum": round(float(array.max()), 6),
        "mean": round(float(array.mean()), 6),
        "median": round(float(np.median(array)), 6),
    }


def classify_separation(positives: list[float], negatives: list[float]) -> str:
    if not positives or not negatives:
        return "INSUFFICIENT_DATA"
    margin = min(positives) - max(negatives)
    if margin >= 0.05:
        return "STRONG_SEPARATION"
    if margin > 0.0:
        return "MODERATE_SEPARATION"
    return "POOR_SEPARATION"


def candidate_threshold(positives: list[float], negatives: list[float]) -> float | str | None:
    if not positives or not negatives:
        return None
    positive_min = min(positives)
    negative_max = max(negatives)
    if positive_min <= negative_max:
        return "NO_SAFE_SINGLE_REID_THRESHOLD_EXISTS"
    return round(float((positive_min + negative_max) / 2.0), 6)


def load_reconciliation_result(run_path: Path) -> dict[str, Any] | None:
    path = run_path / "track_reconciliation_test" / "track_reconciliation_test.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_reconciliation_lookup(result: dict[str, Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if not result:
        return lookup
    accepted_keys = {
        (str(item.get("old_track")), str(item.get("new_track")))
        for item in result.get("accepted_associations", [])
        if isinstance(item, dict)
    }
    for item in result.get("attempts", []):
        if not isinstance(item, dict):
            continue
        key = (str(item.get("old_track_id")), str(item.get("new_track_id")))
        lookup[(str(item.get("old_track_id")), str(item.get("new_track_id")))] = {
            "score": item.get("score"),
            "result": "ACCEPTED" if key in accepted_keys else "REJECTED" if item.get("rejected") else "SCORED_NOT_SELECTED",
            "components": item.get("components") or {},
        }
    for item in result.get("accepted_associations", []):
        lookup[(str(item.get("old_track")), str(item.get("new_track")))] = {
            "score": item.get("score"),
            "result": item.get("result", "ACCEPTED"),
            "components": lookup.get((str(item.get("old_track")), str(item.get("new_track"))), {}).get("components", {}),
        }
    return lookup


def component(recon: dict[str, Any], name: str) -> Any:
    return (recon.get("components") or {}).get(name)


def frame_gap(left: Any, right: Any) -> int | None:
    if left is None or right is None:
        return None
    if left.last_frame <= right.first_frame:
        return int(right.first_frame - left.last_frame)
    if right.last_frame <= left.first_frame:
        return int(left.first_frame - right.last_frame)
    return 0


def _model_path_from_run_config(run_path: Path) -> str:
    try:
        import yaml

        payload = yaml.safe_load((run_path / "run_config.yaml").read_text(encoding="utf-8")) or {}
        florence = payload.get("vehicle_enrichment", {}).get("florence", {})
        return str(florence.get("model_id") or florence.get("base_model_id") or "D:/project/models/Florence-2-base-ft")
    except Exception:
        return "D:/project/models/Florence-2-base-ft"


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_crop_filename(name: str) -> tuple[int, str]:
    stem = Path(name).stem
    parts = stem.split("_")
    frame = -1
    role = "UNKNOWN"
    for index, part in enumerate(parts):
        if part.isdigit():
            frame = int(part)
            role = "_".join(parts[index + 1 :]) or "UNKNOWN"
            break
    return frame, role


def crop_key(crop: SelectedCrop) -> str:
    return f"{crop.local_track_id}|{crop.frame_number}|{crop.role}|{crop.crop_path}"


def safe_id(track_id: str) -> str:
    return track_id.replace(":", "_")


def safe_track_filename(track_id: str) -> str:
    return f"{safe_id(track_id)}.npy"


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_optional_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def to_optional_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def to_int(value: Any, default: int) -> int:
    parsed = to_optional_int(value)
    return default if parsed is None else parsed
