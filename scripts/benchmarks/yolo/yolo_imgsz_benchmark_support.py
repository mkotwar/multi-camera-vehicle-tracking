from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_SMALL_AREA_RATIO_THRESHOLD = 0.015
DEFAULT_MEDIUM_AREA_RATIO_THRESHOLD = 0.06


@dataclass(slots=True, frozen=True)
class BenchmarkDetection:
    camera_id: str
    frame_number: int
    frame_path: str
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    frame_width: int
    frame_height: int
    imgsz: int

    @property
    def area_ratio(self) -> float:
        frame_area = float(self.frame_width * self.frame_height)
        if frame_area <= 0.0:
            return 0.0
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, float((x2 - x1) * (y2 - y1)) / frame_area)


def compute_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def classify_size_group(
    area_ratio: float,
    *,
    small_threshold: float = DEFAULT_SMALL_AREA_RATIO_THRESHOLD,
    medium_threshold: float = DEFAULT_MEDIUM_AREA_RATIO_THRESHOLD,
) -> str:
    if area_ratio < small_threshold:
        return "small"
    if area_ratio < medium_threshold:
        return "medium"
    return "large"


def detection_key(detection: BenchmarkDetection) -> str:
    rounded_box = ",".join(f"{value:.3f}" for value in detection.bbox_xyxy)
    return f"{detection.camera_id}|{detection.frame_number}|{detection.class_name}|{rounded_box}|{detection.confidence:.6f}"


def match_frame_detections(
    baseline: list[BenchmarkDetection],
    candidate: list[BenchmarkDetection],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, Any]:
    match_candidates: list[tuple[float, int, int]] = []
    for baseline_index, baseline_detection in enumerate(baseline):
        for candidate_index, candidate_detection in enumerate(candidate):
            if baseline_detection.class_name != candidate_detection.class_name:
                continue
            iou = compute_iou(baseline_detection.bbox_xyxy, candidate_detection.bbox_xyxy)
            if iou >= iou_threshold:
                match_candidates.append((iou, baseline_index, candidate_index))
    match_candidates.sort(key=lambda item: item[0], reverse=True)
    used_baseline: set[int] = set()
    used_candidate: set[int] = set()
    matched: list[dict[str, Any]] = []
    for iou, baseline_index, candidate_index in match_candidates:
        if baseline_index in used_baseline or candidate_index in used_candidate:
            continue
        used_baseline.add(baseline_index)
        used_candidate.add(candidate_index)
        baseline_detection = baseline[baseline_index]
        candidate_detection = candidate[candidate_index]
        matched.append(
            {
                "baseline": baseline_detection,
                "candidate": candidate_detection,
                "iou": iou,
                "confidence_delta": float(candidate_detection.confidence - baseline_detection.confidence),
            }
        )
    missing = [baseline[index] for index in range(len(baseline)) if index not in used_baseline]
    additional = [candidate[index] for index in range(len(candidate)) if index not in used_candidate]
    class_mismatch_count = 0
    mismatch_used_candidate: set[int] = set()
    for missing_detection in missing:
        best_candidate_index = -1
        best_iou = 0.0
        for candidate_index, candidate_detection in enumerate(additional):
            if candidate_index in mismatch_used_candidate:
                continue
            iou = compute_iou(missing_detection.bbox_xyxy, candidate_detection.bbox_xyxy)
            if iou >= iou_threshold and iou > best_iou and missing_detection.class_name != candidate_detection.class_name:
                best_iou = iou
                best_candidate_index = candidate_index
        if best_candidate_index >= 0:
            class_mismatch_count += 1
            mismatch_used_candidate.add(best_candidate_index)
    return {
        "matched": matched,
        "missing": missing,
        "additional": additional,
        "class_mismatch_count": class_mismatch_count,
    }


def summarize_parity(
    baseline: list[BenchmarkDetection],
    candidate: list[BenchmarkDetection],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    small_threshold: float = DEFAULT_SMALL_AREA_RATIO_THRESHOLD,
    medium_threshold: float = DEFAULT_MEDIUM_AREA_RATIO_THRESHOLD,
) -> dict[str, Any]:
    grouped_baseline: dict[tuple[str, int], list[BenchmarkDetection]] = {}
    grouped_candidate: dict[tuple[str, int], list[BenchmarkDetection]] = {}
    for item in baseline:
        grouped_baseline.setdefault((item.camera_id, item.frame_number), []).append(item)
    for item in candidate:
        grouped_candidate.setdefault((item.camera_id, item.frame_number), []).append(item)

    matches: list[dict[str, Any]] = []
    missing: list[BenchmarkDetection] = []
    additional: list[BenchmarkDetection] = []
    class_mismatch_count = 0
    for frame_key in sorted(set(grouped_baseline) | set(grouped_candidate)):
        frame_result = match_frame_detections(
            grouped_baseline.get(frame_key, []),
            grouped_candidate.get(frame_key, []),
            iou_threshold=iou_threshold,
        )
        matches.extend(frame_result["matched"])
        missing.extend(frame_result["missing"])
        additional.extend(frame_result["additional"])
        class_mismatch_count += int(frame_result["class_mismatch_count"])

    baseline_total = len(baseline)
    matched_total = len(matches)
    size_groups = {"small": {"baseline": 0, "matched": 0}, "medium": {"baseline": 0, "matched": 0}, "large": {"baseline": 0, "matched": 0}}
    per_class: dict[str, dict[str, int]] = {}
    missing_by_key = {detection_key(item) for item in missing}
    for item in baseline:
        size_group = classify_size_group(item.area_ratio, small_threshold=small_threshold, medium_threshold=medium_threshold)
        size_groups[size_group]["baseline"] += 1
        per_class.setdefault(item.class_name, {"baseline": 0, "matched": 0})
        per_class[item.class_name]["baseline"] += 1
        if detection_key(item) not in missing_by_key:
            size_groups[size_group]["matched"] += 1
            per_class[item.class_name]["matched"] += 1
    mean_iou = float(sum(item["iou"] for item in matches) / len(matches)) if matches else 0.0
    mean_confidence_delta = float(sum(item["confidence_delta"] for item in matches) / len(matches)) if matches else 0.0
    return {
        "total_baseline_detections": baseline_total,
        "matched_detections": matched_total,
        "missing_detections": len(missing),
        "additional_detections": len(additional),
        "match_rate": float(matched_total / baseline_total) if baseline_total else 0.0,
        "mean_bbox_iou": mean_iou,
        "mean_confidence_delta": mean_confidence_delta,
        "class_mismatch_count": class_mismatch_count,
        "size_groups": {
            key: {
                "baseline": value["baseline"],
                "matched": value["matched"],
                "match_rate": float(value["matched"] / value["baseline"]) if value["baseline"] else 0.0,
            }
            for key, value in size_groups.items()
        },
        "per_class": {
            key: {
                "baseline": value["baseline"],
                "matched": value["matched"],
                "match_rate": float(value["matched"] / value["baseline"]) if value["baseline"] else 0.0,
            }
            for key, value in sorted(per_class.items())
        },
        "matches": matches,
        "missing": missing,
        "additional": additional,
    }
