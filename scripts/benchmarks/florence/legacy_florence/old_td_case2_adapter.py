from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import time
from typing import Any

from src.runtime_device import move_batch_to_device
from src.vehicle_enrichment.schemas import EnrichmentEvidenceItem
from src.vehicle_enrichment.shared.florence_backend import FlorenceBackend
from .old_crop_preprocessor import OldTdCase2CropPreprocessor
from .old_prompt_adapter import LEGACY_CAPTION_TASK_PROMPT, parse_old_td_case2_caption


@dataclass(slots=True, frozen=True)
class LegacySelectionResult:
    local_track_id: str
    selected_crop_paths: list[str]
    selected_frame_numbers: list[int]
    scores_by_crop_path: dict[str, float]


@dataclass(slots=True, frozen=True)
class LegacyFlorenceInferenceResult:
    crop_path: str
    task_prompt: str
    raw_response: str
    post_processed_output: Any
    parsed_body_type_text: str | None
    parsed_colour_text: str | None
    body_type_label: str
    colour_label: str
    prompt_text: str | None
    original_width: int
    original_height: int
    preprocessed_width: int
    preprocessed_height: int
    tensor_shape: list[int]
    inference_time_ms: float
    decode_mode: str
    manual_resize_applied: bool
    manual_square_padding_applied: bool


def inspect_old_reference_project(old_project_root: str | Path) -> dict[str, Any]:
    root = Path(old_project_root)
    if not root.exists():
        raise FileNotFoundError(f"Old reference project does not exist: {root}")

    candidate_files = [
        root / "step_03b_yolo_detection.py",
        root / "step_04a_florence_model_audit.py",
        root / "step_05_best_track_frame_selector.py",
        root / "step_06_ocr_color_enrichment.py",
        root / "vehicle_color.py",
        root / "streaming_tracking_pipeline" / "florence_inference.py",
        root / "streaming_tracking_pipeline" / "anpr_pipeline.py",
        root / "streaming_tracking_pipeline" / "crop_selection.py",
        root / "streaming_tracking_pipeline" / "vision_backends" / "florence_backend.py",
    ]
    existing = [path for path in candidate_files if path.exists()]
    return {
        "old_project_root": str(root),
        "found_files": [str(path) for path in existing],
        "missing_files": [str(path) for path in candidate_files if not path.exists()],
        "step_path_summary": {
            "detection": str(root / "step_03b_yolo_detection.py"),
            "best_frame_selection": str(root / "step_05_best_track_frame_selector.py"),
            "step06_enrichment": str(root / "step_06_ocr_color_enrichment.py"),
            "streaming_inference": str(root / "streaming_tracking_pipeline" / "florence_inference.py"),
        },
    }


class OldTdCase2Adapter:
    """
    Reproduced from:
    - D:\old_files\reference_pro\Final_vedio_Ai_system\tests\td_case2\step_04a_florence_model_audit.py
    - D:\old_files\reference_pro\Final_vedio_Ai_system\tests\td_case2\step_05_best_track_frame_selector.py
    - D:\old_files\reference_pro\Final_vedio_Ai_system\tests\td_case2\step_06_ocr_color_enrichment.py
    """

    def __init__(self, backend: FlorenceBackend, *, logger: logging.Logger | None = None) -> None:
        self.backend = backend
        self.logger = logger or logging.getLogger(__name__)
        self.preprocessor = OldTdCase2CropPreprocessor()

    def select_track_evidence(
        self,
        evidence_items: list[EnrichmentEvidenceItem],
        *,
        maximum_crops_per_track: int = 3,
        minimum_frame_gap: int = 3,
    ) -> LegacySelectionResult:
        if not evidence_items:
            return LegacySelectionResult("", [], [], {})
        ordered = sorted(evidence_items, key=lambda item: item.frame_number)
        areas = [float(max(0, item.crop_area)) for item in ordered]
        area_min = min(areas) if areas else 0.0
        area_max = max(areas) if areas else 0.0
        scored: list[tuple[float, EnrichmentEvidenceItem]] = []
        total = max(1, len(ordered))
        for index, item in enumerate(ordered):
            confidence_score = max(0.0, min(1.0, float(item.detection_confidence)))
            if area_max > area_min:
                bbox_area_score = max(0.0, min(1.0, (float(item.crop_area) - area_min) / (area_max - area_min)))
            else:
                bbox_area_score = 1.0 if float(item.crop_area) > 0 else 0.0
            not_border_touching_score = 1.0 - max(0.0, min(1.0, float(item.border_penalty)))
            temporal_score = self._temporal_position_score(index, total)
            crop_exists_score = 1.0 if item.vehicle_crop_path else 0.0
            final_score = round(
                0.35 * confidence_score
                + 0.25 * bbox_area_score
                + 0.15 * not_border_touching_score
                + 0.10 * 1.0
                + 0.10 * temporal_score
                + 0.05 * crop_exists_score,
                6,
            )
            scored.append((final_score, item))
        scored.sort(
            key=lambda pair: (
                pair[0],
                pair[1].detection_confidence,
                pair[1].crop_area,
                -pair[1].border_penalty,
            ),
            reverse=True,
        )
        selected: list[EnrichmentEvidenceItem] = []
        for score, item in scored:
            if item.vehicle_crop_path is None:
                continue
            too_close = any(abs(item.frame_number - existing.frame_number) < minimum_frame_gap for existing in selected)
            if too_close:
                continue
            selected.append(item)
            if len(selected) >= maximum_crops_per_track:
                break
        if not selected and scored:
            first = next((item for _score, item in scored if item.vehicle_crop_path), None)
            if first is not None:
                selected.append(first)
        return LegacySelectionResult(
            local_track_id=ordered[0].local_track_id,
            selected_crop_paths=[str(item.vehicle_crop_path) for item in selected if item.vehicle_crop_path],
            selected_frame_numbers=[int(item.frame_number) for item in selected if item.vehicle_crop_path],
            scores_by_crop_path={
                str(item.vehicle_crop_path): score
                for score, item in scored
                if item.vehicle_crop_path
            },
        )

    def run_caption_inference(self, crop_path: str | Path) -> LegacyFlorenceInferenceResult:
        prepared = self.preprocessor.prepare_from_crop_path(crop_path)
        self.backend.load()
        if self.backend._processor is None or self.backend._model is None or self.backend._runtime_device is None:
            raise RuntimeError("Shared Florence backend is not loaded for legacy comparison.")
        processor = self.backend._processor
        model = self.backend._model
        runtime_device = self.backend._runtime_device

        import torch

        started = time.perf_counter()
        inputs = processor(text=LEGACY_CAPTION_TASK_PROMPT, images=prepared.image_rgb, return_tensors="pt")
        tensor_shape = list(getattr(inputs.get("pixel_values"), "shape", []) or [])
        inputs = move_batch_to_device(
            dict(inputs),
            device=runtime_device.device,
            dtype=runtime_device.dtype,
        )
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=self.backend.config.max_new_tokens,
            num_beams=self.backend.config.num_beams,
            do_sample=False,
            use_cache=False,
        )
        decoded_list = processor.batch_decode(generated_ids, skip_special_tokens=True)
        raw_decoded_text = decoded_list[0].strip() if decoded_list else ""
        post_processed_output: Any = None
        if hasattr(processor, "post_process_generation"):
            try:
                post_processed_output = processor.post_process_generation(
                    raw_decoded_text,
                    task=LEGACY_CAPTION_TASK_PROMPT,
                    image_size=(prepared.original_width, prepared.original_height),
                )
            except Exception:
                post_processed_output = None
        parsed_caption = self._extract_text(post_processed_output, LEGACY_CAPTION_TASK_PROMPT) or raw_decoded_text
        parsed = parse_old_td_case2_caption(parsed_caption)
        return LegacyFlorenceInferenceResult(
            crop_path=str(crop_path),
            task_prompt=LEGACY_CAPTION_TASK_PROMPT,
            raw_response=parsed_caption,
            post_processed_output=post_processed_output,
            parsed_body_type_text=parsed.parsed_body_type_text,
            parsed_colour_text=parsed.parsed_colour_text,
            body_type_label=parsed.normalized_body_type,
            colour_label=parsed.normalized_colour,
            prompt_text=None,
            original_width=prepared.original_width,
            original_height=prepared.original_height,
            preprocessed_width=prepared.preprocessed_width,
            preprocessed_height=prepared.preprocessed_height,
            tensor_shape=tensor_shape,
            inference_time_ms=float((time.perf_counter() - started) * 1000.0),
            decode_mode="batch_decode_skip_special_plus_post_process_generation",
            manual_resize_applied=prepared.manual_resize_applied,
            manual_square_padding_applied=prepared.manual_square_padding_applied,
        )

    @staticmethod
    def _temporal_position_score(index: int, total_count: int) -> float:
        if total_count <= 1:
            return 1.0
        midpoint = (total_count - 1) / 2.0
        distance = abs(index - midpoint)
        return max(0.0, 1.0 - (distance / max(midpoint, 1.0)))

    @staticmethod
    def _extract_text(post_processed: Any, task_prompt: str) -> str:
        if post_processed is None:
            return ""
        if isinstance(post_processed, str):
            return post_processed
        if isinstance(post_processed, dict):
            if task_prompt in post_processed:
                return str(post_processed[task_prompt] or "")
            for value in post_processed.values():
                if isinstance(value, str) and value.strip():
                    return value
            return json.dumps(post_processed, ensure_ascii=False)
        if isinstance(post_processed, list):
            return " ".join(str(item) for item in post_processed if str(item).strip())
        return str(post_processed)
