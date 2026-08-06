from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .ocr_mukul.attribute_parser import OCR_MUKUL_UNKNOWN, ParsedCaptionAttributes, looks_like_plate_text, parse_caption_attributes


@dataclass(slots=True, frozen=True)
class GenerationProfile:
    name: str
    max_new_tokens: int
    num_beams: int
    do_sample: bool
    use_cache: bool
    early_stopping: bool


@dataclass(slots=True, frozen=True)
class PromptPreset:
    name: str
    attribute_task: str
    task_token: str
    prompt: str
    generation_profile: str


GENERATION_PROFILES: dict[str, GenerationProfile] = {
    "short_vqa": GenerationProfile("short_vqa", max_new_tokens=16, num_beams=1, do_sample=False, use_cache=True, early_stopping=False),
    "beam_vqa": GenerationProfile("beam_vqa", max_new_tokens=16, num_beams=3, do_sample=False, use_cache=True, early_stopping=True),
    "caption": GenerationProfile("caption", max_new_tokens=64, num_beams=3, do_sample=False, use_cache=True, early_stopping=True),
}


PROMPT_PRESETS: dict[str, PromptPreset] = {
    "colour_vqa_1": PromptPreset("colour_vqa_1", "colour", "<VQA>", "What colour is the vehicle?", "short_vqa"),
    "colour_vqa_2": PromptPreset("colour_vqa_2", "colour", "<VQA>", "What is the exterior colour of the vehicle?", "short_vqa"),
    "colour_vqa_3": PromptPreset("colour_vqa_3", "colour", "<VQA>", "Name the main colour of the vehicle.", "short_vqa"),
    "colour_vqa_4": PromptPreset("colour_vqa_4", "colour", "<VQA>", "What colour is the main vehicle? Answer with one colour word.", "short_vqa"),
    "body_vqa_1": PromptPreset("body_vqa_1", "body_type", "<VQA>", "What body type is the vehicle?", "short_vqa"),
    "body_vqa_2": PromptPreset("body_vqa_2", "body_type", "<VQA>", "What kind of vehicle body is shown?", "short_vqa"),
    "body_vqa_3": PromptPreset("body_vqa_3", "body_type", "<VQA>", "Is this vehicle an SUV, sedan, hatchback, MPV, van, or pickup?", "short_vqa"),
    "body_caption": PromptPreset("body_caption", "body_type", "<CAPTION>", "", "caption"),
    "body_detailed_caption": PromptPreset("body_detailed_caption", "body_type", "<MORE_DETAILED_CAPTION>", "", "caption"),
}


def get_prompt_preset(name: str) -> PromptPreset:
    return PROMPT_PRESETS[str(name).strip()]


def get_generation_profile(name: str) -> GenerationProfile:
    return GENERATION_PROFILES[str(name).strip()]


def parse_preset_names(raw_value: str) -> list[str]:
    return [item.strip() for item in str(raw_value or "").split(",") if item.strip()]


def assess_response_quality(raw_response: str, parsed: ParsedCaptionAttributes, *, attribute_task: str, prompt: str) -> tuple[str, str]:
    cleaned = re.sub(r"\s+", " ", str(raw_response or "").strip().lower())
    cleaned_prompt = re.sub(r"\s+", " ", str(prompt or "").strip().lower())
    if not cleaned:
        return "invalid", "empty_response"
    if looks_like_plate_text(raw_response):
        return "invalid", "plate_like_response"
    if cleaned_prompt and cleaned == cleaned_prompt:
        return "invalid", "prompt_echo"
    if cleaned in {"vehicle", "car", "body type", "vehicle colour", "color", "colour"}:
        return "invalid", "generic_response"
    if attribute_task == "colour":
        if parsed.normalized_colour == OCR_MUKUL_UNKNOWN:
            if "colour" in cleaned or "color" in cleaned:
                return "invalid", "missing_colour_value"
            return "invalid", "unsupported_colour"
        return "valid", "valid"
    if parsed.normalized_body_type == OCR_MUKUL_UNKNOWN:
        if "body type" in cleaned:
            return "invalid", "missing_body_type_value"
        return "invalid", "unsupported_body_type"
    return "valid", "valid"
