from __future__ import annotations

import re

from ..schemas import VEHICLE_BODY_TYPE_UNKNOWN


BODY_TYPE_TASK_PROMPT = "<VQA>"
BODY_TYPE_PROMPT_TEXT = (
    "What type of car is shown in this image?\nAnswer with one word only:\nsedan, hatchback, suv, or mpv."
)

BODY_TYPE_ALLOWED_LABELS: tuple[str, ...] = (
    "SEDAN",
    "HATCHBACK",
    "SUV",
    "MPV",
    "VAN",
    "PICKUP",
    "COUPE",
    "CONVERTIBLE",
    "WAGON",
    VEHICLE_BODY_TYPE_UNKNOWN,
)

BODY_TYPE_NORMALIZATION_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("SUV", ("sport utility vehicle", "sports utility vehicle", "suv")),
    ("SEDAN", ("sedan", "saloon")),
    ("HATCHBACK", ("hatchback", "hatch back")),
    ("MPV", ("multi purpose vehicle", "multi-purpose vehicle", "multipurpose vehicle", "multi utility vehicle", "muv", "mpv", "minivan")),
    ("VAN", ("van",)),
    ("PICKUP", ("pickup truck", "pickup", "pick-up", "pick up")),
    ("COUPE", ("coupé", "coupe")),
    ("CONVERTIBLE", ("convertible", "cabriolet")),
    ("WAGON", ("station wagon", "estate", "wagon")),
)

BODY_TYPE_UNKNOWN_PHRASES = {
    "",
    "unknown",
    "unclear",
    "not visible",
    "cannot determine",
    "cant determine",
    "unable to determine",
    "cannot classify",
    "not sure",
    "unsure",
    "unanswerable",
    "car",
    "vehicle",
    "body type",
}


def normalize_body_type_label(raw_value: str) -> tuple[str, str]:
    cleaned = " ".join(str(raw_value or "").strip().lower().replace("_", " ").split())
    cleaned = re.sub(r"[^\w\s-]", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    if cleaned in BODY_TYPE_UNKNOWN_PHRASES:
        return VEHICLE_BODY_TYPE_UNKNOWN, "unknown_phrase"
    exact_matches = [label for label, phrases in BODY_TYPE_NORMALIZATION_RULES if cleaned in phrases]
    if len(exact_matches) == 1:
        return exact_matches[0], "exact_phrase_match"
    matches: list[str] = []
    for label, phrases in BODY_TYPE_NORMALIZATION_RULES:
        for phrase in phrases:
            if re.search(rf"\b{re.escape(phrase)}\b", cleaned):
                matches.append(label)
                break
    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return unique_matches[0], "contained_phrase_match"
    if len(unique_matches) > 1:
        return VEHICLE_BODY_TYPE_UNKNOWN, "ambiguous_multiple_labels"
    return VEHICLE_BODY_TYPE_UNKNOWN, "unexpected_output"
