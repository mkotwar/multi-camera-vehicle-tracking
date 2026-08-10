from __future__ import annotations

from dataclasses import dataclass
import re


LEGACY_CAPTION_TASK_PROMPT = "<CAPTION>"

# Reproduced from:
# D:\old_files\reference_pro\Final_vedio_Ai_system\tests\td_case2\step_06_ocr_color_enrichment.py
# function: extract_structured_florence_metadata
_LEGACY_BODY_TYPES = (
    "pickup truck",
    "truck",
    "bus",
    "minibus",
    "van",
    "motorcycle",
    "scooter",
    "bicycle",
    "auto rickshaw",
    "hatchback",
    "sedan",
    "suv",
)

# Reproduced from:
# D:\old_files\reference_pro\Final_vedio_Ai_system\tests\td_case2\vehicle_color.py
# functions: extract_florence_vehicle_color, normalize_color_phrase
_LEGACY_SHADE_TO_CANONICAL = {
    "pearl white": "white",
    "off white": "white",
    "off-white": "white",
    "ivory": "white",
    "cream": "white",
    "alabaster": "white",
    "snow white": "white",
    "white": "white",
    "jet black": "black",
    "matte black": "black",
    "midnight black": "black",
    "black": "black",
    "silver gray": "gray",
    "silver grey": "gray",
    "charcoal": "gray",
    "graphite": "gray",
    "gunmetal": "gray",
    "slate gray": "gray",
    "slate grey": "gray",
    "dark gray": "gray",
    "dark grey": "gray",
    "light gray": "gray",
    "light grey": "gray",
    "gray": "gray",
    "grey": "gray",
    "metallic silver": "silver",
    "silver": "silver",
    "burgundy": "red",
    "maroon": "red",
    "crimson": "red",
    "scarlet": "red",
    "wine red": "red",
    "ruby": "red",
    "red": "red",
    "navy blue": "blue",
    "navy": "blue",
    "metallic blue": "blue",
    "cobalt": "blue",
    "cerulean": "blue",
    "azure": "blue",
    "teal": "blue",
    "turquoise": "blue",
    "blue": "blue",
    "forest green": "green",
    "emerald": "green",
    "olive": "green",
    "lime": "green",
    "green": "green",
    "champagne gold": "gold",
    "champagne": "gold",
    "golden": "gold",
    "gold": "gold",
    "bronze": "brown",
    "copper": "brown",
    "chocolate": "brown",
    "coffee": "brown",
    "brown": "brown",
    "tan": "beige",
    "sand": "beige",
    "khaki": "beige",
    "beige": "beige",
    "mustard": "yellow",
    "lemon": "yellow",
    "yellow": "yellow",
    "amber": "orange",
    "rust": "orange",
    "orange": "orange",
    "violet": "purple",
    "plum": "purple",
    "lavender": "purple",
    "purple": "purple",
    "magenta": "pink",
    "rose": "pink",
    "pink": "pink",
}

_VEHICLE_NOUNS = (
    "car|vehicle|sedan|hatchback|suv|truck|pickup|van|minivan|bus|"
    "minibus|motorcycle|motorbike|scooter|auto rickshaw"
)
_COLOR_PHRASE_PATTERNS = (
    re.compile(rf"\b(?P<phrase>(?:[a-z][a-z-]*\s+){{1,5}})(?:{_VEHICLE_NOUNS})\b", re.IGNORECASE),
    re.compile(
        rf"\b(?:{_VEHICLE_NOUNS})\s+(?:is|appears|looks|painted|finished)\s+"
        r"(?P<phrase>(?:[a-z][a-z-]*\s*){1,4})",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?P<phrase>(?:[a-z][a-z-]*\s*){1,4})(?:paint|body|finish)\b", re.IGNORECASE),
)

_BODY_TYPE_TO_CURRENT = {
    "hatchback": "HATCHBACK",
    "sedan": "SEDAN",
    "suv": "SUV",
    "van": "VAN",
    "pickup truck": "PICKUP",
    "truck": "OTHER",
    "bus": "OTHER",
    "minibus": "OTHER",
    "motorcycle": "OTHER",
    "scooter": "OTHER",
    "bicycle": "OTHER",
    "auto rickshaw": "OTHER",
}

_COLOUR_TO_CURRENT = {
    "black": "BLACK",
    "white": "WHITE",
    "gray": "GREY",
    "silver": "SILVER",
    "red": "RED",
    "blue": "BLUE",
    "green": "GREEN",
    "yellow": "YELLOW",
    "orange": "ORANGE",
    "brown": "BROWN",
    "beige": "BEIGE",
    "purple": "PURPLE",
    "gold": "OTHER",
    "pink": "OTHER",
}


@dataclass(slots=True, frozen=True)
class LegacyCaptionParseResult:
    raw_caption: str
    parsed_body_type_text: str | None
    parsed_colour_text: str | None
    normalized_body_type: str
    normalized_colour: str
    body_type_reason: str
    colour_reason: str


def _first_explicit_term(text: str, terms: tuple[str, ...]) -> str | None:
    normalized = str(text or "").lower()
    for term in sorted(terms, key=len, reverse=True):
        if re.search(rf"\b{re.escape(term)}\b", normalized):
            return term
    return None


def _normalize_color_phrase(value: str | None) -> str | None:
    normalized = re.sub(r"[^a-z -]+", " ", str(value or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return None

    matches: list[tuple[int, int, str]] = []
    for shade, canonical in _LEGACY_SHADE_TO_CANONICAL.items():
        match = re.search(rf"\b{re.escape(shade)}\b", normalized)
        if match:
            matches.append((match.start(), -len(shade), canonical))
    return min(matches)[2] if matches else None


def _extract_florence_vehicle_color(caption: str | None) -> tuple[str | None, str | None]:
    text = str(caption or "")
    for pattern in _COLOR_PHRASE_PATTERNS:
        for match in pattern.finditer(text):
            raw_phrase = re.sub(r"\s+", " ", match.group("phrase")).strip(" ,.-")
            canonical = _normalize_color_phrase(raw_phrase)
            if canonical:
                return raw_phrase.lower(), canonical
    return None, None


def parse_old_td_case2_caption(raw_caption: str) -> LegacyCaptionParseResult:
    caption = str(raw_caption or "").strip()
    body_type_text = _first_explicit_term(caption, _LEGACY_BODY_TYPES)
    colour_text, canonical_colour = _extract_florence_vehicle_color(caption)
    normalized_body_type = _BODY_TYPE_TO_CURRENT.get(str(body_type_text or "").lower(), "UNKNOWN")
    normalized_colour = _COLOUR_TO_CURRENT.get(str(canonical_colour or "").lower(), "UNKNOWN")
    return LegacyCaptionParseResult(
        raw_caption=caption,
        parsed_body_type_text=body_type_text,
        parsed_colour_text=canonical_colour,
        normalized_body_type=normalized_body_type,
        normalized_colour=normalized_colour,
        body_type_reason="caption_keyword_match" if body_type_text else "no_explicit_body_type",
        colour_reason="caption_colour_phrase" if canonical_colour else "no_vehicle_colour_phrase",
    )
