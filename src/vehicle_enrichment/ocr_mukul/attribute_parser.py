from __future__ import annotations

from dataclasses import dataclass
import re


OCR_MUKUL_UNKNOWN = "UNKNOWN"
ATTRIBUTE_REASON_PLATE_LIKE = "plate_like_response_in_attribute_path"

_COLOUR_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("BLACK", ("black", "jet black", "matte black")),
    ("WHITE", ("white", "pearl white", "off white", "off-white", "ivory")),
    ("GREY", ("gray", "grey", "dark gray", "dark grey", "light gray", "light grey", "charcoal", "graphite")),
    ("SILVER", ("silver", "metallic silver")),
    ("BLUE", ("blue", "navy blue", "cobalt", "azure")),
    ("RED", ("red", "maroon", "burgundy", "crimson", "scarlet")),
    ("GREEN", ("green", "olive", "forest green", "emerald")),
    ("YELLOW", ("yellow", "mustard")),
    ("ORANGE", ("orange", "amber", "rust")),
    ("BROWN", ("brown", "bronze", "copper", "chocolate")),
    ("BEIGE", ("beige", "tan", "sand", "khaki", "cream")),
    ("PURPLE", ("purple", "violet", "plum", "lavender")),
]

_BODY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("SUV", ("sport utility vehicle", "sports utility vehicle", "suv")),
    ("SEDAN", ("sedan", "saloon")),
    ("HATCHBACK", ("hatchback", "hatch back")),
    ("MPV", ("multi-purpose vehicle", "multi purpose vehicle", "multi utility vehicle", "multipurpose vehicle", "muv", "mpv")),
    ("PICKUP", ("pickup truck", "pickup", "pick-up", "pick up")),
    ("VAN", ("van", "minivan")),
    ("COUPE", ("coupe",)),
    ("CONVERTIBLE", ("convertible", "cabriolet")),
]

_VEHICLE_NOUNS = r"(?:car|vehicle|sedan|hatchback|suv|truck|pickup|van|minivan|wagon|coupe|convertible|bus|motorcycle|3wheeler)"
_UNCERTAIN_PHRASES = {
    "",
    "vehicle",
    "car",
    "unknown",
    "unclear",
    "cannot determine",
    "not visible",
}
_PLATE_LIKE_PATTERNS = [
    re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$"),
    re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$"),
]


@dataclass(slots=True, frozen=True)
class ParsedCaptionAttributes:
    caption: str
    raw_body_type_phrase: str | None
    normalized_body_type: str
    body_type_reason: str
    raw_colour_phrase: str | None
    normalized_colour: str
    colour_reason: str


def _normalize_text(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 -]+", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _match_phrase(text: str, rules: list[tuple[str, tuple[str, ...]]]) -> tuple[str | None, str | None]:
    normalized = _normalize_text(text)
    for canonical, phrases in rules:
        for phrase in sorted(phrases, key=len, reverse=True):
            if re.search(rf"\b{re.escape(phrase)}\b", normalized):
                return phrase, canonical
    return None, None


def looks_like_plate_text(text: str) -> bool:
    normalized = re.sub(r"[^A-Za-z0-9]", "", str(text or "").upper())
    if not normalized:
        return False
    return any(pattern.fullmatch(normalized) for pattern in _PLATE_LIKE_PATTERNS)


def _extract_explicit_field(text: str, field_name: str) -> str | None:
    match = re.search(rf"\b{re.escape(field_name)}\s*:\s*([A-Za-z -]+)", str(text or ""), flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def _extract_colour_phrase(text: str) -> tuple[str | None, str]:
    explicit = _extract_explicit_field(text, "COLOUR") or _extract_explicit_field(text, "COLOR")
    if explicit:
        raw_phrase, canonical = _match_phrase(explicit, _COLOUR_RULES)
        if canonical is not None:
            return raw_phrase or explicit, canonical
    normalized = _normalize_text(text)
    for canonical, phrases in _COLOUR_RULES:
        for phrase in sorted(phrases, key=len, reverse=True):
            if re.search(rf"\b{re.escape(phrase)}\b", normalized):
                return phrase, canonical
    colour_before_vehicle = re.search(rf"\b((?:[a-z][a-z-]*\s+){{0,3}}[a-z][a-z-]*)\s+{_VEHICLE_NOUNS}\b", normalized)
    if colour_before_vehicle:
        phrase = colour_before_vehicle.group(1).strip()
        raw_phrase, canonical = _match_phrase(phrase, _COLOUR_RULES)
        if canonical is not None:
            return raw_phrase, canonical
    return None, OCR_MUKUL_UNKNOWN


def _extract_body_phrase(text: str) -> tuple[str | None, str]:
    explicit = _extract_explicit_field(text, "BODY_TYPE") or _extract_explicit_field(text, "BODY TYPE")
    if explicit:
        raw_phrase, canonical = _match_phrase(explicit, _BODY_RULES)
        if canonical is not None:
            return raw_phrase or explicit, canonical
    return _match_phrase(text, _BODY_RULES) if _normalize_text(text) not in _UNCERTAIN_PHRASES else (None, OCR_MUKUL_UNKNOWN)


def parse_caption_attributes(caption: str) -> ParsedCaptionAttributes:
    normalized_caption = _normalize_text(caption)
    if looks_like_plate_text(caption):
        return ParsedCaptionAttributes(
            caption=str(caption or "").strip(),
            raw_body_type_phrase=None,
            normalized_body_type=OCR_MUKUL_UNKNOWN,
            body_type_reason=ATTRIBUTE_REASON_PLATE_LIKE,
            raw_colour_phrase=None,
            normalized_colour=OCR_MUKUL_UNKNOWN,
            colour_reason=ATTRIBUTE_REASON_PLATE_LIKE,
        )
    if normalized_caption in _UNCERTAIN_PHRASES:
        return ParsedCaptionAttributes(
            caption=str(caption or "").strip(),
            raw_body_type_phrase=None,
            normalized_body_type=OCR_MUKUL_UNKNOWN,
            body_type_reason="generic_caption",
            raw_colour_phrase=None,
            normalized_colour=OCR_MUKUL_UNKNOWN,
            colour_reason="generic_caption",
        )
    body_phrase, body_label = _extract_body_phrase(caption)
    colour_phrase, colour_label = _extract_colour_phrase(caption)
    return ParsedCaptionAttributes(
        caption=str(caption or "").strip(),
        raw_body_type_phrase=body_phrase,
        normalized_body_type=body_label or OCR_MUKUL_UNKNOWN,
        body_type_reason="caption_keyword_match" if body_label else "no_explicit_body_type",
        raw_colour_phrase=colour_phrase,
        normalized_colour=colour_label or OCR_MUKUL_UNKNOWN,
        colour_reason="caption_colour_phrase" if colour_label != OCR_MUKUL_UNKNOWN else "no_vehicle_colour_phrase",
    )
