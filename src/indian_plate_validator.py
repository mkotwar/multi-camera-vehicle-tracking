from __future__ import annotations

from dataclasses import dataclass, field
import re

from src.plate_text import normalize_plate_text


# Adapted from the legacy OCR_MUKUL project and updated for the canonical pipeline.
VALID_STATE_CODES = frozenset(
    {
        "AN",
        "AP",
        "AR",
        "AS",
        "BR",
        "CG",
        "CH",
        "DD",
        "DL",
        "DN",
        "GA",
        "GJ",
        "HP",
        "HR",
        "JH",
        "JK",
        "KA",
        "KL",
        "LA",
        "LD",
        "MH",
        "ML",
        "MN",
        "MP",
        "MZ",
        "NL",
        "OD",
        "PB",
        "PY",
        "RJ",
        "SK",
        "TN",
        "TR",
        "TS",
        "UK",
        "UP",
        "WB",
    }
)

STANDARD_PLATE_RE = re.compile(
    r"^(?P<state>[A-Z]{2})(?P<rto>\d{1,2})(?P<series>[A-Z]{1,3})(?P<number>\d{4})$"
)
BH_PLATE_RE = re.compile(r"^(?P<year>\d{2})BH(?P<number>\d{4})(?P<series>[A-Z]{1,2})$")
STANDARD_EXPECTED_PATTERNS = tuple(
    f"LL{'D' * rto_digits}{'L' * series_letters}DDDD"
    for rto_digits in (1, 2)
    for series_letters in (1, 2, 3)
)
BH_EXPECTED_PATTERNS = ("DDBHDDDDLL", "DDBHDDDDL")

LETTER_TO_DIGIT = {
    "B": "8",
    "G": "6",
    "I": "1",
    "L": "1",
    "O": "0",
    "S": "5",
    "Z": "2",
}
DIGIT_TO_LETTER = {
    "0": ("O",),
    "1": ("I", "L"),
    "2": ("Z",),
    "5": ("S",),
    "6": ("G",),
    "8": ("B",),
}


@dataclass(slots=True)
class IndianPlateValidationResult:
    raw_text: str | None
    normalized_text: str | None
    canonical_text: str | None
    valid: bool
    format_type: str | None
    correction_applied: bool
    reason: str
    corrected_from: str | None = None
    correction_count: int = 0
    attempted_candidates: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _ValidatedCandidate:
    canonical_text: str
    format_type: str
    reason: str


def validate_indian_plate(raw_text: str | None) -> IndianPlateValidationResult:
    normalized_text = normalize_plate_text(raw_text)
    if not normalized_text:
        return IndianPlateValidationResult(
            raw_text=raw_text,
            normalized_text=normalized_text,
            canonical_text=None,
            valid=False,
            format_type=None,
            correction_applied=False,
            reason="empty_normalized_text",
        )

    direct_match = _validate_candidate(normalized_text)
    if direct_match is not None:
        return IndianPlateValidationResult(
            raw_text=raw_text,
            normalized_text=normalized_text,
            canonical_text=direct_match.canonical_text,
            valid=True,
            format_type=direct_match.format_type,
            correction_applied=False,
            reason=direct_match.reason,
            attempted_candidates=[normalized_text],
        )

    corrected = _attempt_position_aware_correction(normalized_text)
    if corrected is not None:
        return IndianPlateValidationResult(
            raw_text=raw_text,
            normalized_text=normalized_text,
            canonical_text=corrected.candidate.canonical_text,
            valid=True,
            format_type=corrected.candidate.format_type,
            correction_applied=True,
            reason=corrected.candidate.reason,
            corrected_from=normalized_text,
            correction_count=corrected.correction_count,
            attempted_candidates=corrected.attempted_candidates,
        )

    return IndianPlateValidationResult(
        raw_text=raw_text,
        normalized_text=normalized_text,
        canonical_text=None,
        valid=False,
        format_type=None,
        correction_applied=False,
        reason="unsupported_indian_registration_structure",
        attempted_candidates=_correction_probe_candidates(normalized_text),
    )


def is_valid_indian_plate(raw_text: str | None) -> bool:
    return validate_indian_plate(raw_text).valid


def _validate_candidate(candidate: str) -> _ValidatedCandidate | None:
    standard_match = STANDARD_PLATE_RE.fullmatch(candidate)
    if standard_match is not None:
        state = standard_match.group("state")
        if state not in VALID_STATE_CODES:
            return None
        return _ValidatedCandidate(
            canonical_text=candidate,
            format_type="standard_private",
            reason="validated_standard_state_registration",
        )

    bh_match = BH_PLATE_RE.fullmatch(candidate)
    if bh_match is not None:
        return _ValidatedCandidate(
            canonical_text=candidate,
            format_type="bharat_series",
            reason="validated_bharat_series_registration",
        )

    return None


@dataclass(frozen=True, slots=True)
class _CorrectionMatch:
    candidate: _ValidatedCandidate
    correction_count: int
    attempted_candidates: list[str]


def _attempt_position_aware_correction(normalized_text: str) -> _CorrectionMatch | None:
    matches: dict[str, _CorrectionMatch] = {}
    attempted_candidates: list[str] = [normalized_text]

    for expected in (*STANDARD_EXPECTED_PATTERNS, *BH_EXPECTED_PATTERNS):
        if len(normalized_text) != len(expected):
            continue
        for candidate_text, correction_count in _generate_candidates(normalized_text, expected):
            if candidate_text not in attempted_candidates:
                attempted_candidates.append(candidate_text)
            validated = _validate_candidate(candidate_text)
            if validated is None:
                continue
            existing = matches.get(validated.canonical_text)
            match = _CorrectionMatch(
                candidate=validated,
                correction_count=correction_count,
                attempted_candidates=attempted_candidates.copy(),
            )
            if existing is None or correction_count < existing.correction_count:
                matches[validated.canonical_text] = match

    if not matches:
        return None
    if len(matches) > 1:
        return None
    return next(iter(matches.values()))


def _generate_candidates(normalized_text: str, expected: str) -> list[tuple[str, int]]:
    indexes: list[tuple[int, tuple[str, ...]]] = []
    for idx, (char, expected_kind) in enumerate(zip(normalized_text, expected, strict=True)):
        if expected_kind == "L" and char.isdigit():
            replacements = DIGIT_TO_LETTER.get(char, ())
            if replacements:
                indexes.append((idx, replacements))
        elif expected_kind == "D" and char.isalpha():
            replacement = LETTER_TO_DIGIT.get(char)
            if replacement:
                indexes.append((idx, (replacement,)))
        elif expected_kind not in {"L", "D"} and char != expected_kind:
            return []

    if not indexes:
        return []
    if len(indexes) > 2:
        return []

    generated: dict[str, int] = {}
    options_by_index: list[tuple[int, tuple[str | None, ...]]] = [
        (char_index, (None, *replacements))
        for char_index, replacements in indexes
    ]

    def _walk(option_index: int, chars: list[str], correction_count: int) -> None:
        if correction_count > 2:
            return
        if option_index >= len(options_by_index):
            if correction_count > 0:
                candidate = "".join(chars)
                generated[candidate] = min(correction_count, generated.get(candidate, correction_count))
            return
        char_index, replacements = options_by_index[option_index]
        original = chars[char_index]
        for replacement in replacements:
            chars[char_index] = original if replacement is None else replacement
            _walk(option_index + 1, chars, correction_count + (0 if replacement is None else 1))
        chars[char_index] = original

    _walk(0, list(normalized_text), 0)
    return [(candidate, count) for candidate, count in generated.items() if count <= 2]


def _correction_probe_candidates(normalized_text: str) -> list[str]:
    attempted = [normalized_text]
    for expected in (*STANDARD_EXPECTED_PATTERNS, *BH_EXPECTED_PATTERNS):
        if len(normalized_text) != len(expected):
            continue
        for candidate, _correction_count in _generate_candidates(normalized_text, expected):
            if candidate not in attempted:
                attempted.append(candidate)
    return attempted
