from __future__ import annotations

from src.indian_plate_validator import validate_indian_plate


def test_validator_accepts_valid_standard_plates() -> None:
    cases = {
        "MP09AB1234": "standard_private",
        "MH12DE1433": "standard_private",
        "KA01AB1234": "standard_private",
        "DL3CCX1351": "standard_private",
    }

    for value, expected_format in cases.items():
        result = validate_indian_plate(value)
        assert result.valid is True
        assert result.canonical_text == value
        assert result.format_type == expected_format
        assert result.correction_applied is False


def test_validator_accepts_bharat_series() -> None:
    for value in ("22BH1234AA", "23BH5678AB", "24BH0001ZZ"):
        result = validate_indian_plate(value)
        assert result.valid is True
        assert result.canonical_text == value
        assert result.format_type == "bharat_series"


def test_validator_normalizes_input_before_validation() -> None:
    for value in ("dl6cq1126", "DL 6C Q 1126", "DL-6C-Q-1126"):
        result = validate_indian_plate(value)
        assert result.valid is True
        assert result.normalized_text == "DL6CQ1126"
        assert result.canonical_text == "DL6CQ1126"


def test_validator_rejects_invalid_inputs_including_ligaj7519() -> None:
    cases = ("LIGAJ7519", "ABCDEFG", "12345678", "ZZ99999999", "DL@@@@1234")
    for value in cases:
        result = validate_indian_plate(value)
        assert result.valid is False
        assert result.canonical_text is None
        assert result.reason in {"empty_normalized_text", "unsupported_indian_registration_structure"}

    ligaj = validate_indian_plate("LIGAJ7519")
    assert ligaj.normalized_text == "LIGAJ7519"
    assert "LIGAJ7519" in ligaj.attempted_candidates


def test_validator_applies_position_aware_ocr_corrections_only_when_safe() -> None:
    cases = {
        "DL6CQI126": "DL6CQ1126",
        "MH12AB81234": "MH12ABB1234",
        "TS09ASG234": "TS09AS6234",
        "22BH1234A8": "22BH1234AB",
        "22BHIZ34AA": "22BH1234AA",
    }

    for raw_text, canonical in cases.items():
        result = validate_indian_plate(raw_text)
        assert result.valid is True
        assert result.canonical_text == canonical
        assert result.correction_applied is True
        assert result.correction_count <= 2


def test_validator_does_not_force_random_garbage_into_plate() -> None:
    result = validate_indian_plate("8HXXOOPS1")
    assert result.valid is False
    assert result.canonical_text is None
