from __future__ import annotations

import re
from typing import Any


_NON_ALNUM_PLATE = re.compile(r"[^A-Z0-9]+")


def normalize_plate_text(value: Any) -> str | None:
    normalized = _NON_ALNUM_PLATE.sub("", str(value or "").upper())
    return normalized or None


def display_plate_text(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    return text or None
