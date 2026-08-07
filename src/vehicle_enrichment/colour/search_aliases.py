from __future__ import annotations

COLOUR_SEARCH_ALIASES: dict[str, tuple[str, ...]] = {
    "RED": ("RED", "PINK"),
    "PINK": ("PINK",),
}


def expand_colour_search_labels(label: str | None) -> tuple[str, ...]:
    normalized = str(label or "").strip().upper()
    if not normalized:
        return ()
    return COLOUR_SEARCH_ALIASES.get(normalized, (normalized,))
