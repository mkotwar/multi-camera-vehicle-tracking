from __future__ import annotations

from .schemas import VEHICLE_COLOUR_UNKNOWN


SUPPORTED_VEHICLE_CLASSES: tuple[str, ...] = (
    "3WHEELER",
    "BUS",
    "CAR",
    "MOTORCYCLE",
    "TRUCK",
)

SUPPORTED_VEHICLE_COLOUR_LABELS: tuple[str, ...] = (
    "BLACK",
    "WHITE",
    "GREY",
    "SILVER",
    "RED",
    "PINK",
    "BLUE",
    "GREEN",
    "YELLOW",
    "ORANGE",
    "BROWN",
    "BEIGE",
    "PURPLE",
    "OTHER",
    VEHICLE_COLOUR_UNKNOWN,
)
