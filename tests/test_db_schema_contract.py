from __future__ import annotations

from src.db_schema_contract import CANONICAL_SCHEMA
from src.importers.db_writer import MIRROR_TABLES


def test_canonical_schema_covers_all_import_tables() -> None:
    assert set(MIRROR_TABLES) <= set(CANONICAL_SCHEMA)


def test_physical_identity_tables_are_in_canonical_schema() -> None:
    assert {"physical_vehicles", "physical_vehicle_tracks", "identity_decisions"} <= set(CANONICAL_SCHEMA)
    assert "vehicle_key" in CANONICAL_SCHEMA["physical_vehicles"]
    assert "vehicle_track_id" in CANONICAL_SCHEMA["physical_vehicle_tracks"]
    assert "final_score" in CANONICAL_SCHEMA["identity_decisions"]
