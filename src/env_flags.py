from __future__ import annotations

import os

from .env_loader import load_project_env_defaults


TRUE_VALUES = {"1", "true", "yes", "on"}


def env_flag(name: str, *, default: bool = False) -> bool:
    load_project_env_defaults()
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in TRUE_VALUES


def db_import_after_run_enabled() -> bool:
    return env_flag("DB_IMPORT_AFTER_RUN", default=False)
