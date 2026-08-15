from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_env_file(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_project_env_defaults(*, env_path: str | Path | None = None, skip_pytest: bool = True) -> dict[str, str]:
    if skip_pytest and (os.getenv("PYTEST_CURRENT_TEST") or "pytest" in sys.modules):
        return {}
    path = Path(env_path) if env_path is not None else project_root() / ".env"
    values = load_env_file(path)
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return values


def merged_project_env(*, base: dict[str, str] | None = None, env_path: str | Path | None = None) -> dict[str, str]:
    merged = dict(base or os.environ)
    path = Path(env_path) if env_path is not None else project_root() / ".env"
    values = load_env_file(path)
    values.update(merged)
    return values


def set_env_file_value(path: str | Path, key: str, value: str) -> dict[str, str]:
    env_path = Path(path)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    output: list[str] = []
    replaced = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            output.append(line)
            continue
        current_key, _current_value = stripped.split("=", 1)
        if current_key.strip() == key:
            output.append(f"{key}={value}")
            replaced = True
        else:
            output.append(line)
    if not replaced:
        output.append(f"{key}={value}")
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=env_path.parent, delete=False, suffix=".tmp") as handle:
            tmp_name = handle.name
            handle.write("\n".join(output).rstrip() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, env_path)
    finally:
        if tmp_name and Path(tmp_name).exists():
            Path(tmp_name).unlink()
    os.environ[key] = value
    return load_env_file(env_path)
