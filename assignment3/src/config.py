"""Load shared defaults and a model override without depending on the shell cwd."""

from copy import deepcopy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict:
    """Merge config/base.yaml with the given project-relative or absolute file.

    Data/output path strings intentionally stay project-relative for portability.
    Future runners should resolve them against PROJECT_ROOT, not the shell cwd.
    """
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    base_path = PROJECT_ROOT / "config/base.yaml"
    with base_path.open(encoding="utf-8") as file:
        base = yaml.safe_load(file)
    with path.open(encoding="utf-8") as file:
        override = yaml.safe_load(file)
    if not isinstance(base, dict) or not isinstance(override, dict):
        raise ValueError("Configuration files must contain YAML mappings.")
    return _merge(base, override)
