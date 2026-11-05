import os
from pathlib import Path
from typing import Any

import yaml

from soc_triage.utils.exceptions import ConfigError

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[3]))
CONFIG_DIR = PROJECT_ROOT / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(*names: str) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for name in names or ("base",):
        config = _deep_merge(config, load_yaml(CONFIG_DIR / f"{name}.yaml"))
    return config
