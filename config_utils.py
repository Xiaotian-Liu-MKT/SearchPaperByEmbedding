"""Utility helpers for loading configuration values from project files.

This module intentionally avoids reading directly from process environment
variables so that credentials are only sourced from explicit configuration
artifacts such as ``.env`` or ``config.json``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from dotenv import dotenv_values


def _load_dotenv() -> Dict[str, str]:
    env_path = Path(".env")
    if not env_path.exists():
        return {}
    return {k.upper(): v for k, v in dotenv_values(env_path).items() if v is not None}


def _load_json_config() -> Dict[str, str]:
    config_path = Path("config.json")
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc

    def _flatten(prefix: str, data: Dict[str, object], target: Dict[str, str]) -> None:
        for key, value in data.items():
            composite_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                _flatten(composite_key, value, target)
            elif value is not None:
                target[composite_key.upper()] = str(value)

    flattened: Dict[str, str] = {}
    if isinstance(payload, dict):
        _flatten("", payload, flattened)
    return flattened


@lru_cache(maxsize=None)
def _combined_settings() -> Dict[str, str]:
    # Later sources override earlier ones so that config.json wins over .env
    settings: Dict[str, str] = {}
    settings.update(_load_dotenv())
    settings.update(_load_json_config())
    return settings


def get_setting(key: str) -> Optional[str]:
    """Return a configuration value.

    Parameters
    ----------
    key:
        The lookup key. The search is case-insensitive.

    Returns
    -------
    Optional[str]
        The configured value, or ``None`` if it is not defined.
    """

    normalized_key = key.upper()
    return _combined_settings().get(normalized_key)

