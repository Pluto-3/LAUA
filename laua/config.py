"""Configuration loader — merges default.yaml with optional user overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(user_config_path: Path | None = None) -> dict[str, Any]:
    default_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with default_path.open() as f:
        config = yaml.safe_load(f)

    env_cfg = os.environ.get("LAUA_CONFIG")
    search_paths = [
        user_config_path,
        Path(env_cfg) if env_cfg else None,
        Path.home() / ".laua" / "config.yaml",
        Path("laua.yaml"),
    ]
    for path in search_paths:
        if path and path.is_file():
            with path.open() as f:
                user_cfg = yaml.safe_load(f) or {}
            config = _deep_merge(config, user_cfg)
            break

    # Environment variable overrides
    if url := os.environ.get("OLLAMA_BASE_URL"):
        config["ollama"]["base_url"] = url
    if model := os.environ.get("OLLAMA_MODEL"):
        config["ollama"]["default_model"] = model

    return config
