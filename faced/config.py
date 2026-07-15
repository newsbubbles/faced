"""Loaders for config/emotions.yaml (the axis taxonomy + face weights)."""
from __future__ import annotations

from functools import lru_cache

import yaml

from .backends import CONFIG_DIR


@lru_cache(maxsize=1)
def emotions_config() -> dict:
    with open(CONFIG_DIR / "emotions.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def axis_names() -> list[str]:
    return [a["name"] for a in emotions_config()["axes"]]


def axis_by_name() -> dict:
    return {a["name"]: a for a in emotions_config()["axes"]}
