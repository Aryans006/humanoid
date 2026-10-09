"""Config loading. Physical sizes must be measured values; nulls fail loudly."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(os.environ.get("EGO2LIBERO_CONFIG_DIR", REPO_ROOT / "configs"))


def config_dir() -> Path:
    """configs/ of the repo, or $EGO2LIBERO_CONFIG_DIR (tests point it at a temp copy)."""
    return Path(os.environ.get("EGO2LIBERO_CONFIG_DIR", CONFIG_DIR))


def load_yaml(name_or_path: str | Path) -> dict[str, Any]:
    p = Path(name_or_path)
    cfg_dir = config_dir()
    if not p.suffix:
        p = cfg_dir / f"{p}.yaml"
    elif not p.is_absolute() and not p.exists():
        p = cfg_dir / p
    with open(p) as f:
        return yaml.safe_load(f) or {}


def require(cfg: dict, *keys: str, where: str = "config") -> None:
    """Raise if any key is missing or null (i.e. not yet measured / decided)."""
    missing = [k for k in keys if cfg.get(k) is None]
    if missing:
        raise ValueError(f"{where}: fill in {missing} (measured values, not nominal)")
