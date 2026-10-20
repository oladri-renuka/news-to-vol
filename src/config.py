"""YAML config loading with single-parent inheritance.

Every experiment config may declare ``inherit: <relative path>``; the parent is
loaded first and the child is deep-merged on top. Command-line overrides use
dotted keys, e.g. ``--set train.max_epochs=2``.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = deep_merge(load_config(path.parent / parent), cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    cfg.setdefault("_config_path", str(path))
    return cfg


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def resolve(p: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
