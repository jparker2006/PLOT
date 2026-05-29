"""Load the YAML project config so it is a live source of truth, not inert documentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = "configs/default.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read the YAML config; return {} if absent."""
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def eval_bar_lightgbm(cfg: dict) -> tuple[dict, int, int]:
    """Split the eval_bar.lightgbm block into (lgbm param overrides, num_boost_round, early_stopping_rounds)."""
    lgb = dict((cfg.get("eval_bar") or {}).get("lightgbm") or {})
    num_boost_round = int(lgb.pop("num_boost_round", 400))
    early_stopping_rounds = int(lgb.pop("early_stopping_rounds", 50))
    return lgb, num_boost_round, early_stopping_rounds
