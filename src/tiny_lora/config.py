"""Load YAML training config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with cfg_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {cfg_path}")
    _validate(data)
    return data


def _validate(cfg: dict[str, Any]) -> None:
    for key in ("seed", "data", "model", "lora", "train"):
        if key not in cfg:
            raise ValueError(f"Missing config key: {key}")
    lora = cfg["lora"]
    if int(lora["rank"]) < 1:
        raise ValueError("lora.rank must be >= 1")
    if float(lora["alpha"]) <= 0:
        raise ValueError("lora.alpha must be > 0")
