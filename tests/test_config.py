"""Config YAML loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from tiny_lora.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_loads():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert cfg["lora"]["rank"] == 4
    assert cfg["lora"]["alpha"] == 8.0
    assert cfg["data"]["n_classes"] == 3


def test_invalid_rank(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "seed: 1\ndata: {}\nmodel: {}\nlora: {rank: 0, alpha: 1}\ntrain: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rank"):
        load_config(p)
