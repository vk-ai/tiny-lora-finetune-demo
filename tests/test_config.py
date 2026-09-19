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


def test_scaling_default_classic():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert cfg["lora"]["scaling"] == "classic"


def test_rslora_scaling_accepted(tmp_path: Path):
    p = tmp_path / "rslora.yaml"
    p.write_text(
        "seed: 1\n"
        "data: {n_train: 8, n_test: 4, n_features: 4, n_classes: 2, noise: 0.1}\n"
        "model: {hidden_dim: 4}\n"
        "lora: {rank: 2, alpha: 4.0, scaling: rslora}\n"
        "train: {epochs: 1, lr: 0.1, batch_size: 4}\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg["lora"]["scaling"] == "rslora"


def test_invalid_scaling(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "seed: 1\ndata: {}\nmodel: {}\nlora: {rank: 2, alpha: 1, scaling: nope}\ntrain: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="scaling"):
        load_config(p)
