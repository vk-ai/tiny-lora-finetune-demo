"""DoRA + optional QLoRA teaching stub tests."""

from __future__ import annotations

import numpy as np
import pytest

from tiny_lora.config import load_config
from tiny_lora.dora import DoRALinear
from tiny_lora.eval import compare_adapters, run_before_after
from tiny_lora.model import TinyClassifier
from tiny_lora.qlora import apply_fake4bit_to_base, fake_quant_nf4_like, qlora_available
from tiny_lora.train import train_lora
from tiny_lora.data import train_test_split


def test_dora_starts_near_base_when_B_zero():
    rng = np.random.default_rng(0)
    layer = DoRALinear.create(8, 4, rank=2, alpha=4.0, rng=rng)
    x = rng.normal(size=(5, 8))
    # m init = ||W||_row and B=0 → effective ≈ W
    y = layer.forward(x)
    base = x @ layer.W.T + layer.b
    np.testing.assert_allclose(y, base, atol=1e-6)


def test_dora_extra_magnitude_params():
    rng = np.random.default_rng(1)
    lora_like = DoRALinear.create(16, 8, rank=4, alpha=8.0, rng=rng)
    # A+B+m
    assert lora_like.n_trainable() == 4 * 16 + 8 * 4 + 8


def test_dora_merge_matches_forward():
    rng = np.random.default_rng(2)
    layer = DoRALinear.create(6, 3, rank=2, alpha=4.0, rng=rng)
    layer.B = rng.normal(size=layer.B.shape)
    layer.m = np.abs(rng.normal(size=layer.m.shape)) + 0.1
    x = rng.normal(size=(4, 6))
    y = layer.forward(x)
    merged = layer.merge_and_unload()
    np.testing.assert_allclose(merged.forward(x), y, atol=1e-10)


def test_classifier_dora_trains_and_improves():
    cfg = load_config()
    cfg["lora"]["use_dora"] = True
    cfg["lora"]["rank"] = 2
    cfg["train"]["epochs"] = 40
    cfg["data"]["n_train"] = 128
    cfg["data"]["n_test"] = 32
    report = run_before_after(cfg)
    assert report.adapter_kind == "dora"
    assert report.after.accuracy >= report.before.accuracy - 1e-9
    assert report.trainable_params > 0


def test_fake4bit_reduces_estimated_nbytes():
    W = np.random.default_rng(3).normal(size=(32, 16))
    Wq, rep = apply_fake4bit_to_base(W)
    assert Wq.shape == W.shape
    assert rep.base_nbytes_q4_est < rep.base_nbytes_fp64
    assert fake_quant_nf4_like(W).shape == W.shape


def test_qlora_available_is_bool():
    assert isinstance(qlora_available(), bool)


def test_compare_adapters_table():
    cfg = load_config()
    cfg["train"]["epochs"] = 15
    cfg["data"]["n_train"] = 96
    cfg["data"]["n_test"] = 32
    rows = compare_adapters(cfg)
    names = {r["variant"] for r in rows}
    assert names == {"lora", "dora", "qlora_fake4bit"}
    dora = next(r for r in rows if r["variant"] == "dora")
    lora = next(r for r in rows if r["variant"] == "lora")
    # DoRA has +out magnitude params
    assert dora["trainable_params"] > lora["trainable_params"]
