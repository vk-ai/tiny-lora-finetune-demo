"""Unit tests for toy LoRA math."""

from __future__ import annotations

import numpy as np
import pytest

from tiny_lora.lora import LoRALinear


def test_lora_starts_as_noop():
    rng = np.random.default_rng(0)
    layer = LoRALinear.create(8, 4, rank=2, alpha=4.0, rng=rng)
    x = rng.normal(size=(5, 8))
    # B=0 => adapter term is zero
    base = x @ layer.W.T + layer.b
    np.testing.assert_allclose(layer.forward(x), base, atol=1e-12)


def test_delta_w_rank():
    rng = np.random.default_rng(1)
    layer = LoRALinear.create(10, 6, rank=3, alpha=6.0, rng=rng)
    layer.B = rng.normal(size=layer.B.shape)
    dW = layer.delta_W()
    assert dW.shape == (6, 10)
    # Rank of B@A is at most rank
    s = np.linalg.svd(dW, compute_uv=False)
    assert np.sum(s > 1e-8) <= 3


def test_scaling_alpha_over_rank():
    rng = np.random.default_rng(2)
    layer = LoRALinear.create(4, 2, rank=4, alpha=8.0, rng=rng)
    assert layer.scaling == pytest.approx(2.0)


def test_trainable_count():
    rng = np.random.default_rng(3)
    layer = LoRALinear.create(16, 8, rank=4, alpha=8.0, rng=rng)
    # A: 4*16, B: 8*4
    assert layer.n_trainable() == 4 * 16 + 8 * 4
    assert layer.n_frozen() == 8 * 16 + 8


def test_rslora_scaling_alpha_over_sqrt_rank():
    from tiny_lora.lora import lora_scale

    rng = np.random.default_rng(2)
    layer = LoRALinear.create(4, 2, rank=4, alpha=8.0, rng=rng, scale_mode="rslora")
    assert layer.scaling == pytest.approx(8.0 / 2.0)  # sqrt(4)=2
    assert lora_scale(8.0, 4, "classic") == pytest.approx(2.0)
    assert lora_scale(8.0, 4, "rslora") == pytest.approx(4.0)


def test_classic_vs_rslora_scale_differ_for_rank():
    from tiny_lora.lora import lora_scale

    # For rank=16, classic=alpha/16, rslora=alpha/4 — clearly different
    assert lora_scale(8.0, 16, "classic") != pytest.approx(lora_scale(8.0, 16, "rslora"))


def test_merge_and_unload_matches_adapter_forward():
    from tiny_lora.lora import MergedLinear

    rng = np.random.default_rng(7)
    layer = LoRALinear.create(8, 4, rank=2, alpha=4.0, rng=rng, scale_mode="classic")
    layer.B = rng.normal(size=layer.B.shape)
    x = rng.normal(size=(6, 8))
    y_adapter = layer.forward(x)
    merged = layer.merge_and_unload()
    assert isinstance(merged, MergedLinear)
    np.testing.assert_allclose(merged.forward(x), y_adapter, atol=1e-10)
    # Original layer still has A/B (not in-place) — assign-return lesson
    assert layer.A is not None and layer.B is not None
    np.testing.assert_allclose(layer.forward(x), y_adapter, atol=1e-10)


def test_merge_and_unload_rslora_scale():
    rng = np.random.default_rng(8)
    layer = LoRALinear.create(6, 3, rank=4, alpha=8.0, rng=rng, scale_mode="rslora")
    layer.B = rng.normal(size=layer.B.shape)
    x = rng.normal(size=(4, 6))
    y_adapter = layer.forward(x)
    merged = layer.merge_and_unload()
    np.testing.assert_allclose(merged.forward(x), y_adapter, atol=1e-10)
    assert merged.merged_from == "rslora"
    assert merged.n_trainable() == 0


def test_classifier_merge_and_unload_assign_return():
    from tiny_lora.model import TinyClassifier

    rng = np.random.default_rng(9)
    model = TinyClassifier.create(8, 8, 3, rank=2, alpha=4.0, rng=rng)
    model.head.B = rng.normal(size=model.head.B.shape)
    x = rng.normal(size=(5, 8))
    adapter_logits = model.logits(x)
    assert model.mode == "adapter"
    # Wrong pattern would discard the return; we assign.
    merged = model.merge_and_unload()
    assert merged.mode == "merged"
    assert model.mode == "adapter"  # original unchanged
    np.testing.assert_allclose(merged.logits(x), adapter_logits, atol=1e-10)
    assert merged.n_trainable() == 0
