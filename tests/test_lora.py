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
