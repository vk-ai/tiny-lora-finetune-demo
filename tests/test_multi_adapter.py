"""Round 4: multiple named adapters + add_weighted_adapter (linear / cat / TIES-lite)."""

from __future__ import annotations

import numpy as np
import pytest

from tiny_lora.config import load_config
from tiny_lora.dora import DoRALinear
from tiny_lora.lora import LoRALinear
from tiny_lora.merge_eval import run_multi_adapter_eval
from tiny_lora.multi_adapter import MultiAdapterHead, ties_merge

VARIANTS = [
    pytest.param("classic", False, id="classic-lora"),
    pytest.param("rslora", False, id="rslora"),
    pytest.param("classic", True, id="dora"),
    pytest.param("rslora", True, id="dora-rslora"),
]
COMBOS = ["linear", "cat", "ties"]


def _head(seed: int = 0, out: int = 5, inp: int = 12) -> tuple[MultiAdapterHead, np.random.Generator]:
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 0.3, size=(out, inp))
    b = rng.normal(0.0, 0.1, size=out)
    return MultiAdapterHead.from_base(W, b), rng


def _randomize(adapter, rng: np.random.Generator) -> None:
    """Pretend-train: nonzero B (and perturbed DoRA magnitude)."""
    adapter.B = rng.normal(0.0, 0.5, size=adapter.B.shape)
    if isinstance(adapter, DoRALinear):
        adapter.m = adapter.m * (1.0 + rng.uniform(-0.3, 0.3, size=adapter.m.shape))


def _add(head, rng, name, mode, dora, rank=3, alpha=6.0):
    ad = head.add_adapter(name, rank=rank, alpha=alpha, rng=rng, scale_mode=mode, use_dora=dora)
    _randomize(ad, rng)
    return ad


@pytest.mark.parametrize("mode,dora", VARIANTS)
@pytest.mark.parametrize("combo", COMBOS)
def test_merge_single_adapter_at_weight_one_reproduces_it(mode, dora, combo):
    """peft#3761 invariant: combine([a], [1.0]) must give back ``a``."""
    head, rng = _head(1)
    _add(head, rng, "a", mode, dora)
    x = rng.normal(size=(7, head.in_features))
    ref = head.forward(x, "a")
    merged = head.add_weighted_adapter(["a"], [1.0], "m", combination_type=combo, density=1.0)
    assert merged.scaling == pytest.approx(1.0)  # factors carry the scale
    assert type(merged) is type(head.adapters["a"])
    np.testing.assert_allclose(head.forward(x, "m"), ref, atol=1e-10, rtol=0)
    # merged adapter also folds cleanly (merge_and_unload parity)
    np.testing.assert_allclose(head.merge_and_unload("m").forward(x), ref, atol=1e-10)


def test_naive_rescaling_breaks_rslora_invariant():
    """Why scaling must be 1: re-deriving α/√r for the new rank (the upstream bug)."""
    head, rng = _head(2)
    _add(head, rng, "a", "rslora", False, rank=4, alpha=8.0)
    x = rng.normal(size=(5, head.in_features))
    good = head.add_weighted_adapter(["a"], [1.0], "cat", combination_type="cat")
    naive = LoRALinear(
        in_features=good.in_features, out_features=good.out_features, rank=good.rank,
        alpha=float(good.rank), W=head.W, b=head.b, A=good.A, B=good.B, scale_mode="rslora",
    )  # alpha=r under rsLoRA → scaling √r = 2, not 1
    assert naive.scaling == pytest.approx(2.0)
    np.testing.assert_allclose(good.forward(x), head.forward(x, "a"), atol=1e-12)
    assert np.max(np.abs(naive.forward(x) - head.forward(x, "a"))) > 1e-2


@pytest.mark.parametrize("mode_b", ["classic", "rslora"])
def test_cat_and_linear_equal_weighted_sum_of_deltas(mode_b):
    head, rng = _head(3)
    _add(head, rng, "a", "classic", False, rank=2, alpha=4.0)
    _add(head, rng, "b", mode_b, False, rank=3, alpha=3.0)  # mixed ranks / scalings
    w = [0.7, -1.3]
    expected = w[0] * head.delta_W("a") + w[1] * head.delta_W("b")
    cat = head.add_weighted_adapter(["a", "b"], w, "cat", combination_type="cat")
    lin = head.add_weighted_adapter(["a", "b"], w, "lin", combination_type="linear")
    assert cat.rank == 5
    assert lin.rank == min(5, head.out_features, head.in_features)
    np.testing.assert_allclose(cat.delta_W(), expected, atol=1e-12)
    np.testing.assert_allclose(lin.delta_W(), expected, atol=1e-10)
    assert head.merge_info["lin"].svd_rel_error < 1e-10


def test_negative_weight_cancels_to_base():
    head, rng = _head(4)
    _add(head, rng, "a", "rslora", False)
    head.add_weighted_adapter(["a", "a"], [1.0, -1.0], "zero", combination_type="linear")
    x = rng.normal(size=(4, head.in_features))
    np.testing.assert_allclose(head.forward(x, "zero"), head.forward(x, None), atol=1e-10)


def test_svd_rank_truncation_is_reported():
    head, rng = _head(5, out=6, inp=10)
    _add(head, rng, "a", "classic", False, rank=3)
    _add(head, rng, "b", "classic", False, rank=3)
    full = head.add_weighted_adapter(["a", "b"], [1, 1], "full")
    low = head.add_weighted_adapter(["a", "b"], [1, 1], "low", svd_rank=2)
    assert full.rank == 6 and low.rank == 2
    assert head.merge_info["full"].svd_rel_error < 1e-10
    assert head.merge_info["low"].svd_rel_error > 1e-3


def test_ties_merge_trim_elect_disjoint_mean():
    t1 = np.array([[3.0, -1.0, 0.2, 2.0]])
    t2 = np.array([[1.0, 2.0, -0.1, -4.0]])
    # density 1: elected signs [+, +, +, -] → disjoint means [2, 2, 0.2, -4]
    np.testing.assert_allclose(ties_merge([t1, t2], 1.0), [[2.0, 2.0, 0.2, -4.0]])
    # density 0.5: keep top-2 |τ| per vector → t1:[3,0,0,2], t2:[0,2,0,-4]
    np.testing.assert_allclose(ties_merge([t1, t2], 0.5), [[3.0, 2.0, 0.0, -4.0]])
    with pytest.raises(ValueError):
        ties_merge([t1], 0.0)


def test_registry_shares_frozen_base_and_switches():
    head, rng = _head(6)
    a = _add(head, rng, "a", "classic", False)
    d = _add(head, rng, "d", "classic", True)
    assert a.W is head.W and d.W is head.W  # one frozen base, many adapters
    x = rng.normal(size=(3, head.in_features))
    head.set_adapter("d")
    np.testing.assert_allclose(head.forward(x), d.forward(x))
    head.set_adapter(None)
    np.testing.assert_allclose(head.forward(x), x @ head.W.T + head.b)
    with pytest.raises(KeyError):
        head.set_adapter("nope")
    head.delete_adapter("d")
    assert sorted(head.adapters) == ["a"]


def test_merge_validation_errors():
    head, rng = _head(7)
    _add(head, rng, "a", "classic", False)
    _add(head, rng, "d", "classic", True)
    with pytest.raises(ValueError, match="mix"):
        head.add_weighted_adapter(["a", "d"], [1, 1], "x")
    with pytest.raises(NotImplementedError, match="DoRA"):
        head.add_weighted_adapter(["d"], [1.0], "x", dora_policy="raise")
    with pytest.raises(ValueError, match="one weight per adapter"):
        head.add_weighted_adapter(["a"], [1.0, 2.0], "x")
    with pytest.raises(ValueError, match="combination_type"):
        head.add_weighted_adapter(["a"], [1.0], "x", combination_type="dare")  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        head.add_weighted_adapter(["zzz"], [1.0], "x")
    with pytest.raises(ValueError, match="already exists"):
        head.add_weighted_adapter(["a"], [1.0], "a")


@pytest.mark.parametrize("mode,dora", VARIANTS)
def test_two_task_merge_eval(mode, dora):
    cfg = load_config()
    cfg["lora"]["scaling"] = mode
    cfg["lora"]["use_dora"] = dora
    rep = run_multi_adapter_eval(cfg)
    rows = {r["adapter"]: r for r in rep["rows"]}
    chance = rep["chance"]
    # single adapters solve only their own task
    assert rows["A"]["acc_task_A"] > 0.8 and rows["A"]["acc_task_B"] < chance
    assert rows["B"]["acc_task_B"] > 0.8 and rows["B"]["acc_task_A"] < chance
    # merged adapters beat chance on BOTH tasks
    for name in ("linear", "cat", "ties"):
        assert rows[name]["acc_task_A"] > chance and rows[name]["acc_task_B"] > chance, name
    # negation (A+B) − B forgets task B again
    assert rows["linear_minus_B"]["acc_task_B"] < chance
    assert max(rep["single_adapter_invariant_max_abs_logit"].values()) < 1e-9
    assert rep["cat_vs_sum_deltaW_max_abs"] < 1e-10
