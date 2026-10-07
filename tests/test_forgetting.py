"""Round 5: forgetting / retention check (task A → task B, per adapter variant)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tiny_lora.config import load_config
from tiny_lora.forgetting import (
    DEFAULT_VARIANTS,
    format_forgetting_table,
    intruder_dimensions,
    parse_variant,
    run_forgetting_check,
    run_forgetting_seeds,
    train_full_head,
)
from tiny_lora.merge_eval import TASKS, _subset, two_task_split
from tiny_lora.model import TinyClassifier

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def report(cfg):
    return run_forgetting_check(cfg)


def test_parse_variant():
    assert parse_variant("full_head") == ("full_head", 0)
    assert parse_variant("full_W") == ("full_W", 0)
    assert parse_variant("lora_r4") == ("lora", 4)
    assert parse_variant("rslora_r8") == ("rslora", 8)
    assert parse_variant("dora_r2") == ("dora", 2)
    for bad in ("lora", "lora_r0", "qlora_r4", "lora_rx", "full"):
        with pytest.raises(ValueError):
            parse_variant(bad)


def test_unknown_variant_fails_before_training(cfg):
    with pytest.raises(ValueError):
        run_forgetting_check(cfg, ["lora_r2", "nope"])


def test_two_task_split_matches_round4_data(cfg):
    train, test = two_task_split(cfg)
    assert set(np.unique(train.y)) == {0, 1, 2, 3}
    assert len(test) >= 128


def test_stage1_learns_task_a_only(report):
    s1 = report["stage1"]
    assert s1["acc_task_A"] >= 0.95
    assert s1["acc_task_B"] <= report["chance"]  # never saw classes 2/3


def test_report_rows_and_fields(report):
    assert [r["variant"] for r in report["rows"]] == list(DEFAULT_VARIANTS)
    for r in report["rows"]:
        assert r["acc_task_A_before"] == report["stage1"]["acc_task_A"]
        assert math.isclose(r["forgetting_A"], r["acc_task_A_before"] - r["acc_task_A_after"])
        assert 0.0 <= r["acc_task_A_after"] <= 1.0
        assert r["acc_task_B_after"] > report["chance"]  # every variant learns B
        assert r["epochs_used"] == report["stage2_budget"]["epochs"]
        assert 0 <= r["intruder_dims"] <= 4
    params = {r["variant"]: r["n_trainable"] for r in report["rows"]}
    assert params["full_head"] == 4 * 32 + 4
    assert params["full_W"] == 4 * 32
    assert params["lora_r2"] == 2 * (32 + 4)


def test_deterministic(cfg, report):
    again = run_forgetting_check(cfg)
    assert json.dumps(again, sort_keys=True) == json.dumps(report, sort_keys=True)


def test_full_head_with_bias_forgets_more_than_frozen_bias_control(cfg):
    # The bias is the cheapest way to push classes 0/1 down; freezing it (as LoRA does)
    # keeps task A alive under the same fixed budget. Mean over seeds, not one draw.
    rep = run_forgetting_seeds(cfg, [0, 1, 2], ["full_head", "full_W"])
    rows = {r["variant"]: r for r in rep["rows"]}
    assert rows["full_head"]["forgetting_A"] > rows["full_W"]["forgetting_A"] + 0.3


def test_matched_learning_stops_early_and_forgets_less(cfg):
    fixed = run_forgetting_seeds(cfg, [0, 1, 2], ["full_head", "lora_r4"])
    matched = run_forgetting_seeds(cfg, [0, 1, 2], ["full_head", "lora_r4"], target_b_acc=0.9)
    assert matched["stage2_budget"]["target_b_acc"] == 0.9
    for f, m in zip(fixed["rows"], matched["rows"]):
        assert m["epochs_used"] < f["epochs_used"]
        assert m["forgetting_A"] < f["forgetting_A"]
        assert m["acc_task_B_after"] >= 0.5


def test_seed_aggregate_shape(cfg):
    rep = run_forgetting_seeds(cfg, [3, 4], ["lora_r2"])
    assert rep["seeds"] == [3, 4]
    assert len(rep["per_seed"]) == 2
    (row,) = rep["rows"]
    per = [r["rows"][0]["acc_task_A_after"] for r in rep["per_seed"]]
    assert math.isclose(row["acc_task_A_after"], float(np.mean(per)))
    assert row["acc_task_A_after_min"] == min(per)
    with pytest.raises(ValueError):
        run_forgetting_seeds(cfg, [], ["lora_r2"])


def test_intruder_dimensions():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(4, 32))
    assert intruder_dimensions(W, W)[0] == 0
    assert intruder_dimensions(W, 3.0 * W)[0] == 0  # scaling keeps singular vectors
    # Replace W's row space with an orthogonal one → every vector is an intruder.
    Q, _ = np.linalg.qr(rng.normal(size=(32, 8)))
    _, _, Vt = np.linalg.svd(W, full_matrices=False)
    proj = Q - Vt.T @ (Vt @ Q)
    W_orth = rng.normal(size=(4, 8)) @ proj.T
    count, best = intruder_dimensions(W, W_orth)
    assert count == 4 and max(best) < 1e-6


def test_train_full_head_freezes_bias_when_asked(cfg):
    train, _ = two_task_split(cfg)
    rng = np.random.default_rng(0)
    m = TinyClassifier.create(16, 32, 4, rank=1, alpha=1.0, rng=rng)
    sub = _subset(train, TASKS["B"])
    W0, b0 = m.head.W.copy(), m.head.b.copy()
    W, b, losses = train_full_head(m, W0, b0, sub, epochs=3, lr=0.1, batch_size=32,
                                   train_bias=False)
    assert np.array_equal(b, b0) and not np.array_equal(W, W0)
    assert np.array_equal(m.head.W, W0)  # inputs not mutated
    assert losses[-1] < losses[0]


def test_format_table(cfg, report):
    text = format_forgetting_table(report)
    assert "Forgetting / retention check" in text
    assert "fixed budget" in text
    for v in DEFAULT_VARIANTS:
        assert v in text
    matched = run_forgetting_seeds(cfg, [0], ["lora_r2"], target_b_acc=0.9)
    assert "matched learning" in format_forgetting_table(matched)


def test_runner_forgetting_cli():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "evals" / "runner.py"), "--forgetting", "--seeds", "2",
         "--variants", "full_head", "lora_r2"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "matched learning" in proc.stdout
    data = json.loads((ROOT / "evals" / "forgetting.json").read_text())
    assert set(data) == {"fixed_budget", "matched_learning"}
    assert data["matched_learning"]["stage2_budget"]["target_b_acc"] == 0.9
