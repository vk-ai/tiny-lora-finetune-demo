"""Before/after eval harness."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tiny_lora.config import load_config
from tiny_lora.eval import format_report, run_before_after

ROOT = Path(__file__).resolve().parents[1]


def test_before_after_harness():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg = {
        **cfg,
        "data": {**cfg["data"], "n_train": 96, "n_test": 32},
        "train": {**cfg["train"], "epochs": 20},
    }
    report = run_before_after(cfg)
    assert report.after.accuracy >= report.before.accuracy
    assert report.trainable_params < report.frozen_params
    assert report.lora_rank == cfg["lora"]["rank"]
    assert report.scale_mode == "classic"
    text = format_report(report)
    assert "before" in text and "after" in text
    d = report.to_dict()
    assert "accuracy_delta" in d
    assert d["scale_mode"] == "classic"


def test_report_records_scale_mode_rslora():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg = {
        **cfg,
        "data": {**cfg["data"], "n_train": 64, "n_test": 32},
        "train": {**cfg["train"], "epochs": 15},
        "lora": {**cfg["lora"], "scaling": "rslora", "rank": 4, "alpha": 8.0},
    }
    report = run_before_after(cfg)
    d = report.to_dict()
    assert d["scale_mode"] == "rslora"
    assert d["scaling_value"] == pytest.approx(8.0 / 2.0)
    assert report.after.accuracy >= report.before.accuracy


def test_rank_sweep_runner():
    spec = importlib.util.spec_from_file_location(
        "tiny_lora_eval_runner", ROOT / "evals" / "runner.py"
    )
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg = {
        **cfg,
        "data": {**cfg["data"], "n_train": 48, "n_test": 24, "n_features": 8},
        "model": {**cfg["model"], "hidden_dim": 8},
        "train": {**cfg["train"], "epochs": 12},
        "lora": {**cfg["lora"], "scaling": "classic", "alpha": 4.0},
    }
    payload = runner._sweep(cfg, [2, 4])
    assert payload["scale_mode"] == "classic"
    assert len(payload["rows"]) == 2
    assert [r["rank"] for r in payload["rows"]] == [2, 4]
    for row in payload["rows"]:
        assert row["scale_mode"] == "classic"
        assert row["after_accuracy"] >= row["before_accuracy"]


def test_before_after_includes_merged_parity():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg = {
        **cfg,
        "data": {**cfg["data"], "n_train": 64, "n_test": 32},
        "train": {**cfg["train"], "epochs": 15},
        "eval": {"merge_and_unload": True},
    }
    report = run_before_after(cfg)
    d = report.to_dict()
    assert d["mode"] == "merged"
    assert "merged" in d
    assert d["adapter_vs_merged_max_abs_logit"] is not None
    assert d["adapter_vs_merged_max_abs_logit"] < 1e-8
    assert report.merged is not None
    assert abs(report.merged.accuracy - report.after.accuracy) < 1e-12
