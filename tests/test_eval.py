"""Before/after eval harness."""

from __future__ import annotations

from pathlib import Path

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
    text = format_report(report)
    assert "before" in text and "after" in text
    d = report.to_dict()
    assert "accuracy_delta" in d
