#!/usr/bin/env python3
"""Quickstart: train tiny LoRA, print before/after, then merge_and_unload."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_lora.config import load_config
from tiny_lora.eval import format_report, run_before_after


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    report = run_before_after(cfg)
    print(format_report(report))
    print()
    print(
        f"Adapter fraction: "
        f"{report.trainable_params}/{report.trainable_params + report.frozen_params} "
        f"params trainable"
    )
    print()
    print(
        "merge_and_unload lesson: always assign the return value "
        "(model = model.merge_and_unload()) — same footgun as peft#2032. "
        "This demo is numpy-only; not Hugging Face peft."
    )
    if report.adapter_vs_merged_max_abs_logit is not None:
        print(
            f"adapter vs merged max|Δlogit| = "
            f"{report.adapter_vs_merged_max_abs_logit:.2e}  mode={report.mode}"
        )


if __name__ == "__main__":
    main()
