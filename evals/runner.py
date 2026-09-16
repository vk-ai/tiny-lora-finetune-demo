#!/usr/bin/env python3
"""Eval CLI — before vs after LoRA metrics (CI-friendly)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_lora.config import load_config
from tiny_lora.eval import format_report, run_before_after


def main() -> int:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    report = run_before_after(cfg)
    print(format_report(report))
    out = ROOT / "evals" / "last_report.json"
    out.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out.relative_to(ROOT)}")
    # Gate: LoRA should not hurt accuracy on this toy task.
    if report.after.accuracy < report.before.accuracy:
        print("FAIL: after accuracy worse than before", file=sys.stderr)
        return 1
    if report.after.loss > report.before.loss + 1e-6:
        print("FAIL: after loss worse than before", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
