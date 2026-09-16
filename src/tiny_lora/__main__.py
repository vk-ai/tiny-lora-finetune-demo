"""CLI: python -m tiny_lora [--config PATH]."""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .eval import format_report, run_before_after


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tiny LoRA fine-tune demo (OSS learning)")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON report",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    report = run_before_after(cfg)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))
    # Soft success signal: after accuracy should beat before on this toy task.
    return 0 if report.after.accuracy >= report.before.accuracy else 1


if __name__ == "__main__":
    sys.exit(main())
