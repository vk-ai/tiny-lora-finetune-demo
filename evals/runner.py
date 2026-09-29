#!/usr/bin/env python3
"""Eval CLI — before vs after LoRA metrics (CI-friendly) + optional rank sweep."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_lora.config import load_config
from tiny_lora.eval import format_report, run_before_after
from tiny_lora.merge_eval import format_multi_adapter_table, run_multi_adapter_eval


def _run_one(cfg: dict) -> dict:
    report = run_before_after(cfg)
    print(format_report(report))
    return report.to_dict()


def _sweep(cfg: dict, ranks: list[int]) -> dict:
    rows = []
    for rank in ranks:
        one = deepcopy(cfg)
        one["lora"]["rank"] = int(rank)
        print(f"\n=== rank sweep: r={rank} scale_mode={one['lora'].get('scaling', 'classic')} ===")
        row = _run_one(one)
        rows.append(
            {
                "rank": int(rank),
                "scale_mode": row["scale_mode"],
                "scaling_value": row["scaling_value"],
                "after_accuracy": row["after"]["accuracy"],
                "after_loss": row["after"]["loss"],
                "before_accuracy": row["before"]["accuracy"],
                "before_loss": row["before"]["loss"],
            }
        )
    print("\nRank sweep summary")
    print(f"{'rank':>6}  {'scale':>8}  {'acc_after':>10}  {'loss_after':>11}")
    for r in rows:
        print(
            f"{r['rank']:6d}  {r['scaling_value']:8.4f}  "
            f"{r['after_accuracy']:10.4f}  {r['after_loss']:11.4f}"
        )
    return {
        "scale_mode": cfg["lora"].get("scaling", "classic"),
        "ranks": ranks,
        "rows": rows,
    }


def _multi_adapter(cfg: dict) -> int:
    reports = []
    for kind, use_dora in (("lora", False), ("dora", True)):
        for mode in ("classic", "rslora"):
            one = deepcopy(cfg)
            one["lora"]["scaling"] = mode
            one["lora"]["use_dora"] = use_dora
            rep = run_multi_adapter_eval(one)
            print("\n" + format_multi_adapter_table(rep))
            reports.append(rep)
    out = ROOT / "evals" / "multi_adapter.json"
    out.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out.relative_to(ROOT)}")
    for rep in reports:
        worst = max(rep["single_adapter_invariant_max_abs_logit"].values())
        if worst > 1e-9:
            print(f"FAIL: merge([A],[1.0]) != A ({rep['adapter_kind']}/"
                  f"{rep['scale_mode']}: {worst:.2e})", file=sys.stderr)
            return 1
        rows = {r["adapter"]: r for r in rep["rows"]}
        for name in ("linear", "cat", "ties"):
            if min(rows[name]["acc_task_A"], rows[name]["acc_task_B"]) <= rep["chance"]:
                print(f"FAIL: merged {name} not above chance on both tasks", file=sys.stderr)
                return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tiny LoRA before/after eval")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.yaml",
        help="YAML config path",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run optional tiny rank sweep (lora.ranks or [2,4,8])",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=None,
        help="Ranks for --sweep (overrides lora.ranks)",
    )
    parser.add_argument(
        "--multi-adapter",
        action="store_true",
        help="Train adapters A/B on two label subsets, merge (linear/cat/ties/negation), "
        "and check merge([A],[1.0]) == A for classic, rsLoRA and DoRA",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.multi_adapter:
        return _multi_adapter(cfg)
    if args.sweep:
        ranks = args.ranks or cfg["lora"].get("ranks") or [2, 4, 8]
        payload = _sweep(cfg, [int(r) for r in ranks])
        out = ROOT / "evals" / "rank_sweep.json"
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {out.relative_to(ROOT)}")
        # Gate: every sweep row should not regress vs its own before
        for row in payload["rows"]:
            if row["after_accuracy"] < row["before_accuracy"]:
                print(f"FAIL: rank {row['rank']} accuracy regressed", file=sys.stderr)
                return 1
            if row["after_loss"] > row["before_loss"] + 1e-6:
                print(f"FAIL: rank {row['rank']} loss regressed", file=sys.stderr)
                return 1
        return 0

    report = run_before_after(cfg)
    print(format_report(report))
    payload = report.to_dict()
    out = ROOT / "evals" / "last_report.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out.relative_to(ROOT)}")
    if payload["scale_mode"] not in ("classic", "rslora"):
        print("FAIL: missing/invalid scale_mode", file=sys.stderr)
        return 1
    if report.after.accuracy < report.before.accuracy:
        print("FAIL: after accuracy worse than before", file=sys.stderr)
        return 1
    if report.after.loss > report.before.loss + 1e-6:
        print("FAIL: after loss worse than before", file=sys.stderr)
        return 1
    if payload.get("mode") == "merged":
        max_abs = payload.get("adapter_vs_merged_max_abs_logit")
        if max_abs is None or max_abs > 1e-6:
            print(f"FAIL: adapter vs merged logits diverge ({max_abs})", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
