"""Two-task multi-adapter demo + merge eval table (numpy, CPU, seconds).

Adapter ``A`` learns labels {0, 1} and adapter ``B`` learns labels {2, 3} of one
4-class synthetic blob problem, over the same frozen base. We then merge them
(linear / cat / TIES-lite / negation) and score each merged adapter on both
task subsets with a full 4-way argmax (chance = 0.25).

Numbers are on synthetic blobs, not LLM benchmarks.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .data import Dataset, train_test_split
from .model import TinyClassifier
from .multi_adapter import MultiAdapterHead
from .train import train_lora

TASKS: dict[str, list[int]] = {"A": [0, 1], "B": [2, 3]}
N_CLASSES = 4


def _subset(d: Dataset, labels: list[int]) -> Dataset:
    mask = np.isin(d.y, labels)
    return Dataset(X=d.X[mask], y=d.y[mask])


def two_task_split(cfg: dict[str, Any]) -> tuple[Dataset, Dataset]:
    """The shared 4-class blob train/test split used by every two-task eval."""
    data = cfg["data"]
    return train_test_split(
        n_train=int(data["n_train"]),
        n_test=max(int(data["n_test"]), 128),
        n_features=int(data["n_features"]),
        n_classes=N_CLASSES,
        noise=float(data["noise"]),
        seed=int(cfg["seed"]),
    )


def build_two_task(cfg: dict[str, Any]) -> tuple[TinyClassifier, MultiAdapterHead, Dataset]:
    """Train adapters ``A`` and ``B`` on disjoint label subsets; return (model, head, test)."""
    lora = cfg["lora"]
    data = cfg["data"]
    seed = int(cfg["seed"])
    train, test = two_task_split(cfg)
    rng = np.random.default_rng(seed)
    base = TinyClassifier.create(
        in_features=int(data["n_features"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        n_classes=N_CLASSES,
        rank=int(lora["rank"]),
        alpha=float(lora["alpha"]),
        rng=rng,
        scale_mode=lora.get("scaling", "classic"),
    )
    head = MultiAdapterHead.from_base(base.head.W, base.head.b)
    for i, (name, labels) in enumerate(TASKS.items()):
        adapter = head.add_adapter(
            name,
            rank=int(lora["rank"]),
            alpha=float(lora["alpha"]),
            rng=rng,
            scale_mode=lora.get("scaling", "classic"),
            use_dora=bool(lora.get("use_dora", False)),
        )
        view = TinyClassifier(W1=base.W1, b1=base.b1, head=adapter)
        train_lora(
            view,
            _subset(train, labels),
            epochs=int(cfg["train"]["epochs"]),
            lr=float(cfg["train"]["lr"]),
            batch_size=int(cfg["train"]["batch_size"]),
            l2=float(cfg["train"].get("l2", 0.0)),
            seed=seed + 1 + i,
        )
    model = TinyClassifier(W1=base.W1, b1=base.b1, head=head, adapter_kind="multi")
    return model, head, test


def single_adapter_invariant(
    model: TinyClassifier, head: MultiAdapterHead, X: np.ndarray, name: str = "A"
) -> dict[str, float]:
    """max |Δlogit| between adapter ``name`` and merge([name], [1.0]) per combination."""
    out: dict[str, float] = {}
    head.set_adapter(name)
    ref = model.logits(X)
    for combo in ("linear", "cat", "ties"):
        tmp = f"__inv_{combo}"
        head.add_weighted_adapter([name], [1.0], tmp, combination_type=combo, density=1.0)
        head.set_adapter(tmp)
        out[combo] = float(np.max(np.abs(model.logits(X) - ref)))
        head.delete_adapter(tmp)
    head.set_adapter(name)
    return out


def run_multi_adapter_eval(cfg: dict[str, Any]) -> dict[str, Any]:
    """Train A/B, merge several ways, return the eval table + invariant check."""
    cfg = deepcopy(cfg)
    density = float(cfg.get("multi_adapter", {}).get("density", 0.5))
    model, head, test = build_two_task(cfg)
    head.add_weighted_adapter(["A", "B"], [1.0, 1.0], "linear", combination_type="linear")
    head.add_weighted_adapter(["A", "B"], [1.0, 1.0], "cat", combination_type="cat")
    head.add_weighted_adapter(
        ["A", "B"], [1.0, 1.0], "ties", combination_type="ties", density=density
    )
    # Negative weight = task negation: (A+B) − B removes task B again.
    head.add_weighted_adapter(["linear", "B"], [1.0, -1.0], "linear_minus_B")

    rows = []
    for name in (None, "A", "B", "linear", "cat", "ties", "linear_minus_B"):
        head.set_adapter(name)
        row: dict[str, Any] = {"adapter": name or "base"}
        for task, labels in TASKS.items():
            sub = _subset(test, labels)
            row[f"acc_task_{task}"] = float(np.mean(model.logits(sub.X).argmax(1) == sub.y))
        row["acc_all"] = float(np.mean(model.logits(test.X).argmax(1) == test.y))
        info = head.merge_info.get(name) if name else None
        row["rank"] = head.adapters[name].rank if name else 0
        row["svd_rel_error"] = info.svd_rel_error if info else 0.0
        rows.append(row)
    invariant = single_adapter_invariant(model, head, test.X, "A")
    cat_vs_sum = float(
        np.max(np.abs(head.delta_W("cat") - (head.delta_W("A") + head.delta_W("B"))))
    )
    return {
        "scale_mode": cfg["lora"].get("scaling", "classic"),
        "adapter_kind": "dora" if cfg["lora"].get("use_dora") else "lora",
        "tasks": TASKS,
        "chance": 1.0 / N_CLASSES,
        "ties_density": density,
        "rows": rows,
        "single_adapter_invariant_max_abs_logit": invariant,
        "cat_vs_sum_deltaW_max_abs": cat_vs_sum,
    }


def format_multi_adapter_table(report: dict[str, Any]) -> str:
    lines = [
        f"Multi-adapter merge eval (kind={report['adapter_kind']} "
        f"scale_mode={report['scale_mode']} chance={report['chance']:.2f})",
        f"{'adapter':<16} {'rank':>4} {'task_A':>7} {'task_B':>7} {'all':>7}",
    ]
    for r in report["rows"]:
        lines.append(
            f"{r['adapter']:<16} {r['rank']:>4d} {r['acc_task_A']:>7.3f} "
            f"{r['acc_task_B']:>7.3f} {r['acc_all']:>7.3f}"
        )
    inv = report["single_adapter_invariant_max_abs_logit"]
    lines.append(
        "merge([A],[1.0]) == A  max|Δlogit|: "
        + "  ".join(f"{k}={v:.1e}" for k, v in inv.items())
    )
    lines.append(f"cat ΔW == ΔW_A + ΔW_B  max|Δ|={report['cat_vs_sum_deltaW_max_abs']:.1e}")
    return "\n".join(lines)


__all__ = [
    "TASKS",
    "two_task_split",
    "build_two_task",
    "single_adapter_invariant",
    "run_multi_adapter_eval",
    "format_multi_adapter_table",
]
