"""Forgetting / retention check: sequential task A → task B (numpy, CPU, seconds).

This reuses the round-4 two-task data: one 4-class blob problem where task ``A`` is
labels {0, 1} and task ``B`` is labels {2, 3}.

1. **Stage 1:** fine-tune the full classifier head on task A only. That head
   becomes the frozen "pretrained" base ``W_A, b_A`` (W1 stays frozen throughout).
2. **Stage 2:** train each *variant* on task B only, starting from that same base
   and with the same epochs / lr / batch size:

   - ``full_head``: update all of ``W`` and ``b`` (full fine-tune of the head)
   - ``full_W``: update ``W`` only, with ``b`` frozen (a control: LoRA freezes the
     bias too, so this separates "low rank" from "can't move the bias")
   - ``lora_r{r}``: LoRA ``ΔW = s·B·A`` on the frozen base (``b`` stays frozen)
   - ``rslora_r{r}`` / ``dora_r{r}``: the rsLoRA and DoRA variants from earlier rounds

3. **Report**, per variant: task-A accuracy before and after stage 2, the forgetting
   (before − after), the retention ratio (after / before), task-B accuracy after,
   trainable parameters, ‖ΔW‖_F, and an **intruder-dimension** count. That count
   is the number of right singular vectors of the stage-2 weight whose best
   |cosine| with any singular vector of ``W_A`` is below ``intruder_eps``
   (arXiv 2410.21228).

All accuracies use a full 4-way argmax (chance = 0.25). Training on task B alone
pushes the logits of classes 0/1 down, so *some* forgetting is expected for
every variant. The question this answers is "how much, per variant", the
"LoRA learns less and forgets less" trade-off
(https://github.com/huggingface/peft/issues/2907). It is also the stage-1 metric
the peft maintainers suggest monitoring when you train LoRA twice
(https://github.com/huggingface/peft/issues/2873).

Caveat: the head has only 4 outputs, so rank ≥ 4 is already a full-rank ΔW. The
differences between r=4, r=8 and ``full_head`` come from optimisation dynamics
(B starts at zero, the α/r scaling) and the frozen bias, not from extra capacity.
These are synthetic blobs, not LLM benchmarks.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import numpy as np

from .data import Dataset
from .dora import DoRALinear
from .lora import LoRALinear, MergedLinear
from .merge_eval import N_CLASSES, TASKS, _subset, two_task_split
from .model import TinyClassifier
from .train import _cross_entropy, _softmax, train_lora

DEFAULT_VARIANTS: tuple[str, ...] = (
    "full_head",
    "full_W",
    "lora_r2",
    "lora_r4",
    "lora_r8",
    "rslora_r8",
    "dora_r4",
)
_KINDS = ("lora", "rslora", "dora")


def parse_variant(name: str) -> tuple[str, int]:
    """``"lora_r4"`` → ``("lora", 4)``; ``"full_head"`` → ``("full_head", 0)``."""
    if name in ("full_head", "full_W"):
        return name, 0
    kind, sep, rank = name.partition("_r")
    if not sep or kind not in _KINDS or not rank.isdigit() or int(rank) < 1:
        raise ValueError(
            f"unknown variant {name!r}; use 'full_head', 'full_W' or '<lora|rslora|dora>_r<rank>'"
        )
    return kind, int(rank)


def train_full_head(
    model: TinyClassifier,
    W: np.ndarray,
    b: np.ndarray,
    train: Dataset,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    l2: float = 0.0,
    seed: int = 0,
    train_bias: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Plain SGD over the head ``W`` (and ``b`` unless ``train_bias=False``); W1 frozen.

    Returns copies ``(W, b, losses)``.
    """
    W = np.array(W, dtype=np.float64, copy=True)
    b = np.array(b, dtype=np.float64, copy=True)
    rng = np.random.default_rng(seed)
    n = len(train)
    losses: list[float] = []
    for _epoch in range(epochs):
        order = rng.permutation(n)
        total, batches = 0.0, 0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            h = model.hidden(train.X[idx])
            yb = train.y[idx]
            logits = h @ W.T + b
            total += _cross_entropy(logits, yb)
            batches += 1
            d = _softmax(logits)
            d[np.arange(len(yb)), yb] -= 1.0
            d /= max(len(yb), 1)
            dW = d.T @ h + l2 * W
            db = d.sum(axis=0)
            W -= lr * dW
            if train_bias:
                b -= lr * db
        losses.append(total / max(batches, 1))
    return W, b, losses


def _accuracy(model: TinyClassifier, d: Dataset) -> float:
    if len(d) == 0:
        return float("nan")
    return float(np.mean(model.predict(d.X) == d.y))


def intruder_dimensions(
    W_base: np.ndarray, W_new: np.ndarray, eps: float = 0.5
) -> tuple[int, list[float]]:
    """Count right singular vectors of ``W_new`` that are far from all of ``W_base``'s.

    For each right singular vector vⱼ of ``W_new`` (top ``min(shape)``), take
    maxᵢ |cos(vⱼ, uᵢ)| over the right singular vectors uᵢ of ``W_base``. It is an
    *intruder* when that best match is below ``eps`` (arXiv 2410.21228).
    Returns ``(count, best_match_per_vector)``.
    """
    _, _, Vb = np.linalg.svd(np.asarray(W_base, dtype=np.float64), full_matrices=False)
    _, _, Vn = np.linalg.svd(np.asarray(W_new, dtype=np.float64), full_matrices=False)
    sims = np.abs(Vn @ Vb.T)  # rows already unit-norm
    best = sims.max(axis=1)
    return int(np.sum(best < eps)), [float(x) for x in best]


def run_forgetting_check(
    cfg: dict[str, Any],
    variants: Sequence[str] = DEFAULT_VARIANTS,
    *,
    intruder_eps: float = 0.5,
    target_b_acc: float | None = None,
) -> dict[str, Any]:
    """Stage 1 (full head on A) → stage 2 (each variant on B); return the report.

    ``target_b_acc=None`` gives every variant the same fixed budget (``train.epochs``).
    With a float, each variant stops after the first epoch whose task-B *train*
    accuracy reaches it (still capped at ``train.epochs``). That is the "matched
    learning" view: how much of A is lost per unit of B learned.
    """
    cfg = deepcopy(cfg)
    variants = list(variants)
    parsed = [parse_variant(v) for v in variants]  # validate before training
    seed = int(cfg["seed"])
    tr = cfg["train"]
    epochs, lr, bs = int(tr["epochs"]), float(tr["lr"]), int(tr["batch_size"])
    l2 = float(tr.get("l2", 0.0))
    alpha = float(cfg["lora"]["alpha"])

    train, test = two_task_split(cfg)
    train_a, train_b = _subset(train, TASKS["A"]), _subset(train, TASKS["B"])
    test_a, test_b = _subset(test, TASKS["A"]), _subset(test, TASKS["B"])

    rng = np.random.default_rng(seed)
    init = TinyClassifier.create(
        in_features=int(cfg["data"]["n_features"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        n_classes=N_CLASSES,
        rank=1,
        alpha=1.0,
        rng=rng,
    )
    W_A, b_A, _ = train_full_head(
        init, init.head.W, init.head.b, train_a,
        epochs=epochs, lr=lr, batch_size=bs, l2=l2, seed=seed + 1,
    )
    hidden_dim, n_out = W_A.shape[1], W_A.shape[0]

    def _model(head: Any) -> TinyClassifier:
        return TinyClassifier(W1=init.W1, b1=init.b1, head=head, adapter_kind="lora")

    stage1 = _model(MergedLinear(hidden_dim, n_out, W_A.copy(), b_A.copy()))
    acc_a_before = _accuracy(stage1, test_a)
    acc_b_before = _accuracy(stage1, test_b)

    rows: list[dict[str, Any]] = []
    for name, (kind, rank) in zip(variants, parsed):
        vseed = seed + 100 + len(rows)
        if kind in ("full_head", "full_W"):
            W_new, b_new = W_A, b_A
            after = _model(MergedLinear(hidden_dim, n_out, W_new.copy(), b_new.copy()))
            epochs_used = 0
            for ep in range(epochs):
                W_new, b_new, _ = train_full_head(
                    stage1, W_new, b_new, train_b,
                    epochs=1, lr=lr, batch_size=bs, l2=l2, seed=vseed * 1000 + ep,
                    train_bias=kind == "full_head",
                )
                after.head.W, after.head.b = W_new, b_new
                epochs_used = ep + 1
                if target_b_acc is not None and _accuracy(after, train_b) >= target_b_acc:
                    break
            n_trainable = int(W_new.size + (b_new.size if kind == "full_head" else 0))
            scaling = None
        else:
            cls = DoRALinear if kind == "dora" else LoRALinear
            head = cls.create(
                in_features=hidden_dim,
                out_features=n_out,
                rank=rank,
                alpha=alpha,
                rng=np.random.default_rng(vseed),
                scale_mode="rslora" if kind == "rslora" else "classic",
            )
            head.W, head.b = W_A.copy(), b_A.copy()
            if isinstance(head, DoRALinear):
                head.m = np.maximum(np.linalg.norm(head.W, axis=1), 1e-12)
            after = _model(head)
            epochs_used = 0
            for ep in range(epochs):
                train_lora(
                    after, train_b, epochs=1, lr=lr, batch_size=bs, l2=l2,
                    seed=vseed * 1000 + ep,
                )
                epochs_used = ep + 1
                if target_b_acc is not None and _accuracy(after, train_b) >= target_b_acc:
                    break
            W_new = head.effective_W()
            n_trainable = int(head.n_trainable())
            scaling = float(head.scaling)
        acc_a_after = _accuracy(after, test_a)
        n_intruders, best = intruder_dimensions(W_A, W_new, intruder_eps)
        rows.append(
            {
                "variant": name,
                "kind": kind,
                "rank": rank,
                "scaling": scaling,
                "n_trainable": n_trainable,
                "epochs_used": epochs_used,
                "acc_task_A_before": acc_a_before,
                "acc_task_A_after": acc_a_after,
                "forgetting_A": acc_a_before - acc_a_after,
                "retention_A": acc_a_after / acc_a_before if acc_a_before > 0 else float("nan"),
                "acc_task_B_after": _accuracy(after, test_b),
                "acc_all_after": _accuracy(after, test),
                "delta_W_fro": float(np.linalg.norm(W_new - W_A)),
                "intruder_dims": n_intruders,
                "intruder_best_match": best,
            }
        )
    return {
        "tasks": TASKS,
        "chance": 1.0 / N_CLASSES,
        "stage1": {
            "trained": "full head on task A",
            "acc_task_A": acc_a_before,
            "acc_task_B": acc_b_before,
        },
        "stage2_budget": {
            "epochs": epochs,
            "lr": lr,
            "batch_size": bs,
            "l2": l2,
            "alpha": alpha,
            "target_b_acc": target_b_acc,
        },
        "intruder_eps": float(intruder_eps),
        "rows": rows,
    }


_MEAN_KEYS = (
    "epochs_used",
    "acc_task_A_before",
    "acc_task_A_after",
    "forgetting_A",
    "retention_A",
    "acc_task_B_after",
    "acc_all_after",
    "delta_W_fro",
    "intruder_dims",
)


def run_forgetting_seeds(
    cfg: dict[str, Any],
    seeds: Sequence[int],
    variants: Sequence[str] = DEFAULT_VARIANTS,
    *,
    intruder_eps: float = 0.5,
    target_b_acc: float | None = None,
) -> dict[str, Any]:
    """:func:`run_forgetting_check` over several seeds; rows hold per-variant means.

    Tiny synthetic runs are noisy, so the table reports the mean over ``seeds`` plus
    the worst-seed task-A accuracy (``acc_task_A_after_min``). Per-seed reports are
    kept under ``per_seed``.
    """
    seeds = [int(s) for s in seeds]
    if not seeds:
        raise ValueError("need at least one seed")
    per_seed = []
    for sd in seeds:
        one = deepcopy(cfg)
        one["seed"] = sd
        per_seed.append(
            run_forgetting_check(
                one, variants, intruder_eps=intruder_eps, target_b_acc=target_b_acc
            )
        )
    rows = []
    for i, first in enumerate(per_seed[0]["rows"]):
        runs = [rep["rows"][i] for rep in per_seed]
        row = {k: first[k] for k in ("variant", "kind", "rank", "scaling", "n_trainable")}
        for k in _MEAN_KEYS:
            row[k] = float(np.mean([r[k] for r in runs]))
        row["acc_task_A_after_min"] = float(min(r["acc_task_A_after"] for r in runs))
        rows.append(row)
    return {
        "tasks": TASKS,
        "chance": 1.0 / N_CLASSES,
        "seeds": seeds,
        "stage1": {
            "trained": "full head on task A",
            "acc_task_A": float(np.mean([r["stage1"]["acc_task_A"] for r in per_seed])),
            "acc_task_B": float(np.mean([r["stage1"]["acc_task_B"] for r in per_seed])),
            "acc_task_A_min": float(min(r["stage1"]["acc_task_A"] for r in per_seed)),
        },
        "stage2_budget": per_seed[0]["stage2_budget"],
        "intruder_eps": float(intruder_eps),
        "rows": rows,
        "per_seed": per_seed,
    }


def format_forgetting_table(report: dict[str, Any]) -> str:
    s1 = report["stage1"]
    budget = report["stage2_budget"]
    tgt = budget.get("target_b_acc")
    mode = (
        f"matched learning: stop at task-B train acc >= {tgt:.2f} (max {budget['epochs']} epochs)"
        if tgt is not None
        else f"fixed budget: {budget['epochs']} epochs, lr={budget['lr']}"
    )
    seeds = report.get("seeds")
    lines = [
        "Forgetting / retention check (stage 1: full head on task A; stage 2: each variant on task B)",
        f"stage 2 {mode}" + (f"; mean over seeds {seeds}" if seeds else ""),
        f"stage 1: acc_A={s1['acc_task_A']:.3f} acc_B={s1['acc_task_B']:.3f} "
        f"(4-way argmax, chance={report['chance']:.2f})",
        f"{'variant':<11} {'params':>6} {'epochs':>6} {'A_before':>8} {'A_after':>7} "
        f"{'A_min':>6} {'forget':>7} {'retain':>6} {'B_after':>7} {'|dW|F':>6} {'intrud':>6}",
    ]
    for r in report["rows"]:
        a_min = r.get("acc_task_A_after_min", r["acc_task_A_after"])
        lines.append(
            f"{r['variant']:<11} {r['n_trainable']:>6d} {r['epochs_used']:>6.1f} "
            f"{r['acc_task_A_before']:>8.3f} {r['acc_task_A_after']:>7.3f} {a_min:>6.3f} "
            f"{r['forgetting_A']:>+7.3f} {r['retention_A']:>6.2f} "
            f"{r['acc_task_B_after']:>7.3f} {r['delta_W_fro']:>6.2f} {r['intruder_dims']:>6.2f}"
        )
    lines.append(
        f"intrud = right singular vectors of W' with max|cos| < {report['intruder_eps']} "
        "vs W_A (arXiv 2410.21228)"
    )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_VARIANTS",
    "parse_variant",
    "train_full_head",
    "intruder_dimensions",
    "run_forgetting_check",
    "run_forgetting_seeds",
    "format_forgetting_table",
]
