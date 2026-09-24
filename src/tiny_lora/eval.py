"""Eval harness: loss + accuracy before vs after LoRA fine-tune (+ optional merge)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .data import Dataset, train_test_split
from .lora import ScaleMode, lora_scale
from .model import TinyClassifier
from .train import train_lora


@dataclass
class Metrics:
    loss: float
    accuracy: float
    n_samples: int


@dataclass
class BeforeAfterReport:
    before: Metrics
    after: Metrics
    trainable_params: int
    frozen_params: int
    lora_rank: int
    lora_alpha: float
    scale_mode: str
    scaling_value: float
    final_train_loss: float
    mode: str = "adapter"
    merged: Metrics | None = None
    adapter_vs_merged_max_abs_logit: float | None = None
    adapter_kind: str = "lora"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "before": asdict(self.before),
            "after": asdict(self.after),
            "trainable_params": self.trainable_params,
            "frozen_params": self.frozen_params,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "scale_mode": self.scale_mode,
            "scaling_value": self.scaling_value,
            "final_train_loss": self.final_train_loss,
            "mode": self.mode,
            "adapter_kind": self.adapter_kind,
            "accuracy_delta": self.after.accuracy - self.before.accuracy,
            "loss_delta": self.after.loss - self.before.loss,
        }
        if self.merged is not None:
            d["merged"] = asdict(self.merged)
        if self.adapter_vs_merged_max_abs_logit is not None:
            d["adapter_vs_merged_max_abs_logit"] = self.adapter_vs_merged_max_abs_logit
        return d


def _cross_entropy(logits: np.ndarray, y: np.ndarray) -> float:
    z = logits - logits.max(axis=1, keepdims=True)
    log_sum = np.log(np.exp(z).sum(axis=1) + 1e-12)
    return float(np.mean(log_sum - z[np.arange(len(y)), y]))


def evaluate(model: TinyClassifier, data: Dataset) -> Metrics:
    logits = model.logits(data.X)
    loss = _cross_entropy(logits, data.y)
    preds = logits.argmax(axis=1)
    acc = float(np.mean(preds == data.y))
    return Metrics(loss=loss, accuracy=acc, n_samples=len(data))


def run_before_after(cfg: dict[str, Any]) -> BeforeAfterReport:
    """Full harness: build model, score test, train LoRA, score test again.

    When ``cfg["eval"]["merge_and_unload"]`` is true (default), also merges the
    adapter into base weights and records ``mode: merged`` metrics that must
    match adapter logits within atol (teaching check).
    """
    train, test = train_test_split(
        n_train=int(cfg["data"]["n_train"]),
        n_test=int(cfg["data"]["n_test"]),
        n_features=int(cfg["data"]["n_features"]),
        n_classes=int(cfg["data"]["n_classes"]),
        noise=float(cfg["data"]["noise"]),
        seed=int(cfg["seed"]),
    )
    scale_mode: ScaleMode = cfg["lora"].get("scaling", "classic")  # type: ignore[assignment]
    rank = int(cfg["lora"]["rank"])
    alpha = float(cfg["lora"]["alpha"])
    rng = np.random.default_rng(int(cfg["seed"]))
    use_dora = bool(cfg["lora"].get("use_dora", False))
    qlora = bool(cfg["lora"].get("qlora", False))
    model = TinyClassifier.create(
        in_features=int(cfg["data"]["n_features"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        n_classes=int(cfg["data"]["n_classes"]),
        rank=rank,
        alpha=alpha,
        rng=rng,
        scale_mode=scale_mode,
        use_dora=use_dora,
        qlora_fake4bit=qlora,
    )
    before = evaluate(model, test)
    losses = train_lora(
        model,
        train,
        epochs=int(cfg["train"]["epochs"]),
        lr=float(cfg["train"]["lr"]),
        batch_size=int(cfg["train"]["batch_size"]),
        l2=float(cfg["train"].get("l2", 0.0)),
        seed=int(cfg["seed"]) + 1,
    )
    after = evaluate(model, test)

    do_merge = bool(cfg.get("eval", {}).get("merge_and_unload", True))
    merged_metrics: Metrics | None = None
    max_abs: float | None = None
    mode = "adapter"
    if do_merge:
        adapter_logits = model.logits(test.X)
        # Assign-return lesson: must keep the returned standalone model.
        merged_model = model.merge_and_unload()
        mode = merged_model.mode
        merged_metrics = evaluate(merged_model, test)
        merged_logits = merged_model.logits(test.X)
        max_abs = float(np.max(np.abs(adapter_logits - merged_logits)))

    return BeforeAfterReport(
        before=before,
        after=after,
        trainable_params=model.n_trainable(),
        frozen_params=model.n_frozen(),
        lora_rank=rank,
        lora_alpha=alpha,
        scale_mode=scale_mode,
        scaling_value=lora_scale(alpha, rank, scale_mode),
        final_train_loss=losses[-1] if losses else float("nan"),
        mode=mode,
        merged=merged_metrics,
        adapter_vs_merged_max_abs_logit=max_abs,
        adapter_kind=model.adapter_kind,
    )


def format_report(report: BeforeAfterReport) -> str:
    d = report.to_dict()
    lines = [
        "Tiny LoRA before/after eval",
        f"  LoRA rank={report.lora_rank}  alpha={report.lora_alpha}  "
        f"scale_mode={report.scale_mode}  scale={report.scaling_value:.4f}",
        f"  kind={report.adapter_kind}  trainable={report.trainable_params}  frozen={report.frozen_params}  mode={report.mode}",
        f"  before  loss={report.before.loss:.4f}  acc={report.before.accuracy:.4f}",
        f"  after   loss={report.after.loss:.4f}  acc={report.after.accuracy:.4f}",
        f"  delta   loss={d['loss_delta']:+.4f}  acc={d['accuracy_delta']:+.4f}",
        f"  final train loss={report.final_train_loss:.4f}",
    ]
    if report.merged is not None:
        lines.append(
            f"  merged  loss={report.merged.loss:.4f}  acc={report.merged.accuracy:.4f}  "
            f"(max|Δlogit|={report.adapter_vs_merged_max_abs_logit:.2e})"
        )
    return "\n".join(lines)




def compare_adapters(cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Side-by-side LoRA / DoRA / optional fake-QLoRA teaching table.

    Decision axes (community): start LoRA → QLoRA if memory-bound → DoRA if
    quality-bound at low rank. peft maps: LoraConfig / use_dora=True / bitsandbytes.
    """
    from .config import load_config
    from .qlora import qlora_available, qlora_skip_message

    base = dict(cfg or load_config())
    rows: list[dict[str, Any]] = []
    variants = [
        ("lora", {"use_dora": False, "qlora": False}),
        ("dora", {"use_dora": True, "qlora": False}),
        ("qlora_fake4bit", {"use_dora": False, "qlora": True}),
    ]
    for name, flags in variants:
        c = {**base, "lora": {**base["lora"], **flags}}
        # Keep runs fast for comparison
        c["train"] = {**c["train"], "epochs": min(int(c["train"]["epochs"]), 30)}
        report = run_before_after(c)
        row = {
            "variant": name,
            "trainable_params": report.trainable_params,
            "train_loss": report.final_train_loss,
            "test_acc": report.after.accuracy,
            "test_loss": report.after.loss,
            "adapter_kind": report.adapter_kind,
            "bnb_available": qlora_available(),
        }
        if name.startswith("qlora") and not qlora_available():
            row["note"] = qlora_skip_message()
        rows.append(row)
    return rows


def format_comparison(rows: list[dict[str, Any]]) -> str:
    lines = [
        "LoRA / DoRA / QLoRA-fake comparison (toy)",
        f"{'variant':<16} {'trainable':>10} {'train_loss':>12} {'test_acc':>10}",
    ]
    for r in rows:
        lines.append(
            f"{r['variant']:<16} {r['trainable_params']:>10d} "
            f"{r['train_loss']:>12.4f} {r['test_acc']:>10.4f}"
        )
    return "\n".join(lines)


__all__ = [
    "Metrics",
    "BeforeAfterReport",
    "evaluate",
    "run_before_after",
    "format_report",
    "compare_adapters",
    "format_comparison",
]
