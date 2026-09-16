"""Eval harness: loss + accuracy before vs after LoRA fine-tune."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .data import Dataset, train_test_split
from .model import TinyClassifier
from .train import build_and_train, train_lora


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
    final_train_loss: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "before": asdict(self.before),
            "after": asdict(self.after),
            "trainable_params": self.trainable_params,
            "frozen_params": self.frozen_params,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "final_train_loss": self.final_train_loss,
            "accuracy_delta": self.after.accuracy - self.before.accuracy,
            "loss_delta": self.after.loss - self.before.loss,
        }


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
    """Full harness: build model, score test, train LoRA, score test again."""
    train, test = train_test_split(
        n_train=int(cfg["data"]["n_train"]),
        n_test=int(cfg["data"]["n_test"]),
        n_features=int(cfg["data"]["n_features"]),
        n_classes=int(cfg["data"]["n_classes"]),
        noise=float(cfg["data"]["noise"]),
        seed=int(cfg["seed"]),
    )
    rng = np.random.default_rng(int(cfg["seed"]))
    model = TinyClassifier.create(
        in_features=int(cfg["data"]["n_features"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        n_classes=int(cfg["data"]["n_classes"]),
        rank=int(cfg["lora"]["rank"]),
        alpha=float(cfg["lora"]["alpha"]),
        rng=rng,
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
    return BeforeAfterReport(
        before=before,
        after=after,
        trainable_params=model.n_trainable(),
        frozen_params=model.n_frozen(),
        lora_rank=int(cfg["lora"]["rank"]),
        lora_alpha=float(cfg["lora"]["alpha"]),
        final_train_loss=losses[-1] if losses else float("nan"),
    )


def format_report(report: BeforeAfterReport) -> str:
    d = report.to_dict()
    lines = [
        "Tiny LoRA before/after eval",
        f"  LoRA rank={report.lora_rank}  alpha={report.lora_alpha}",
        f"  trainable={report.trainable_params}  frozen={report.frozen_params}",
        f"  before  loss={report.before.loss:.4f}  acc={report.before.accuracy:.4f}",
        f"  after   loss={report.after.loss:.4f}  acc={report.after.accuracy:.4f}",
        f"  delta   loss={d['loss_delta']:+.4f}  acc={d['accuracy_delta']:+.4f}",
        f"  final train loss={report.final_train_loss:.4f}",
    ]
    return "\n".join(lines)


# re-export for callers that build via config
__all__ = [
    "Metrics",
    "BeforeAfterReport",
    "evaluate",
    "run_before_after",
    "format_report",
    "build_and_train",
]
