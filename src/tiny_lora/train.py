"""SGD train loop over LoRA adapters only (base weights frozen)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .data import Dataset
from .model import TinyClassifier


def _cross_entropy(logits: np.ndarray, y: np.ndarray) -> float:
    z = logits - logits.max(axis=1, keepdims=True)
    log_sum = np.log(np.exp(z).sum(axis=1) + 1e-12)
    return float(np.mean(log_sum - z[np.arange(len(y)), y]))


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def train_lora(
    model: TinyClassifier,
    train: Dataset,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    l2: float = 0.0,
    seed: int = 0,
) -> list[float]:
    """Train only LoRA A/B on the classification head. Returns epoch losses."""
    rng = np.random.default_rng(seed)
    n = len(train)
    losses: list[float] = []
    head = model.head

    for _epoch in range(epochs):
        order = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = train.X[idx]
            yb = train.y[idx]

            h = model.hidden(xb)  # (B, hidden) — frozen path
            logits = head.forward(h)
            probs = _softmax(logits)
            loss = _cross_entropy(logits, yb)
            epoch_loss += loss
            n_batches += 1

            # dL/dlogits (copy — do not mutate probs in place)
            dlogits = probs.copy()
            dlogits[np.arange(len(yb)), yb] -= 1.0
            dlogits /= max(len(yb), 1)

            # logits = base + scale * (h @ A.T @ B.T)
            scale = head.scaling
            hA = h @ head.A.T  # (B, rank)
            dB = scale * (dlogits.T @ hA)  # (out, rank)
            dA = scale * (head.B.T @ dlogits.T @ h)  # (rank, in)

            if l2 > 0:
                dA = dA + l2 * head.A
                dB = dB + l2 * head.B

            head.A -= lr * dA
            head.B -= lr * dB

        losses.append(epoch_loss / max(n_batches, 1))

    return losses


def build_and_train(cfg: dict[str, Any], train: Dataset) -> tuple[TinyClassifier, list[float]]:
    rng = np.random.default_rng(int(cfg["seed"]))
    model = TinyClassifier.create(
        in_features=int(cfg["data"]["n_features"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        n_classes=int(cfg["data"]["n_classes"]),
        rank=int(cfg["lora"]["rank"]),
        alpha=float(cfg["lora"]["alpha"]),
        rng=rng,
        scale_mode=cfg["lora"].get("scaling", "classic"),
    )
    losses = train_lora(
        model,
        train,
        epochs=int(cfg["train"]["epochs"]),
        lr=float(cfg["train"]["lr"]),
        batch_size=int(cfg["train"]["batch_size"]),
        l2=float(cfg["train"].get("l2", 0.0)),
        seed=int(cfg["seed"]) + 1,
    )
    return model, losses
