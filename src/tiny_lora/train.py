"""SGD train loop over LoRA / DoRA adapters only (base weights frozen)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .data import Dataset
from .dora import DoRALinear
from .lora import LoRALinear
from .model import TinyClassifier


def _cross_entropy(logits: np.ndarray, y: np.ndarray) -> float:
    z = logits - logits.max(axis=1, keepdims=True)
    log_sum = np.log(np.exp(z).sum(axis=1) + 1e-12)
    return float(np.mean(log_sum - z[np.arange(len(y)), y]))


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def _train_lora_step(head: LoRALinear, h: np.ndarray, yb: np.ndarray, lr: float, l2: float) -> float:
    logits = head.forward(h)
    probs = _softmax(logits)
    loss = _cross_entropy(logits, yb)
    dlogits = probs.copy()
    dlogits[np.arange(len(yb)), yb] -= 1.0
    dlogits /= max(len(yb), 1)
    scale = head.scaling
    hA = h @ head.A.T
    dB = scale * (dlogits.T @ hA)
    dA = scale * (head.B.T @ dlogits.T @ h)
    if l2 > 0:
        dA = dA + l2 * head.A
        dB = dB + l2 * head.B
    head.A -= lr * dA
    head.B -= lr * dB
    return loss


def _train_dora_step(head: DoRALinear, h: np.ndarray, yb: np.ndarray, lr: float, l2: float) -> float:
    """DoRA step via finite-diff-free analytic grads on effective W path.

    We backprop through y = h @ (m * direction).T where direction = (W+ΔW)/||.||.
    For the toy we treat direction as locally constant w.r.t. A/B for a stable
    teaching update (magnitude + LoRA direction), then refresh — same spirit as
    peft's detached norm in early DoRA implementations.
    """
    direction, _row_norm = head.direction_and_norm()
    # Detach direction for A/B update; m gets exact grad through effective W
    W_eff = head.m[:, None] * direction
    logits = h @ W_eff.T + head.b
    probs = _softmax(logits)
    loss = _cross_entropy(logits, yb)

    dlogits = probs.copy()
    dlogits[np.arange(len(yb)), yb] -= 1.0
    dlogits /= max(len(yb), 1)

    # dW_eff = dlogits.T @ h
    dW_eff = dlogits.T @ h  # (out, in)
    # m grad: sum over in of dW_eff * direction
    dm = np.sum(dW_eff * direction, axis=1)
    # Direction path for A/B (detached norm): d(direction≈W+ΔW) ≈ dW_eff * m / norm
    # Use: dΔW ≈ dW_eff * (m / row_norm) with detached row_norm from forward
    scale = head.scaling
    _, row_norm = head.direction_and_norm()
    dW_prime = dW_eff * (head.m / row_norm)[:, None]
    hA = h @ head.A.T
    # ΔW = scale * B @ A; same structure as LoRA but gradient from dW_prime
    # dB: for each sample contribution via chain — use weight-space grads:
    # dΔW = scale * (dB @ A + B @ dA) → dB = scale * dΔW @ A.T, dA = scale * B.T @ dΔW
    dB = scale * (dW_prime @ head.A.T)
    dA = scale * (head.B.T @ dW_prime)

    if l2 > 0:
        dA = dA + l2 * head.A
        dB = dB + l2 * head.B
        dm = dm + l2 * head.m

    head.A -= lr * dA
    head.B -= lr * dB
    head.m -= lr * dm
    return loss


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
    """Train only adapter params on the classification head. Returns epoch losses."""
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
            h = model.hidden(xb)
            if isinstance(head, DoRALinear):
                loss = _train_dora_step(head, h, yb, lr, l2)
            elif isinstance(head, LoRALinear):
                loss = _train_lora_step(head, h, yb, lr, l2)
            else:
                raise TypeError(f"cannot train merged head ({type(head).__name__})")
            epoch_loss += loss
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))

    return losses


def build_and_train(cfg: dict[str, Any], train: Dataset) -> tuple[TinyClassifier, list[float]]:
    rng = np.random.default_rng(int(cfg["seed"]))
    lora_cfg = cfg["lora"]
    model = TinyClassifier.create(
        in_features=int(cfg["data"]["n_features"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        n_classes=int(cfg["data"]["n_classes"]),
        rank=int(lora_cfg["rank"]),
        alpha=float(lora_cfg["alpha"]),
        rng=rng,
        scale_mode=lora_cfg.get("scaling", "classic"),
        use_dora=bool(lora_cfg.get("use_dora", False)),
        qlora_fake4bit=bool(lora_cfg.get("qlora", False)),
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
